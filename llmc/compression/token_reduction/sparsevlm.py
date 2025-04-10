import functools
import types
from typing import Callable, Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from llmc.utils.registry_factory import TOKEN_REDUCTION_REGISTRY
from transformers.models.llama.modeling_llama import LlamaFlashAttention2, apply_rotary_pos_emb
from transformers import Cache

from .token_reduction_module import TokenReductionModule


@TOKEN_REDUCTION_REGISTRY.register('SparseVLM')
class SparseVLM(TokenReductionModule):
    def __init__(self, config, model, blocks):
        super().__init__(config, model, blocks)
        self.add_sparse_config()

        total_layers = len(self.blocks)
        if self.pruning_loc < 0 or self.pruning_loc >= total_layers:
            raise ValueError(
                f"pruning_loc {self.pruning_loc} is out of range for model with {total_layers} layers.")

        if self.num_image_tokens is None:
            print(
                "[SparseVLM] Warning: num_image_tokens not set; will attempt to infer at runtime.")

        self._last_pruned_mask = None  # Placeholder for storing attention mask after pruning

        self.register_reduction_modules()
        self.patch_layer()

    def add_sparse_config(self):
        """
        Extract pruning-related parameters from the config.
        """
        special_config = self.config.get('special', {})

        self.pruning_loc = special_config.get('pruning_loc', [2, 6, 15])
        self.tau = special_config.get('pruning_ratio', 0.5)
        self.theta = special_config.get('cluster_ratio', 0.5)
        self.use_adaptive_ratio = special_config.get('adaptive_ratio', False)
        self.num_image_tokens = special_config.get('num_image_tokens', None)

        self.model.model.parameters = special_config

    def patch_layer(self):
        for idx, block in enumerate(self.blocks):
            if idx in self.pruning_loc and len(pre_prompt_length_list) != 0 and hidden_states.shape[1] != 1:
                flash_attn = LlamaDynamicvitFlashAttention2(
                    block.self_attn, layer_idx=idx)
                block.add_module("flash_attn", flash_attn)
                state_dict = block.flash_attn.state_dict()
                for key in block.self_attn.state_dict().keys():
                    if key in state_dict.keys():
                        state_dict[key] = block.self_attn.state_dict()[key]
                block.flash_attn.load_state_dict(state_dict)

                block.original_forward = block.forward
                block.forward = types.MethodType(
                    sparse_LlamaDecoderLayer_forward,
                    block
                )

    def register_reduction_modules(self):
        sorted_pruning_locs = sorted(self.pruning_loc)
        for loc in sorted_pruning_locs:
            self.blocks[loc].register_forward_hook(
                functools.partial(self._prune_forward_hook, layer_idx=loc),
                with_kwargs=True
            )
        total_layers = len(self.blocks)
        for i, loc in enumerate(sorted_pruning_locs):
            start_idx = loc + 1
            if i < len(sorted_pruning_locs) - 1:
                end_idx = sorted_pruning_locs[i + 1]
            else:
                end_idx = total_layers
            for j in range(start_idx, end_idx):
                layer = self.blocks[j]
                layer.register_forward_pre_hook(self._inject_params_hook)

    def _prune_forward_hook(self, module, inputs, output):
        # inputs: tuple of (hidden_states, *args). We assume hidden_states is inputs[0].
        # shape: [B, N, D], where N = num_image_tokens + num_text_tokens.

        hidden_states = inputs[0]
        # Determine number of visual tokens in the sequence
        B, N, D = hidden_states.shape
        if self.num_image_tokens is None:
            # Infer num_image_tokens as the count of initial visual tokens.
            # Here we assume visual tokens come first in the sequence.
            # (If image tokens are placed elsewhere, this logic should be adjusted or provided via config.)
            # We detect by comparing attention mask if available in inputs.
            if len(inputs) > 1 and inputs[1] is not None:
                # If an attention mask is provided, try to infer contiguous image tokens from it (if masked differently).
                # In many VLM implementations, all prompt tokens (image+text) are attended (mask=1), so can't distinguish here.
                # default to None path below if mask doesn't indicate separation.
                inf_num_img = self.num_image_tokens
            else:
                inf_num_img = None
            # Fallback: if cannot deduce, assume that image tokens occupy the beginning until text tokens start.
            # If model has an attribute for num image tokens (like model.vision_num_tokens), it should be used.
            if inf_num_img is None:
                # Heuristic: if model provides token type IDs or known counts, we could use them. Otherwise, cannot infer.
                raise RuntimeError(
                    "[SparseVLM] num_image_tokens is required but could not be inferred from inputs.")
            else:
                self.num_image_tokens = inf_num_img
        V = self.num_image_tokens
        if V <= 0 or V > N:
            # If V is invalid, skip pruning to avoid errors.
            print(
                f"[SparseVLM] Warning: num_image_tokens={V} is invalid for sequence length {N}. Skipping pruning.")
            return output  # return original output unmodified

        # Split visual and text token hidden states
        # visual token states (to be pruned)
        vis_h = hidden_states[:, :V, :]
        # text token states (retain fully)
        text_h = hidden_states[:, V:, :]
        T = text_h.shape[1]  # number of text tokens
        if T == 0 or V == 0:
            # No visual or no text tokens, nothing to prune
            return output

        # **1. Compute importance scores for each visual token**
        # using attention of text "raters" to visual tokens&#8203;:contentReference[oaicite:16]{index=16}.
        # We use the module's attention weights if possible, otherwise recompute from Q,K of this layer.
        try:
            # Attempt to use module's self-attention sublayer weights to compute text->image attention
            # Query projection (assuming Llama-like structure)
            q_proj = module.self_attn.q_proj
            k_proj = module.self_attn.k_proj  # Key projection
        except AttributeError:
            # If the module structure is different, fallback: assume 'module' is the attention layer itself
            q_proj = getattr(module, 'q_proj', None)
            k_proj = getattr(module, 'k_proj', None)
        if q_proj is None or k_proj is None:
            # If unable to get projections, skip pruning (to preserve compatibility).
            print("[SparseVLM] Warning: Could not access Q/K projections for importance scoring. Falling back to no pruning.")
            return output

        # Project hidden states to Q and K space
        # Only compute attention of text queries onto image keys (sparsity in computing full NxN attention)
        # shape [B, T, H*D_h] where H=num_heads, D_h=head_dim
        Q_text = q_proj(text_h)
        K_all = k_proj(hidden_states)  # shape [B, N, H*D_h]
        # Reshape for multi-head attention (assuming q_proj/k_proj combine heads):
        # Hidden size D = H * D_h. We get number of heads from config or module if available.
        H_heads = getattr(module.self_attn, "num_heads", None)
        if H_heads is None:
            # If module doesn't expose num_heads, infer from weight shapes (assuming q_proj out_features = D)
            H_heads = Q_text.shape[-1] // (D //
                                           getattr(module.self_attn, "head_dim", 1))
            # (In HuggingFace Llama, q_proj.out_features = hidden_size, head_dim = hidden_size/num_heads)
        head_dim = Q_text.shape[-1] // H_heads
        # Reshape Q and K for separate heads
        Q_text = Q_text.view(B, T, H_heads, head_dim).transpose(
            1, 2)  # [B, H, T, D_h]
        K_all = K_all.view(B, N, H_heads, head_dim).transpose(
            1, 2)   # [B, H, N, D_h]
        # Compute scaled dot-product attention scores between text queries and all keys
        attn_scores = torch.matmul(
            Q_text, K_all.transpose(-2, -1)) / math.sqrt(head_dim)  # [B, H, T, N]
        # If an attention mask exists (e.g., to prevent attending to padding), apply it here:
        if len(inputs) > 1 and inputs[1] is not None:
            mask = inputs[1]  # shape [B, N] or [B, 1, 1, N]
            # Expand mask to [B, 1, 1, N] if needed
            if mask.dim() == 2:
                mask = mask[:, None, None, :]  # 1 for allowed tokens
            attn_scores = attn_scores.masked_fill(
                mask.unsqueeze(1) == 0, float('-inf'))
        # Softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, H, T, N]
        # Extract average attention weight on each visual token across selected text tokens&#8203;:contentReference[oaicite:17]{index=17}.
        # Use only the first V positions (visual tokens) from N (visual+text).
        vis_attn = attn_weights[..., :V]  # [B, H, T, V]
        # Average over text tokens and heads to get an importance score per visual token
        # (Alternatively, summing yields similar ranking since sum = const * mean when const = H*T)
        # shape [B, V], importance scores for each visual token
        P = vis_attn.mean(dim=-2).mean(dim=-2)

        # **(Optional) Adaptive ratio:** adjust tau based on attention matrix rank (redundancy)&#8203;:contentReference[oaicite:18]{index=18}.
        current_tau = self.tau
        if self.use_adaptive_ratio:
            # Estimate redundancy via attention score matrix rank (e.g., count of eigenvalues above threshold)
            # For simplicity, use the effective rank of the visual attention (averaged over heads).
            # Compute per-sample rank of (T x V) attention (flatten heads):
            P_matrix = vis_attn.mean(
                dim=1)[0] if B == 1 else vis_attn.mean(dim=1)  # [B, T, V]
            # Use SVD to estimate rank (this can be costly; in practice one might use a faster heuristic)
            ranks = []
            for b in range(B):
                # Compute singular values of attention submatrix for sample b
                M = P_matrix[b]  # [T, V]
                try:
                    sv = torch.linalg.svdvals(M)
                except Exception:
                    # If SVD fails (e.g., not implemented for certain device), fall back to full rank
                    r = min(M.shape[0], M.shape[1])
                else:
                    # Count singular values above a small threshold (e.g., 1e-3 of max)
                    thresh = 1e-3 * float(sv.max().item())
                    r = int((sv > thresh).sum().item())
                ranks.append(r)
            avg_rank = sum(ranks) / len(ranks)
            # Adjust tau: e.g., if rank is low (more redundancy), prune more aggressively, and vice versa.
            # Here we linearly interpolate tau between [min_tau, max_tau] based on rank fraction.
            # (These bounds can be tuned; using 0.3 to 0.7 for example)
            min_tau, max_tau = 0.3, 0.7
            current_tau = min_tau + (max_tau - min_tau) * (avg_rank / V)
            current_tau = max(min_tau, min(max_tau, current_tau))
            if B > 1:
                print(
                    f"[SparseVLM] Adaptive ratio (batch avg): rank={avg_rank:.1f}/{V}, tau->{current_tau:.3f}")

        # **2. Select tokens to prune vs. keep.** Determine how many visual tokens to retain.
        # number of visual tokens to keep (after pruning & recycling)
        V_keep = max(1, int(current_tau * V))
        if V_keep >= V:
            # Nothing to prune (tau suggests keeping all)
            self._last_pruned_mask = None  # no change to mask
            return output  # no pruning performed
        # For multi-batch, ensure uniform length across batch for compatibility&#8203;:contentReference[oaicite:19]{index=19}.
        if B > 1:
            # Use the maximum V_keep among all samples so that all outputs have same length.
            # This ensures the batch can be processed as a single tensor without padding.
            # Note: This may keep slightly more tokens for some images than needed (for safety).
            # We also warn that per-sample adaptive pruning is limited in batch mode.
            desired_V_keep = V_keep
            # In adaptive mode, each sample could have its own V_keep, use max to unify:
            if self.use_adaptive_ratio:
                per_sample_keep = [max(1, int(current_tau * V))
                                   for _ in range(B)]
                desired_V_keep = max(per_sample_keep)
            if desired_V_keep != V_keep:
                print(
                    f"[SparseVLM] Multi-batch: using unified V_keep={desired_V_keep} (max among batch) for consistency.")
            V_keep = desired_V_keep

        # Get indices of top-V_keep importance scores P for each sample
        # We'll construct a mask of shape [B, V] marking tokens to keep.
        device = hidden_states.device
        keep_mask = torch.zeros((B, V), dtype=torch.bool, device=device)
        # For each sample in batch, select highest P scores
        vals, idx = torch.topk(P, k=V_keep, dim=1, largest=True, sorted=True)
        # idx is shape [B, V_keep]
        for b in range(B):
            keep_mask[b, idx[b]] = True

        # **3. Token recycling via clustering for pruned tokens**&#8203;:contentReference[oaicite:20]{index=20}&#8203;:contentReference[oaicite:21]{index=21}.
        # We will aggregate pruned tokens (those with lower scores) into compact tokens via density clustering.
        # Determine the set of "recycled" tokens among pruned ones: top-τ% of pruned tokens (by P)&#8203;:contentReference[oaicite:22]{index=22}.
        # Here, we interpret tau as we already have, so recycled set = all tokens selected to keep (since we intend to keep V_keep tokens total).
        # Actually, per SparseVLM, they recycle some pruned tokens (with highest P among pruned) to aggregate. For simplicity, we treat all kept tokens as the "recycled" set.
        # (This aligns with keeping top-V_keep by P and clustering them, effectively similar outcome.)
        # Extract embeddings of tokens to keep (these will serve as initial cluster candidates)
        kept_vis_h = []
        for b in range(B):
            # shape [1, V_keep, D] for each image
            kept_vis_h.append(vis_h[b:b+1, keep_mask[b], :])
        # (If B>1 and we unified V_keep, some samples may have a few lowest-importance tokens marked keep to match length.)
        kept_vis_h = torch.cat(kept_vis_h, dim=0)  # [B, V_keep, D]

        # If all visual tokens are kept (no pruning), just return original output
        if V_keep == V:
            self._last_pruned_mask = None
            # Construct new output by concatenating (kept visual tokens + text tokens)
            new_output = torch.cat([kept_vis_h, text_h], dim=1)
            return new_output

        # Now cluster the kept visual token embeddings to further reduce their number.
        # Target number of cluster centers C = theta * V_keep&#8203;:contentReference[oaicite:23]{index=23}.
        C = max(1, int(self.theta * V_keep))
        C = min(C, V_keep)  # cannot have more centers than tokens
        # Compute pairwise distances (squared L2) among kept token embeddings for clustering
        # We'll do clustering per sample for simplicity.
        new_vis_tokens = []  # list to collect new visual tokens after aggregation
        for b in range(B):
            tokens = kept_vis_h[b]  # [V_keep, D]
            # Compute local density ρ for each token (using kNN Gaussian kernel as per equation (8)&#8203;:contentReference[oaicite:24]{index=24})
            if tokens.shape[0] <= 1:
                # Edge case: if only one token, it's the cluster center by itself
                new_vis_tokens.append(tokens)
                continue
            with torch.no_grad():
                # Compute pairwise distances
                # [V_keep, V_keep, D]
                diff = tokens.unsqueeze(1) - tokens.unsqueeze(0)
                dist_matrix = torch.sum(
                    diff * diff, dim=-1)      # [V_keep, V_keep]
                # Determine k (for kNN density). We use k = V_keep - 1 (all others) for global density.
                # e.g., 5-NN for density (hyperparameter, can adjust)
                k_val = min(5, tokens.shape[0] - 1)
                # Sort distances for each token to find k nearest
                knn_vals, knn_idx = torch.topk(
                    dist_matrix, k=k_val, largest=False)
                # Compute density: ρ_i = exp(-1/k * sum_{j in KNN} dist(i,j))&#8203;:contentReference[oaicite:25]{index=25}
                rho = torch.exp(- (knn_vals.mean(dim=1)))
                # Compute δ: distance to nearest token with higher density&#8203;:contentReference[oaicite:26]{index=26}
                rho_sorted, order = torch.sort(rho, descending=True)
                delta = torch.zeros_like(rho)
                for ii, idx_i in enumerate(order):
                    if ii == 0:
                        # highest density point: δ = max distance to any other point
                        delta[idx_i] = dist_matrix[idx_i].max()
                    else:
                        # distance to the nearest higher-density point
                        # those with higher density
                        higher_indices = order[:ii]
                        d_to_higher = dist_matrix[idx_i, higher_indices]
                        delta[idx_i] = d_to_higher.min()
                # Compute score = ρ * δ:contentReference[oaicite:27]{index=27}
                score = rho * delta
                # Select top C tokens by score as cluster centers
                center_idx = torch.topk(score, k=C, largest=True).indices
            # sort indices for reproducibility (optional)
            center_idx = center_idx.sort().values
            centers = tokens[center_idx]  # [C, D]
            # Assign every token (including centers themselves) to nearest cluster center by cosine similarity&#8203;:contentReference[oaicite:28]{index=28}.
            # Compute cosine similarity between each token and each center
            # Normalize tokens and centers
            norm_tokens = F.normalize(tokens, dim=1)
            norm_centers = F.normalize(centers, dim=1)
            sim_matrix = torch.mm(norm_tokens, norm_centers.t())  # [V_keep, C]
            # cluster index for each token
            assignments = sim_matrix.argmax(dim=1)
            # Aggregate tokens in each cluster: sum embeddings of each cluster&#8203;:contentReference[oaicite:29]{index=29}.
            new_tokens = torch.zeros(
                (C, D), device=tokens.device, dtype=tokens.dtype)
            for c in range(C):
                members = (assignments == c)
                if not torch.any(members):
                    # no token assigned (should not happen as center itself is a member)
                    continue
                # element-wise sum of all tokens in cluster&#8203;:contentReference[oaicite:30]{index=30}
                new_tokens[c] = tokens[members].sum(dim=0)
            new_vis_tokens.append(new_tokens.unsqueeze(0))  # [1, C, D]
        # Concatenate new visual tokens for all batch elements
        # [B, C, D] (may differ per b, but we enforced uniform length if B>1)
        new_vis_tokens = torch.cat(new_vis_tokens, dim=0)

        # **4. Construct the output with pruned and aggregated visual tokens.**
        # The new sequence = [clustered_visual_tokens] + [text_tokens] (text tokens remain unchanged).
        new_hidden_states = torch.cat(
            [new_vis_tokens, text_h], dim=1)  # [B, C + T, D]
        # Update the attention mask for subsequent layers, if any
        if len(inputs) > 1 and inputs[1] is not None:
            # Original mask size [B, N]; we produce new mask [B, C+T]
            if inputs[1].dim() == 2:
                # inputs[1] is [B, N] (binary 0/1 for each token)
                # We know original first V were image tokens, next T text tokens (all ones likely if no padding).
                # New mask will be of length C+T (all ones, since we keep C visual and all T text).
                new_mask = torch.ones(
                    (B, new_hidden_states.shape[1]), dtype=inputs[1].dtype, device=inputs[1].device)
            else:
                # If mask is of higher dimensional (like attention bias mask), regenerate accordingly.
                # (Alternatively, handle broadcasting. Here we simplify to None.)
                new_mask = None
            self._last_pruned_mask = new_mask
        else:
            self._last_pruned_mask = None

        return new_hidden_states  # The modified output will replace the original layer output

    def _inject_params_hook(self, module, inputs):
        """
        Forward pre-hook for layers after the pruning layer. Injects modified parameters (e.g., updated attention mask) 
        into the forward pass of subsequent layers to account for the reduced sequence length.
        """
        if self._last_pruned_mask is not None and isinstance(inputs, tuple) and len(inputs) > 1:
            # Replace the attention mask in inputs tuple with the pruned version
            # Assume inputs = (hidden_states, attention_mask, *args) or similar
            hidden_states = inputs[0]
            orig_mask = inputs[1]
            new_mask = self._last_pruned_mask
            # Ensure new_mask is on the same device and type as orig_mask
            new_mask = new_mask.to(orig_mask.device).type_as(orig_mask)
            # Construct new inputs tuple with updated mask
            new_inputs = (hidden_states, new_mask) + inputs[2:]
            return new_inputs  # this will override the inputs for the layer
        # If no stored mask or mask not in inputs, do nothing (return None means no change)
        return None

    def register_attention_hooks(self):
        """
        仿照 register_reduction_modules 方式注册三个 hook：
        1. 对 self.self_attn.q_proj 注册 store_q_proj_hook；
        2. 对 self.self_attn.k_proj 注册 store_k_proj_hook；
        3. 对 self.self_attn 注册 attn_logits_hook，在 forward 结束后增加 attn_logits 返回值。
        """
        # 注册 q_proj 的 forward hook
        self.self_attn.q_proj.register_forward_hook(store_q_proj_hook)
        # 注册 k_proj 的 forward hook
        self.self_attn.k_proj.register_forward_hook(store_k_proj_hook)
        # 注册 self_attn 的 forward hook
        self.self_attn.register_forward_hook(attn_logits_hook)
    
    def store_q_proj_hook(module, inputs, output):
        """
        对 q_proj 输出注册 Hook，用于保存 query_states：
        输出 shape: [B, L, hidden_dim]
        转换为： [B, num_heads, L, head_dim]
        """
        # 假定 module 所在的父模块具有 num_heads 与 head_dim 属性
        parent = module.__self__  # 注意：这种方法依赖于 q_proj 是模块的属性，具体可能需要调整
        B, L, _ = output.size()
        # 这里假设 q_proj 的输出经过 parent.num_heads 与 parent.head_dim 切分正确
        query_states = output.view(B, L, parent.num_heads, parent.head_dim).transpose(1, 2)
        parent.cached_query_states = query_states.clone()

    def store_k_proj_hook(module, inputs, output):
        """
        对 k_proj 输出注册 Hook，用于保存 key_states：
        输出 shape: [B, L, hidden_dim]
        转换为： [B, num_heads, L, head_dim]
        """
        parent = module.__self__
        B, L, _ = output.size()
        key_states = output.view(B, L, parent.num_key_value_heads, parent.head_dim).transpose(1, 2)
        parent.cached_key_states = key_states.clone()

    def attn_logits_hook(module, inputs, output):
        """
        Hook 函数：在 self_attn 的 forward 结束后计算 attn_logits，并将其既追加到输出 tuple 中，
        同时缓存到模块属性 cached_attn_logits 供后续 decoder 层使用。
        
        假定：
        - module.cached_query_states 和 module.cached_key_states 分别为 [B, num_heads, L, head_dim] 的张量；
        - module.is_causal 指示是否因果注意力。
        """
        # 检查中间变量是否存在
        if not (hasattr(module, "cached_query_states") and hasattr(module, "cached_key_states")):
            return output

        query_states = module.cached_query_states  # [B, num_heads, L, head_dim]
        key_states = module.cached_key_states      # [B, num_heads, L, head_dim]

        scale_factor = 1 / math.sqrt(query_states.shape[-1])
        # 计算原始 attn_logits，形状 [B, num_heads, L, L]
        attn_logits = torch.matmul(query_states, key_states.transpose(-1, -2)) * scale_factor

        # 如果是因果注意力则添加 mask
        if getattr(module, "is_causal", False):
            L, S = query_states.shape[-2], key_states.shape[-2]
            temp_mask = torch.ones((L, S), dtype=torch.bool, device=query_states.device).tril(diagonal=0)
            attn_bias = torch.zeros((L, S), dtype=query_states.dtype, device=query_states.device)
            attn_bias.masked_fill_(~temp_mask, float("-inf"))
            attn_logits = attn_logits + attn_bias

        # Softmax 得到 attention 分布
        attn_logits = torch.softmax(attn_logits, dim=-1)

        # 将 attn_logits 缓存到 self_attn 模块上，后续 decoder 层使用
        module.cached_attn_logits = attn_logits

        # 将 attn_logits 追加到 self_attn 的输出中
        # 这里应该可以不用追加了
        if isinstance(output, tuple):
            # 假定原输出为 (attn_output, attn_weights, present_key_value)
            if len(output) == 3:
                output = output + (attn_logits,)
            else:
                output = output[:3] + (attn_logits,)
        else:
            output = (output, attn_logits)
        return output

def decoder_attn_logits_hook(module, inputs, output):
    """
    Decoder 层的 hook：
    在 decoder forward 返回后，直接使用 layer_outputs[2] 作为 attn_logits，
    然后执行如下后处理：
      1. 调用 attn_postprocess_topk 得到 pred_score_vis, s_flag, relation_vis_text
      2. 构造策略 tensor policy，并对 pre_prompt 部分、question 部分进行调整
      3. 根据策略中稀疏 token 的索引，执行 merge 和 cluster 操作
      4. 更新 layer_outputs 的隐藏状态部分，并更新 position_ids、v_token_num、text_token_start 等信息
    最后，将处理后的结果（以及 attn_logits）作为额外项追加到 decoder 的输出 tuple 中。
    """
    # 假定 decoder forward 返回的输出即 layer_outputs，为 tuple，其中：
    #   layer_outputs[0]: 当前层的 hidden_states (B, L, C)
    #   layer_outputs[1]: 可能的其他信息（如 attn_weights 等）
    #   layer_outputs[2]: attn_logits（直接从 self_attn 的 hook 得到）
    layer_outputs = output

    # 直接用 layer_outputs[2] 作为 attn_logits
    attn_logits = layer_outputs[2]

    # 获取当前层所需的变量（前向中需提前保存到 module 属性上）
    v_token_start        = getattr(module, "v_token_start", 0)
    v_token_num          = getattr(module, "v_token_num", 0)
    text_token_start     = getattr(module, "text_token_start", 0)
    t_token_idx          = getattr(module, "t_token_idx", None)
    layer_idx            = getattr(module, "layer_idx", 0)
    retained_tokens      = getattr(module, "retained_tokens", 192)
    pre_prompt_length_list = getattr(module, "pre_prompt_length_list", [])
    image_shape          = getattr(module, "image_shape", 576)
    hidden_states        = getattr(module, "current_hidden_states", None)  # shape: (B, L, C)
    B                    = getattr(module, "B", None)
    position_ids         = getattr(module, "position_ids", None)

    # 若缺少必要信息，则直接返回附加 attn_logits
    if attn_logits is None or hidden_states is None or B is None or position_ids is None:
        if isinstance(layer_outputs, tuple):
            return layer_outputs + (attn_logits,)
        else:
            return (layer_outputs, attn_logits)

    # 1. 使用 attn_logits（即 layer_outputs[2]）调用后处理函数，得到可视化分数、标志和关系信息
    pred_score_vis, s_flag, relation_vis_text = attn_postprocess_topk(
        attn_logits,
        v_token_start,
        v_token_num,
        text_token_start,
        t_token_idx,
        layer_idx,
        retained_tokens
    )  # 预测分数 pred_score_vis shape: (B, L_v)

    # 2. 构造策略 policy，初始全1；并在 [v_token_start, text_token_start) 区间内赋值为 pred_score_vis
    policy = torch.ones(B, hidden_states.shape[1], dtype=hidden_states.dtype, device=hidden_states.device)
    policy[:, v_token_start:text_token_start] = pred_score_vis.type(dtype=hidden_states.dtype)

    # 3. 针对每个 batch，根据 pre_prompt_length_list 保留 pre prompt 和 question 部分
    for batch in range(len(pre_prompt_length_list)):
        prompt_length = pre_prompt_length_list[batch]
        # 保留 pre prompt 部分
        policy[batch, :prompt_length] = 1
        # 保留 question 部分：假定 question 从 prompt_length 开始，到 prompt_length + image_shape 结束
        text_token_start = prompt_length + image_shape
        policy[batch, text_token_start:] = 1

    # 4. 找出策略中值为 0 的 token 索引（即稀疏 token）
    total_sparse_token_idx = torch.where(policy == 0)[1].unsqueeze(0)  # shape: (1, num_sparse)

    # 5. 根据是否存在稀疏 token进行 merge&cluster 处理
    if s_flag and total_sparse_token_idx.shape[1] > 0:
        # 重新获取稀疏 token索引（可重复，以确保正确性）
        total_sparse_token_idx = torch.where(policy == 0)[1].unsqueeze(0)
        # 从当前层的隐藏状态中选取对应的 token（batch_index_select 为你提供的工具函数）
        total_sparse_token = batch_index_select(layer_outputs[0], total_sparse_token_idx)

        # 第一阶段：找出预测得分为0的 token 索引
        merge_token_idx_stage1 = torch.where(pred_score_vis == 0)[1]
        merge_token_stage1 = relation_vis_text[0][merge_token_idx_stage1]
        merge_token_num_stage1 = int(merge_token_idx_stage1.shape[0] * 0.3) + 1  # Top 30%
        merge_token_stage2_idx = merge_token_stage1.topk(merge_token_num_stage1)[1]

        # 第二阶段：从所有稀疏 token 中选择对应的 token，并进行 merge
        merge_token_stage2 = total_sparse_token[:, merge_token_stage2_idx, :]
        cluster_num = int(merge_token_stage2.shape[1] / 10) + 1
        if cluster_num == 0:
            cluster_num = merge_token_stage2.shape[1]

        merge_sparse_token = cluster_and_merge(merge_token_stage2, cluster_num)

        # 选出策略中值为1的 token
        select_token_idx = torch.where(policy == 1)[1].unsqueeze(0)  # shape: (1, num_selected)
        select_token = batch_index_select(layer_outputs[0], select_token_idx)
        select_vis_token_num = pred_score_vis.sum()

        # 将选中的 token、merge 得到的 token拼接起来
        select_and_merge_token = torch.cat(
            (
                select_token[:, :v_token_start + select_vis_token_num, :],
                merge_sparse_token,
                select_token[:, v_token_start + select_vis_token_num:, :]
            ),
            dim=1
        )
        layer_outputs = (select_and_merge_token, layer_outputs[1])
        # 更新 position_ids（假定其维度可按选取后的 token 数更新）
        position_ids = position_ids[:, :len(select_token_idx[0]) + cluster_num]
        prev_decision = policy
        # 更新 v_token_num，假定其为预测得分之和加上 cluster 数
        v_token_num = pred_score_vis.sum() + cluster_num
        text_token_start = v_token_start + v_token_num
    else:
        # 如果稀疏 token 不存在或 s_flag 为 False，则直接选取策略中值为1的 token
        select_token_idx = torch.where(policy == 1)[1].unsqueeze(0)
        layer_outputs = (batch_index_select(layer_outputs[0], select_token_idx), layer_outputs[1])
        position_ids = position_ids[:, :len(select_token_idx[0])]
        prev_decision = policy
        v_token_num = pred_score_vis.sum()
        text_token_start = v_token_start + v_token_num

    # 最后，将原始 attn_logits（或其它你需要的信息）作为额外输出项追加到最终输出 tuple 中
    new_output = layer_outputs + (attn_logits,)

    return new_output



layer_dict = {2: 0, 6: 1, 15: 2}     #

sparse_token_list_192 = [300, 200, 110]       # 2*576  4*300 10*200  16*110
sparse_token_list_128 = [303, 110, 36]
sparse_token_list_64 = [66, 30, 17]

sparse_token_dict = {
    192: sparse_token_list_192,
    128: sparse_token_list_128,
    64: sparse_token_list_64
}


def attn_postprocess_topk(self_attn_weights, v_token_start, v_token_num, text_token_start, t_token_idx, layer_idx, retained_tokens):
    '''
    self_attn_weights: [B, H, L, L]
    '''
    self_attn_weights = self_attn_weights.mean(1)  # B, L[Q], L[K]

    t_token_idx = t_token_idx[1] + text_token_start
    relation_vis_text = self_attn_weights[:, t_token_idx,
                                          v_token_start: v_token_start+v_token_num]  # B, L2, L1

    relation_vis_text = relation_vis_text.mean(1)  # B, L1

    relation_vis = relation_vis_text
    s_flag = True       # s_flag controls whether token merge is needed.

    sparse_token_list = sparse_token_dict[retained_tokens]

    if v_token_num != 0:
        mask = torch.zeros_like(relation_vis, dtype=bool)
        _, indices = torch.topk(relation_vis, min(
            sparse_token_list[layer_dict[layer_idx]], v_token_num - 1), dim=1)
        mask[0][indices] = 1
    else:
        mask = torch.ones_like(relation_vis_text, dtype=bool)
        s_flag = False
    return mask, s_flag, relation_vis_text


def sparse_LlamaDecoderLayer_forward(
    self,
    hidden_states: torch.Tensor,
    policy=None,
    sparse_layer=None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
    v_token_start=None,
    v_token_num=None,
    t_token_idx=None,
    **kwargs,
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

    residual = hidden_states

    hidden_states = self.input_layernorm(hidden_states)

    # ------------------------ ADD attribute "policy" -----------------------------------
    # Self Attention
    if sparse_layer:   #
        hidden_states, self_attn_weights, present_key_value, relation_vis_text = self.flash_attn(
            hidden_states=hidden_states,
            policy=policy,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            return_logits=True,
            v_token_start=v_token_start,
            v_token_num=v_token_num,
            t_token_idx=t_token_idx,
            **kwargs,
        )
        # ------------------------ ADD Over -----------------------------------

        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # hidden_states = hidden_states.to(device)
        # print(hidden_states,residual)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)
        outputs += (relation_vis_text, )
        return outputs

    else:
        hidden_states, self_attn_weights, present_key_value, attn_logits = self.flash_attn(
            hidden_states=hidden_states,
            policy=policy,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            return_logits=False,
            **kwargs,
        )
        # ------------------------ ADD Over -----------------------------------

        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # hidden_states = hidden_states.to(device)
        # print(hidden_states,residual)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)
        attn_logits = None
        outputs += (attn_logits, )
        return outputs


class LlamaDynamicvitFlashAttention2(LlamaFlashAttention2):
    def __init__(self, base_attn: LlamaFlashAttention2, layer_idx: int = 0):
        super().__init__(base_attn.config)
        self.load_state_dict(base_attn.state_dict())
        self.layer_idx = layer_idx

    def forward(
        self,
        hidden_states: torch.Tensor,        # torch.Size([2, 687, 4096])
        policy=None,
        attention_mask: Optional[torch.LongTensor] = None,  # None
        # torch.Size([1, 687])
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,     # None
        output_attentions: bool = False,        # False
        use_cache: bool = False,
        return_logits=False,
        v_token_start=None,
        v_token_num=None,
        t_token_idx=None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # LlamaFlashAttention2 attention does not support output_attentions
        if "padding_mask" in kwargs:
            attention_mask = kwargs.pop("padding_mask")

        output_attentions = False

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        # Change
        if len(position_ids[0]) == 1:
            position_ids = torch.tensor([[past_key_value.get_usable_length(
                kv_seq_len, self.layer_idx)]], dtype=torch.int64).cuda()
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(
                kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs)

        if return_logits:
            # v_token_start+v_token_num = text_token_start
            t_token_idx = t_token_idx[1] + v_token_start+v_token_num
            # attn_logits  = query_states[:,:,t_token_idx,:] @ key_states.transpose(2,3)[:,:,:,v_token_start:v_token_start+v_token_num]
            L, S = query_states.size(-2), key_states.size(-2)
            scale_factor = 1 / math.sqrt(query_states.size(-1))
            attn_bias = torch.zeros(L, S, dtype=query_states.dtype)
            if self.is_causal:
                assert attention_mask is None
                temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
                attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
                attn_bias.to(query_states.dtype)

            attn_logits = query_states @ key_states.transpose(
                2, 3) * scale_factor
            attn_logits += attn_bias.to(query_states.device)
            attn_logits = torch.softmax(attn_logits, dim=-1)
        else:
            attn_logits = None
        # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
        # to be able to avoid many of these transpose/reshape/view.
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        dropout_rate = self.attention_dropout if self.training else 0.0

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (LlamaRMSNorm handles it correctly)

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)
        if not output_attentions:
            attn_weights = None
        attn_output = self._flash_attention_forward(
            query_states, key_states, value_states, attention_mask, q_len, dropout=dropout_rate
        )

        attn_output = attn_output.reshape(
            bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights, past_key_value, attn_logits

def  batch_index_select(x, idx):

    if len(x.size()) == 4:
        B, H, N, C = x.size()
        N_new = idx.size(1)
        offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
        idx = idx + offset
        out = x.reshape(B*N, H, C)[idx.reshape(-1)].reshape(B, H, N_new, C)
        return out
    elif len(x.size()) == 3:
        # in this condition
        B, N, C = x.size()
        N_new = idx.size(1)
        offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
        idx = idx + offset
        out = x.reshape(B*N, C)[idx.reshape(-1)].reshape(B, N_new, C)
        return out
    elif len(x.size()) == 2:
        B, N = x.size()
        N_new = idx.size(1)
        offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
        idx = idx + offset
        out = x.reshape(B*N)[idx.reshape(-1)].reshape(B, N_new)
        return out
    else:
        raise NotImplementedError