import torch
import torch.nn as nn
import torch.nn.functional as F

layer_dict = {2:0,6:1,15:2}     # 

sparse_token_list_192 = [300,200,110]       # 2*576  4*300 10*200  16*110
sparse_token_list_128 = [303,110,36]
sparse_token_list_64 = [66,30,17]          

sparse_token_dict = {
    192: sparse_token_list_192,
    128: sparse_token_list_128,
    64 : sparse_token_list_64
}

def attn_postprocess_topk(self_attn_weights, v_token_start, v_token_num, text_token_start, t_token_idx, layer_idx,retained_tokens):
    '''
    self_attn_weights: [B, H, L, L]
    '''
    self_attn_weights = self_attn_weights.mean(1) # B, L[Q], L[K]

    t_token_idx = t_token_idx[1] + text_token_start
    relation_vis_text = self_attn_weights[:, t_token_idx , v_token_start: v_token_start+v_token_num] # B, L2, L1

    relation_vis_text = relation_vis_text.mean(1) # B, L1

    relation_vis = relation_vis_text
    s_flag = True       # s_flag controls whether token merge is needed.

    sparse_token_list = sparse_token_dict[retained_tokens]

    if v_token_num != 0:
        mask = torch.zeros_like(relation_vis, dtype=bool)
        _, indices = torch.topk(relation_vis, min(sparse_token_list[layer_dict[layer_idx]], v_token_num - 1), dim=1)
        mask[0][indices] = 1
    else:
        mask = torch.ones_like(relation_vis_text, dtype=bool)
        s_flag = False
    return mask, s_flag, relation_vis_text

if __name__ == "__main__":
    tensor1 = torch.zeros(18, dtype=torch.long)
    tensor2 = torch.tensor([2, 3, 4, 5, 8, 9, 10, 14, 15, 16, 21, 29, 39, 43, 44, 47, 48, 50], dtype=torch.long)

    # 构造元组
    t_token_idx = (tensor1, tensor2)

    #self_attn_weights = torch.rand(1, 32, 663, 663)
    self_attn_weights = torch.rand(1, 32, 396, 396)
    v_token_start = 35
    v_token_num = 576
    text_token_start = 611
    layer_idx = 15  # 这里需要你根据实际情况提供合适的值
    retained_tokens = 192  # 这里需要你根据实际情况提供合适的值
    mask, s_flag, relation_vis_text = attn_postprocess_topk(self_attn_weights, v_token_start, v_token_num, text_token_start, t_token_idx, layer_idx, retained_tokens)
    print(mask.shape)