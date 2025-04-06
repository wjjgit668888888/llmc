import functools
import torch
import logging

from llmc.compression.sparsification.attn_utils import _update_causal_mask
from llmc.utils.registry_factory import TOKEN_REDUCTION_REGISTRY
from .token_reduction_module import TokenReductionModule

from loguru import logger


@TOKEN_REDUCTION_REGISTRY.register('FastV')
class FastV(TokenReductionModule):
    def __init__(self, config, model, blocks):
        super().__init__(config, model, blocks)
        logger.info('[FastV] Initializing FastV token reduction module...')
        self.add_sparse_config()
        self.register_reduction_modules()

    def add_sparse_config(self):
        special_config = self.config.get('special', {})
        self.pruning_loc = special_config['pruning_loc']
        logger.info(f"[FastV] pruning_loc set to {self.pruning_loc}")

        special_config['image_token_start_index'] = self.model.pruning_config['image_token_start_index']
        special_config['image_token_length'] = self.model.pruning_config['image_token_length']
        logger.info(f"[FastV] image_token_start_index: {special_config['image_token_start_index']}, "
                    f"image_token_length: {special_config['image_token_length']}")

        special_config['attn_scores'] = None
        self.model.model.parameters = special_config

    def register_reduction_modules(self):
        logger.info("[FastV] Registering forward hooks...")

        def update_output_attentions_hook(module, args, kwargs):
            logger.info(
                f"[FastV] update_output_attentions_hook triggered on {module.__class__.__name__}")
            kwargs['output_attentions'] = True
            return args, kwargs

        def store_attention_hook(m, x, layer_outputs, pruning_pars):
            logger.info(
                f"[FastV] store_attention_hook storing attention from layer {m.__class__.__name__}")
            pruning_pars['attn_scores'] = layer_outputs[1]

        def fastv_pruning_hook(module, args, kwargs, pruning_pars):
            logger.info(
                f"[FastV] fastv_pruning_hook triggered at pruning_loc = {self.pruning_loc}")

            rate = pruning_pars['rate']
            image_token_start_index = pruning_pars['image_token_start_index']
            image_token_length = pruning_pars['image_token_length']

            hidden_states = args[0]
            causal_mask = kwargs['attention_mask']
            cache_position = kwargs['cache_position']
            device = hidden_states.device

            logger.info(f"[FastV] Pruning with rate = {rate}, image_token_start_index = {image_token_start_index}, "
                        f"image_token_length = {image_token_length}")

            last_layer_attention = pruning_pars['attn_scores']
            last_layer_attention_avg = torch.mean(
                last_layer_attention, dim=1)[0]
            last_layer_attention_avg_last_tok = last_layer_attention_avg[-1]
            last_layer_attention_avg_last_tok_image = \
                last_layer_attention_avg_last_tok[image_token_start_index:
                                                  image_token_start_index + image_token_length]

            top_attention_rank_index = \
                last_layer_attention_avg_last_tok_image.topk(
                    round(image_token_length * (1 - rate))).indices + image_token_start_index

            keep_indexs = torch.cat(
                (
                    torch.arange(image_token_start_index, device=device),
                    top_attention_rank_index,
                    torch.arange(image_token_start_index + image_token_length,
                                 hidden_states.shape[1], device=device)
                )
            )

            keep_indexs = keep_indexs.sort().values
            new_seq_length = keep_indexs.shape[0]
            hidden_states = hidden_states[:, keep_indexs, :]

            logger.info(f"[FastV] Pruned sequence length: {new_seq_length}")

            position_ids = keep_indexs.unsqueeze(0)
            causal_mask = _update_causal_mask(
                causal_mask, None, hidden_states, 0) if causal_mask is not None else None

            kwargs['attention_mask'] = causal_mask
            kwargs['cache_position'] = cache_position[:new_seq_length]
            kwargs['position_ids'] = position_ids
            kwargs['position_embeddings'] = None

            pruning_pars['attention_mask'] = causal_mask
            pruning_pars['cache_position'] = cache_position[:new_seq_length]
            pruning_pars['position_ids'] = position_ids
            pruning_pars['position_embeddings'] = None

            logger.info("[FastV] fastv_pruning_hook completed")

            return (hidden_states,), kwargs

        def read_parameter_hook(module, args, kwargs, pruning_pars):
            logger.info(
                f"[FastV] read_parameter_hook injecting pruning parameters into layer {module.__class__.__name__}")
            kwargs['attention_mask'] = pruning_pars['attention_mask']
            kwargs['cache_position'] = pruning_pars['cache_position']
            kwargs['position_ids'] = pruning_pars['position_ids']
            kwargs['position_embeddings'] = pruning_pars['position_embeddings']
            return args, kwargs

        self.blocks[self.pruning_loc - 1].register_forward_pre_hook(
            update_output_attentions_hook,
            with_kwargs=True
        )
        logger.info(
            f"[FastV] Hook: update_output_attentions_hook registered at layer {self.pruning_loc - 1}")

        self.blocks[self.pruning_loc - 1].register_forward_hook(
            functools.partial(store_attention_hook,
                              pruning_pars=self.model.model.parameters),
        )
        logger.info(
            f"[FastV] Hook: store_attention_hook registered at layer {self.pruning_loc - 1}")

        self.blocks[self.pruning_loc].register_forward_pre_hook(
            functools.partial(fastv_pruning_hook,
                              pruning_pars=self.model.model.parameters),
            with_kwargs=True
        )
        logger.info(
            f"[FastV] Hook: fastv_pruning_hook registered at layer {self.pruning_loc}")

        for idx in range(self.pruning_loc + 1, len(self.blocks)):
            self.blocks[idx].register_forward_pre_hook(
                functools.partial(read_parameter_hook,
                                  pruning_pars=self.model.model.parameters),
                with_kwargs=True
            )
            logger.info(
                f"[FastV] Hook: read_parameter_hook registered at layer {idx}")
