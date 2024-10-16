# coding=utf-8
# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/gpt2/modeling_gpt2.py
# Copyright 2023 The vLLM team.
# Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only GPT-2 model compatible with HuggingFace weights."""
from typing import Iterable, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch import nn
from transformers import GPT2Config

from vllm.attention import Attention, AttentionMetadata
from vllm.config import CacheConfig
from vllm.distributed.parallel_state import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.layers.activation import get_act_fn

from vllm.distributed import (get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size,
                              tensor_model_parallel_all_gather,
                              split_tensor_along_first_dim)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.sampler import Sampler, SamplerOutput
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors

from .utils import is_pp_missing_parameter, make_layers, _linear_cls_from_config, _qkv_linear_cls_from_config, _layer_norm_cls_from_config, _residual_cls_from_config

class GPT2Attention(nn.Module):

    def __init__(
        self,
        config: GPT2Config,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        sharding_config: dict = {},
    ):
        super().__init__()
        self.prefix = prefix
        self.hidden_size = config.hidden_size
        total_num_heads = config.num_attention_heads
        tensor_model_parallel_world_size = get_tensor_model_parallel_world_size()
        assert total_num_heads % tensor_model_parallel_world_size == 0
        self.num_heads = total_num_heads // tensor_model_parallel_world_size
        self.head_dim = self.hidden_size // total_num_heads
        self.scale = self.head_dim**-0.5
        self.sharding_config = sharding_config

        # Get sharding types
        c_attn_type = sharding_config.get(f"{prefix}.c_attn", "column")
        attn_type = sharding_config.get(f"{prefix}.attn", "head")
        c_proj_type = sharding_config.get(f"{prefix}.c_proj", "row")

        self.c_attn_type = c_attn_type
        self.c_proj_type = c_proj_type
        
        # Last linear layer of the MLP in the previous GPT2 block
        layer_num = "".join(filter(str.isdigit, prefix))
        previous_layer_num = str(int(layer_num) - 1)
        ln_1_type = sharding_config.get(f"{prefix.replace(layer_num, previous_layer_num).replace('.attn', '')}.ln_1", "replicated")
        res_1_type = sharding_config.get(f"{prefix.replace(layer_num, previous_layer_num).replace('.attn', '')}.res_1", "replicated")

        # c_attn
        c_attn_args = {
            "hidden_size": self.hidden_size,
            "head_size": self.head_dim,
            "total_num_heads": total_num_heads,
            "bias": True,
            "quant_config": quant_config,
            "prefix": f"{prefix}.c_attn",
        }
        if c_attn_type == "column" and attn_type != "head":
            c_attn_args["gather_output"] = True
        if c_attn_type == "row":
            c_attn_args["input_is_parallel"] = False
        if c_attn_type == "data":
            c_attn_args["gather_output"] = True

            if ln_1_type != "data":
                c_attn_args["input_is_parallel"] = False
        self.c_attn = _qkv_linear_cls_from_config(c_attn_type)(
            **c_attn_args,
        )

        # attn
        attn_args = {
            "num_heads": self.num_heads,
            "head_size": self.head_dim,
            "scale": self.scale,
            "cache_config": cache_config,
            "quant_config": quant_config,
            "prefix": f"{prefix}.attn",
        }

        if attn_type == "replicated":
            attn_args["num_heads"] = total_num_heads
        if attn_type == "head" and c_attn_type != "column":
            attn_args["input_is_parallel"] = False
        if attn_type == "head" and c_proj_type != "row":
            attn_args["gather_output"] = True

        self.attn = Attention(
            **attn_args,
        )

        # c_proj
        c_proj_args = {
            "input_size": self.hidden_size,
            "output_size": self.hidden_size,
            "bias": True,
            "quant_config": quant_config,
            "prefix": f"{prefix}.c_proj",
        }
        if c_proj_type == "column":
            c_proj_args["gather_output"] = True
        if c_proj_type == "row" and attn_type != "head":
            c_proj_args["input_is_parallel"] = False
        if c_proj_type == "data":
            c_proj_args["input_is_parallel"] = False

            if res_1_type != "data":
                c_proj_args["gather_output"] = True

        self.c_proj = _linear_cls_from_config(c_proj_type)(
            **c_proj_args,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        num_tokens = attn_metadata.num_prefill_tokens + attn_metadata.num_decode_tokens

        kwargs = {"num_tokens": num_tokens} if self.c_attn_type == "data" else {}
        qkv, _ = self.c_attn(
            hidden_states,
            **kwargs,
        )
        
        q, k, v = qkv.chunk(chunks=3, dim=-1)
        attn_output = self.attn(q, k, v, kv_cache, attn_metadata)

        kwargs = {"num_tokens": num_tokens} if self.c_proj_type == "data" else {}
        attn_output, _ = self.c_proj(
            attn_output,
            **kwargs,
        )

        return attn_output


class GPT2MLP(nn.Module):

    def __init__(
        self,
        intermediate_size: int,
        config: GPT2Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        sharding_config: dict = {},
    ):
        super().__init__()
        hidden_size = config.hidden_size

        self.prefix = prefix

        # Get sharding types
        c_fc_type = sharding_config.get(f"{prefix}.c_fc", "column")
        c_proj_type = sharding_config.get(f"{prefix}.c_proj", "row")
        ln_2_type = sharding_config.get(f"{prefix.replace('.mlp', '')}.ln_2", "replicated")
        res_2_type = sharding_config.get(f"{prefix.replace('.mlp', '')}.res_2", "replicated")
        
        self.c_fc_type = c_fc_type
        self.c_proj_type = c_proj_type
        self.ln_2_type = ln_2_type
        self.res_2_type = res_2_type

        # c_fc
        c_fc_args = {
            "input_size": hidden_size,
            "output_size": intermediate_size,
            "bias": True,
            "quant_config": quant_config,
            "prefix": f"{prefix}.c_fc",
        }

        # Column parallel will generate RS sharding, and only RowParallelLinear
        # can accept that
        if c_fc_type == "column" and c_proj_type != "row":
            c_fc_args["gather_output"] = True
        # Input will never have RS sharding due to LayerNorm
        elif c_fc_type == "row":
            c_fc_args["input_is_parallel"] = False
        elif c_fc_type == "data":
            if c_proj_type != "data":
                c_fc_args["gather_output"] = True

            if ln_2_type != "data":
                c_fc_args["input_is_parallel"] = False

        self.c_fc = _linear_cls_from_config(c_fc_type)(
            **c_fc_args,
        )

        # act
        self.act = get_act_fn(
            config.activation_function,
            quant_config,
            intermediate_size,
        )

        # c_proj
        self.gather_output = False
        c_proj_args = {
            "input_size": intermediate_size,
            "output_size": hidden_size,
            "bias": True,
            "quant_config": quant_config,
            "prefix": f"{prefix}.c_proj",
        }
        # Always need to gather output due to layer norm
        if c_proj_type == "column":
            c_proj_args["gather_output"] = True
        # If previous layer was column parallel, input tensor already has RS sharding
        elif c_proj_type == "row" and c_fc_type != "column":
            c_proj_args["input_is_parallel"] = False
        if c_proj_type == "data":
            if c_fc_type != "data":
                c_proj_args["input_is_parallel"] = False

            # Gather only if the attn.c_fc for the next layer is not data
            if res_2_type != "data":
                self.gather_output = True
                c_proj_args["gather_output"] = True

        self.c_proj = _linear_cls_from_config(c_proj_type)(
            **c_proj_args,
        )

    def forward(
        self, 
        hidden_states: torch.Tensor,
        num_tokens: Optional[int] = None,
    ) -> torch.Tensor:
        kwargs = {"num_tokens": num_tokens} if self.c_fc_type == "data" else {}
        h0, _ = self.c_fc(
            hidden_states,
            **kwargs,
        )

        h1 = self.act(h0)
        
        kwargs = {"num_tokens": num_tokens} if self.c_proj_type == "data" else {}
        h2, _ = self.c_proj(
            h1,
            **kwargs,
        )
        return h2


class GPT2Block(nn.Module):

    def __init__(
        self,
        config: GPT2Config,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        sharding_config: dict = {},
    ):
        super().__init__()
        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

        self.sharding_config = sharding_config
        self.prefix = prefix

        # Last linear layer of the MLP in the last GPT2 block
        layer_num = "".join(filter(str.isdigit, prefix))
        self.is_first_layer = layer_num == "0"
        self.is_last_layer = str(config.n_layer - 1) in prefix
        last_layer_num = str(int(layer_num) - 1)
        next_layer_num = str(int(layer_num) + 1)

        # Get sharding types
        self.prev_layer_res_2 = self.sharding_config.get(f"{self.prefix.replace(layer_num, last_layer_num)}.res_2", "replicated")

        self.ln_1_type = self.sharding_config.get(f"{self.prefix}.ln_1", "replicated")
        self.attn_c_attn_type = self.sharding_config.get(f"{self.prefix}.attn.c_attn", "column")
        self.attn_c_proj_type = self.sharding_config.get(f"{self.prefix}.attn.c_proj", "row")
        self.res_1_type = self.sharding_config.get(f"{self.prefix}.res_1", "replicated")
        self.ln_2_type = self.sharding_config.get(f"{self.prefix}.ln_2", "replicated")
        self.mlp_c_fc_type = self.sharding_config.get(f"{self.prefix}.mlp.c_fc", "column")
        self.mlp_c_proj_type = self.sharding_config.get(f"{self.prefix}.mlp.c_proj", "row")
        self.res_2_type = self.sharding_config.get(f"{self.prefix}.res_2", "replicated")
        
        self.next_layer_ln_1_type = self.sharding_config.get(f"{self.prefix.replace(layer_num, next_layer_num)}.ln_1", "replicated")

        # ln_1
        ln_1_cls = _layer_norm_cls_from_config(self.ln_1_type)
        ln_1_kwargs = {
            "normalized_shape": hidden_size,
            "prefix": f"{prefix}.ln_1",
            "eps": config.layer_norm_epsilon,
        }
        if self.ln_1_type == "data":
            if self.is_first_layer or self.prev_layer_res_2 != "data":
                ln_1_kwargs["input_is_parallel"] = False
            if self.attn_c_attn_type != "data":
                ln_1_kwargs["gather_output"] = True

        self.ln_1 = ln_1_cls(**ln_1_kwargs)

        # attn
        self.attn = GPT2Attention(
            config,
            cache_config,
            quant_config,
            prefix=f"{prefix}.attn",
            sharding_config=sharding_config,
        )

        # res_1
        res_1_cls = _residual_cls_from_config(self.res_1_type)
        res_1_kwargs = {
            "prefix": f"{prefix}.res_1",
            "hidden_size": hidden_size,
        }
        if self.res_1_type == "data":
            # Feedforward is data parallel, residual is replicated
            if self.attn_c_proj_type != "data":
                res_1_kwargs["input_is_parallel"] = False
            if self.is_first_layer or self.prev_layer_res_2 != "data":
                res_1_kwargs["residual_is_parallel"] = False
            if self.ln_2_type != "data":
                res_1_kwargs["gather_output"] = True
        self.res_1 = res_1_cls(**res_1_kwargs)

        # ln_2
        ln_2_cls = _layer_norm_cls_from_config(self.ln_2_type)
        ln_2_kwargs = {
            "normalized_shape": hidden_size,
            "eps": config.layer_norm_epsilon,
            "prefix": f"{prefix}.ln_2",
        }
        if self.ln_2_type == "data":
            if self.res_1_type != "data":
                ln_2_kwargs["input_is_parallel"] = False
            if self.mlp_c_fc_type != "data":
                ln_2_kwargs["gather_output"] = True
        self.ln_2 = ln_2_cls(**ln_2_kwargs)

        # mlp
        self.mlp = GPT2MLP(
            inner_dim,
            config,
            quant_config,
            prefix=f"{prefix}.mlp",
            sharding_config=sharding_config,
        )

        # res_2
        res_2_cls = _residual_cls_from_config(self.res_2_type)
        res_2_kwargs = {
            "prefix": f"{prefix}.res_2",
            "hidden_size": hidden_size,
        }
        if self.res_2_type == "data":
            if self.mlp_c_proj_type != "data":
                res_2_kwargs["input_is_parallel"] = False
            if self.res_1_type != "data":
                res_2_kwargs["residual_is_parallel"] = False
            if self.is_last_layer or self.next_layer_ln_1_type != "data":
                res_2_kwargs["gather_output"] = True

        self.res_2 = res_2_cls(**res_2_kwargs)

        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        num_tokens = attn_metadata.num_prefill_tokens + attn_metadata.num_decode_tokens

        residual = hidden_states
        kwargs = {"num_tokens": num_tokens} if self.ln_1_type == "data" else {}
        hidden_states = self.ln_1(
            hidden_states,
            **kwargs,
            )
        attn_output = self.attn(
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
        )

        kwargs = {"num_tokens": num_tokens} if self.res_1_type == "data" else {}
        hidden_states = self.res_1(
            feedforward=attn_output,
            residual=residual,
            **kwargs,
        )

        residual = hidden_states
        kwargs = {"num_tokens": num_tokens} if self.ln_2_type == "data" else {}
        hidden_states = self.ln_2(
            hidden_states,
            **kwargs,
        )
        
        feed_forward_hidden_states = self.mlp(
            hidden_states,
            num_tokens=num_tokens,
        )

        kwargs = {"num_tokens": num_tokens} if self.res_2_type == "data" else {}
        hidden_states = self.res_2(
            feedforward=feed_forward_hidden_states,
            residual=residual,
            **kwargs,
        )

        return hidden_states


class GPT2Model(nn.Module):

    def __init__(
        self,
        config: GPT2Config,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        sharding_config: dict = {},
    ):
        super().__init__()
        self.config = config
        assert not config.add_cross_attention
        assert not config.scale_attn_by_inverse_layer_idx
        assert not config.reorder_and_upcast_attn
        self.embed_dim = config.hidden_size
        self.wte = VocabParallelEmbedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)
        self.start_layer, self.end_layer, self.h = make_layers(
            config.num_hidden_layers,
            lambda prefix: GPT2Block(
                config,
                cache_config,
                quant_config,
                prefix=prefix,
                sharding_config=sharding_config,
            ),
            prefix=f"{prefix}.h",
        )

        ln_f_type = sharding_config.get(f"{prefix}.ln_f", "replicated")
        final_mlp_c_proj = sharding_config.get(f"{prefix}.h.{self.end_layer}.mlp.c_proj", "row")

        ln_f_cls = _layer_norm_cls_from_config(ln_f_type)
        ln_f_kwargs = {
            "normalized_shape": self.embed_dim,
            "eps": config.layer_norm_epsilon,
            "prefix": f"{prefix}.ln_f",
        }
        if ln_f_type == "data":
            if final_mlp_c_proj != "data":
                ln_f_kwargs["input_is_parallel"] = False
            ln_f_kwargs["gather_output"] = True

        self.ln_f = ln_f_cls(**ln_f_kwargs)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors],
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if get_pp_group().is_first_rank:
            inputs_embeds = self.wte(input_ids)
            position_embeds = self.wpe(position_ids)
            hidden_states = inputs_embeds + position_embeds
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]

        for i in range(self.start_layer, self.end_layer):
            layer = self.h[i]
            hidden_states = layer(
                hidden_states, kv_caches[i - self.start_layer], attn_metadata
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states})

        hidden_states = self.ln_f(hidden_states)
        return hidden_states


class GPT2LMHeadModel(nn.Module, SupportsPP):

    def __init__(
        self,
        config: GPT2Config,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        sharding_config: dict = {},
    ):
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.transformer = GPT2Model(config,
                                     cache_config,
                                     quant_config,
                                     prefix="transformer",
                                     sharding_config=sharding_config,)
        if self.config.tie_word_embeddings:
            self.lm_head = self.transformer.wte
        else:
            self.lm_head = ParallelLMHead(self.config.vocab_size,
                                          self.config.hidden_size)
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.sampler = Sampler()
        self.make_empty_intermediate_tensors = (
            self.transformer.make_empty_intermediate_tensors)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> torch.Tensor:
        hidden_states = self.transformer(
            input_ids, positions, kv_caches, attn_metadata, intermediate_tensors
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states, sampling_metadata)
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def make_empty_intermediate_tensors(
        self, batch_size: int, dtype: torch.dtype, device: torch.device
    ) -> IntermediateTensors:
        return IntermediateTensors(
            {
                "hidden_states": torch.zeros(
                    (batch_size, self.config.hidden_size), dtype=dtype, device=device
                ),
            }
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        for name, loaded_weight in weights:
            if "lm_head.weight" in name:
                # GPT-2 ties the weights of the embedding layer and the final
                # linear layer.
                continue
            if ".attn.bias" in name or ".attn.masked_bias" in name:
                # Skip attention mask.
                # NOTE: "c_attn.bias" should not be skipped.
                continue
            if not name.startswith("transformer."):
                name = "transformer." + name

            if is_pp_missing_parameter(name, self):
                continue

            param = params_dict[name]
            # The HF's GPT-2 implementation uses Conv1D instead of Linear.
            # Because of this, we need to transpose the weights.
            # Note(zhuohan): the logic below might break quantized models.
            for conv1d_weight_name in ["c_attn", "c_proj", "c_fc"]:
                if conv1d_weight_name not in name:
                    continue
                if not name.endswith(".weight"):
                    continue
                loaded_weight = loaded_weight.t()
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
