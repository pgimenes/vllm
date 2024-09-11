# coding=utf-8
# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
"""Inference-only LLaMA model compatible with HuggingFace weights."""
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch
from torch import nn
from transformers import LlamaConfig

from vllm.attention import Attention, AttentionMetadata
from vllm.config import CacheConfig, LoRAConfig
from vllm.distributed import (get_pp_group, get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size)
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (MergedColumnParallelLinear,
                                               MergedDataParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.quantization.compressed_tensors.utils import (
    get_compressed_tensors_cache_scale)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import Sampler, SamplerOutput
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, kv_cache_scales_loader, maybe_remap_kv_scale_name)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors
from vllm.utils import is_hip

from .interfaces import SupportsLoRA
from .utils import PPMissingLayer, is_pp_missing_parameter, make_layers

from .utils import _linear_cls_from_config, _qkv_linear_cls_from_config, _rms_norm_cls_from_config, _residual_cls_from_config

class LlamaMLP(nn.Module):

    def __init__(
        self,
        config: LlamaConfig,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        bias: bool = False,
        prefix: str = "",
        sharding_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.sharding_config = sharding_config

        # layer info
        layer_num = "".join(filter(str.isdigit, prefix))
        self.is_first_layer = layer_num == "0"
        self.is_last_layer = layer_num == str(config.num_hidden_layers - 1)
        next_layer_num = str(int(layer_num) + 1)

        # get sharding configs
        post_attention_layernorm_type = sharding_config.get(
            f"{prefix.replace('.mlp', '')}.post_attention_layernorm", "replicated")
        self.gate_up_proj_type = sharding_config.get(f"{prefix}.gate_up_proj", "column")
        self.down_proj_type = sharding_config.get(f"{prefix}.down_proj", "row")
        next_layer_input_layernorm_type = sharding_config.get(
            f"{prefix.replace(layer_num, next_layer_num).replace('.mlp', '')}.input_layernorm", "replicated")

        # gate_up_proj
        if self.gate_up_proj_type == "column":
            gate_up_proj_cls = MergedColumnParallelLinear
        elif self.gate_up_proj_type == "data":
            gate_up_proj_cls = MergedDataParallelLinear
        else:
            raise ValueError(f"Unsupported gate_up_proj type: {self.gate_up_proj_type}")

        gate_up_proj_kwargs = {
            "input_size": hidden_size,
            "output_sizes": [intermediate_size] * 2,
            "bias": bias,
            "quant_config": quant_config,
            "prefix": f"{prefix}.gate_up_proj",
        }

        if self.gate_up_proj_type == "data":
            if post_attention_layernorm_type != "data":
                gate_up_proj_kwargs["input_is_parallel"] = False
            if self.down_proj_type != "data":
                gate_up_proj_kwargs["gather_output"] = True

        self.gate_up_proj = gate_up_proj_cls(**gate_up_proj_kwargs)
        
        # down_proj
        down_proj_cls = _linear_cls_from_config(self.down_proj_type)
        down_proj_kwargs = {
            "input_size": intermediate_size,
            "output_size": hidden_size,
            "bias": bias,
            "quant_config": quant_config,
            "prefix": f"{prefix}.down_proj",
        }
        if self.down_proj_type == "data":
            if self.gate_up_proj_type != "data":
                down_proj_kwargs["input_is_parallel"] = False
            if self.is_last_layer or next_layer_input_layernorm_type != "data":
                down_proj_kwargs["gather_output"] = True
            
        self.down_proj = down_proj_cls(**down_proj_kwargs)

        # act_fn
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. "
                             "Only silu is supported for now.")
        self.act_fn = SiluAndMul()

    def forward(
        self,
        x,
        num_tokens: Optional[int] = None,
    ):
        kwargs = {"num_tokens": num_tokens} if self.gate_up_proj_type == "data" else {}
        gate_up, _ = self.gate_up_proj(
            x,
            **kwargs,
        )

        x = self.act_fn(gate_up)
        
        kwargs = {"num_tokens": num_tokens} if self.down_proj_type == "data" else {}
        x, _ = self.down_proj(
            x,
            **kwargs,
        )
        return x


class LlamaAttention(nn.Module):

    def __init__(
        self,
        config: LlamaConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        quant_config: Optional[QuantizationConfig] = None,
        bias: bool = False,
        cache_config: Optional[CacheConfig] = None,
        prefix: str = "",
        sharding_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size

        # KV heads
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)

        # MistralConfig has an optional head_dim introduced by Mistral-Nemo
        self.head_dim = getattr(config, "head_dim",
                                self.hidden_size // self.total_num_heads)
        
        # get sharding config for each submodule
        input_layernorm_type = sharding_config.get(f"{prefix.replace('.self_attn', '')}.input_layernorm", "replicated")
        self.qkv_proj_type = sharding_config.get(f"{prefix}.qkv_proj", "column")
        self.attn_type = sharding_config.get(f"{prefix}.attn", "column")
        self.o_proj_type = sharding_config.get(f"{prefix}.o_proj", "row")
        post_attention_layernorm_type = sharding_config.get(f"{prefix.replace('.self_attn', '')}.post_attention_layernorm", "replicated")
        
        # Parameters used for split
        if self.qkv_proj_type == "data":
            self.q_size = self.total_num_heads * self.head_dim
            self.kv_size = self.total_num_kv_heads * self.head_dim
        else:
            self.q_size = self.num_heads * self.head_dim
            self.kv_size = self.num_kv_heads * self.head_dim

        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        # qkv_proj
        qkv_proj_cls = _qkv_linear_cls_from_config(self.qkv_proj_type)
        qkv_proj_kwargs = {
            "hidden_size": hidden_size,
            "head_size": self.head_dim,
            "total_num_heads": self.total_num_heads,
            "total_num_kv_heads": self.total_num_kv_heads,
            "bias": bias,
            "quant_config": quant_config,
            "prefix": f"{prefix}.qkv_proj",
        }
        if self.qkv_proj_type == "data":
            if input_layernorm_type != "data":
                qkv_proj_kwargs["input_is_parallel"] = False

            # todo: check if we can run rotary emb with data parallelism
            qkv_proj_kwargs["gather_output"] = True

        self.qkv_proj = qkv_proj_cls(**qkv_proj_kwargs)

        # rotary_emb
        is_neox_style = True
        if quant_config is not None and quant_config.get_name() == "gguf":
            is_neox_style = False
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
            is_neox_style=is_neox_style,
        )

        # attn
        attn_kwargs = {
            "num_heads": self.num_heads,
            "head_size": self.head_dim,
            "scale": self.scaling,
            "num_kv_heads": self.num_kv_heads,
            "cache_config": cache_config,
            "quant_config": quant_config,
        }
        if self.qkv_proj_type != "column":
            attn_kwargs["input_is_parallel"] = False
        if self.o_proj_type != "row":
            attn_kwargs["gather_output"] = True
        self.attn = Attention(**attn_kwargs)
        
        # o_proj
        o_proj_cls = _linear_cls_from_config(self.o_proj_type)
        o_proj_kwargs = {
            "input_size": self.total_num_heads * self.head_dim,
            "output_size": hidden_size,
            "bias": bias,
            "quant_config": quant_config,
            "prefix": f"{prefix}.o_proj",
        }
        if self.o_proj_type == "data":
            o_proj_kwargs["input_is_parallel"] = False

            if post_attention_layernorm_type != "data":
                o_proj_kwargs["gather_output"] = True
        self.o_proj = o_proj_cls(**o_proj_kwargs)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        num_tokens = attn_metadata.num_prefill_tokens + attn_metadata.num_decode_tokens

        kwargs = {"num_tokens": num_tokens} if self.qkv_proj_type == "data" else {}
        qkv, _ = self.qkv_proj(hidden_states, **kwargs)

        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, kv_cache, attn_metadata)

        kwargs = {"num_tokens": num_tokens} if self.o_proj_type == "data" else {}
        output, _ = self.o_proj(attn_output, **kwargs)
        return output


class LlamaDecoderLayer(nn.Module):

    def __init__(
        self,
        config: LlamaConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        sharding_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        if rope_scaling is not None and getattr(
                config, "original_max_position_embeddings", None):
            rope_scaling["original_max_position_embeddings"] = (
                config.original_max_position_embeddings)
        max_position_embeddings = getattr(config, "max_position_embeddings",
                                          8192)

        self.prefix = prefix
        self.sharding_config = sharding_config

        # layer info
        layer_num = "".join(filter(str.isdigit, prefix))
        self.is_first_layer = layer_num == "0"
        self.is_last_layer = layer_num == str(config.num_hidden_layers - 1)
        last_layer_num = str(int(layer_num) - 1)
        next_layer_num = str(int(layer_num) + 1)

        # get sharding config for each submodule
        last_layer_post_attention_layernorm_type = sharding_config.get(
            f"{prefix.replace(layer_num, last_layer_num)}.post_attention_layernorm", "replicated")

        last_layer_mlp_down_proj_type = sharding_config.get(
            f"{prefix.replace(layer_num, last_layer_num)}.mlp.down_proj", "row")   

        self.last_layer_post_attention_layernorm_type = last_layer_post_attention_layernorm_type
        self.last_layer_mlp_down_proj_type = last_layer_mlp_down_proj_type
             
        self.input_layernorm_type = sharding_config.get(f"{prefix}.input_layernorm", "replicated")
        self.attn_qkv_proj_type = sharding_config.get(f"{prefix}.self_attn.qkv_proj", "column")
        self.attn_o_proj_type = sharding_config.get(f"{prefix}.self_attn.o_proj", "row")
        self.mlp_gate_up_proj_type = sharding_config.get(f"{prefix}.mlp.gate_up_proj", "column")
        self.post_attention_layernorm_type = sharding_config.get(f"{prefix}.post_attention_layernorm", "replicated")

        next_layer_input_layernorm_type = sharding_config.get(
            f"{prefix.replace(layer_num, next_layer_num)}.input_layernorm", "replicated")

        # input_layernorm
        input_layernorm_cls = _rms_norm_cls_from_config(self.input_layernorm_type)
        input_layernorm_kwargs = {
            "hidden_size": config.hidden_size,
            "eps": config.rms_norm_eps,
        }
        if self.input_layernorm_type == "data":
            if self.is_first_layer or last_layer_mlp_down_proj_type != "data":
                input_layernorm_kwargs["input_is_parallel"] = False
            if self.is_first_layer or last_layer_post_attention_layernorm_type != "data":
                input_layernorm_kwargs["residual_is_parallel"] = False
            if self.attn_qkv_proj_type != "data":
                input_layernorm_kwargs["gather_output"] = True
            if self.post_attention_layernorm_type != "data":
                input_layernorm_kwargs["gather_output_residual"] = True
        self.input_layernorm = input_layernorm_cls(**input_layernorm_kwargs)
        
        # attention
        # Support abacusai/Smaug-72B-v0.1 with attention_bias
        # Support internlm/internlm-7b with bias
        attention_bias = getattr(config, "attention_bias", False) or getattr(
            config, "bias", False)
        self.self_attn = LlamaAttention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=getattr(config, "num_key_value_heads",
                                    config.num_attention_heads),
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            bias=attention_bias,
            cache_config=cache_config,
            prefix=f"{prefix}.self_attn",
            sharding_config=sharding_config,
        )

        # post_attention_layernorm
        post_attention_layernorm_cls = _rms_norm_cls_from_config(self.post_attention_layernorm_type)
        post_attention_layernorm_kwargs = {
            "hidden_size": config.hidden_size,
            "eps": config.rms_norm_eps,
        }
        if self.post_attention_layernorm_type == "data":
            if self.attn_o_proj_type != "data":
                post_attention_layernorm_kwargs["input_is_parallel"] = False
            if self.is_first_layer or self.input_layernorm_type != "data":
                post_attention_layernorm_kwargs["residual_is_parallel"] = False
            if self.mlp_gate_up_proj_type != "data":
                post_attention_layernorm_kwargs["gather_output"] = True
            if self.is_last_layer or next_layer_input_layernorm_type != "data":
                post_attention_layernorm_kwargs["gather_output_residual"] = True

        self.post_attention_layernorm = post_attention_layernorm_cls(**post_attention_layernorm_kwargs)

        # mlp
        self.mlp = LlamaMLP(
            config = config,
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            bias=getattr(config, "mlp_bias", False),
            prefix=f"{prefix}.mlp",
            sharding_config=sharding_config,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        num_tokens = attn_metadata.num_prefill_tokens + attn_metadata.num_decode_tokens

        # Self Attention
        kwargs = {"num_tokens": num_tokens} if self.input_layernorm_type == "data" else {}
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states, **kwargs)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual, **kwargs)
        
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
        )

        kwargs = {"num_tokens": num_tokens} if self.post_attention_layernorm_type == "data" else {}
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual, **kwargs)
        
        hidden_states = self.mlp(hidden_states, num_tokens = num_tokens)
        
        return hidden_states, residual


class LlamaModel(nn.Module):

    def __init__(
        self,
        config: LlamaConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        lora_config: Optional[LoRAConfig] = None,
        prefix: str = "",
        sharding_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        lora_vocab = (lora_config.lora_extra_vocab_size *
                      (lora_config.max_loras or 1)) if lora_config else 0
        self.vocab_size = config.vocab_size + lora_vocab
        self.org_vocab_size = config.vocab_size
        self.sharding_config = sharding_config
        if get_pp_group().is_first_rank or (config.tie_word_embeddings
                                            and get_pp_group().is_last_rank):
            self.embed_tokens = VocabParallelEmbedding(
                self.vocab_size,
                config.hidden_size,
                org_num_embeddings=config.vocab_size,
                quant_config=quant_config,
            )
        else:
            self.embed_tokens = PPMissingLayer()
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: LlamaDecoderLayer(config=config,
                                             cache_config=cache_config,
                                             quant_config=quant_config,
                                             sharding_config=sharding_config,
                                             prefix=prefix),
            prefix=f"{prefix}.layers")
        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors],
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions,
                hidden_states,
                kv_caches[i - self.start_layer],
                attn_metadata,
                residual,
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class LlamaForCausalLM(nn.Module, SupportsLoRA):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    # LoRA specific attributes
    supported_lora_modules = [
        "qkv_proj", "o_proj", "gate_up_proj", "down_proj", "embed_tokens",
        "lm_head"
    ]
    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings",
    }
    embedding_padding_modules = ["lm_head"]
    bitsandbytes_stacked_params_mapping = {
        # shard_name, weight_name, index
        "q_proj": ("qkv_proj", 0),
        "k_proj": ("qkv_proj", 1),
        "v_proj": ("qkv_proj", 2),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }
    # Mistral/Llama models can also be loaded with --load-format mistral
    # from consolidated.safetensors checkpoints
    mistral_mapping = {
        "layers": "model.layers",
        "attention": "self_attn",
        "wq": "q_proj",
        "wk": "k_proj",
        "wv": "v_proj",
        "wo": "o_proj",
        "attention_norm": "input_layernorm",
        "feed_forward": "mlp",
        "w1": "gate_proj",
        "w2": "down_proj",
        "w3": "up_proj",
        "ffn_norm": "post_attention_layernorm",
        "tok_embeddings": "model.embed_tokens",
        "output": "lm_head",
        "norm": "model.norm"
    }

    def __init__(
        self,
        config: LlamaConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        lora_config: Optional[LoRAConfig] = None,
        sharding_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()

        self.config = config
        self.lora_config = lora_config

        self.model = LlamaModel(config,
                                cache_config,
                                quant_config,
                                lora_config=lora_config,
                                sharding_config=sharding_config,
                                prefix="model")
        if get_pp_group().is_last_rank:
            self.unpadded_vocab_size = config.vocab_size
            if lora_config:
                self.unpadded_vocab_size += lora_config.lora_extra_vocab_size
            self.lm_head = ParallelLMHead(
                self.unpadded_vocab_size,
                config.hidden_size,
                org_num_embeddings=config.vocab_size,
                padding_size=DEFAULT_VOCAB_PADDING_SIZE
                # We need bigger padding if using lora for kernel
                # compatibility
                if not lora_config else lora_config.lora_vocab_padding_size,
                quant_config=quant_config,
            )
            if config.tie_word_embeddings:
                self.lm_head.weight = self.model.embed_tokens.weight

            logit_scale = getattr(config, "logit_scale", 1.0)
            self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                    config.vocab_size,
                                                    logit_scale)
            self.sampler = Sampler()
        else:
            self.lm_head = PPMissingLayer()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        model_output = self.model(input_ids, positions, kv_caches,
                                  attn_metadata, intermediate_tensors)
        return model_output

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def make_empty_intermediate_tensors(
            self, batch_size: int, dtype: torch.dtype,
            device: torch.device) -> IntermediateTensors:
        return IntermediateTensors({
            "hidden_states":
            torch.zeros((batch_size, self.config.hidden_size),
                        dtype=dtype,
                        device=device),
            "residual":
            torch.zeros((batch_size, self.config.hidden_size),
                        dtype=dtype,
                        device=device),
        })

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            name, loaded_weight = self.maybe_remap_mistral(name, loaded_weight)

            if "rotary_emb.inv_freq" in name:
                continue
            if ("rotary_emb.cos_cached" in name
                    or "rotary_emb.sin_cached" in name):
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
            # With tie_word_embeddings, we can skip lm_head.weight
            # The weight might appear unnecessarily in the files if the model is
            # processed with quantization, LoRA, fine-tuning, etc.
            if self.config.tie_word_embeddings and "lm_head.weight" in name:
                continue
            if scale_name := get_compressed_tensors_cache_scale(name):
                # Loading kv cache scales for compressed-tensors quantization
                param = params_dict[scale_name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                loaded_weight = loaded_weight[0]
                weight_loader(param, loaded_weight)
                continue
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if is_pp_missing_parameter(name, self):
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)

                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Remapping the name of FP8 kv-scale.
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue

                if is_pp_missing_parameter(name, self):
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)

    # If this function is called, it should always initialize KV cache scale
    # factors (or else raise an exception). Thus, handled exceptions should
    # make sure to leave KV cache scale factors in a known good (dummy) state
    def load_kv_cache_scales(self, quantization_param_path: str) -> None:
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        for layer_idx, scaling_factor in kv_cache_scales_loader(
                quantization_param_path, tp_rank, tp_size,
                self.config.num_hidden_layers,
                self.config.__class__.model_type):
            if not isinstance(self.model.layers[layer_idx], nn.Identity):
                layer_self_attn = self.model.layers[layer_idx].self_attn

            if is_hip():
                # The scaling factor convention we are assuming is
                # quantized_value * scaling_factor ~= true_value
                # which is consistent with the practice of setting
                # scaling_factor = tensor_amax / FPtype_max
                scaling_factor *= 2
            if hasattr(layer_self_attn, "kv_scale"):
                layer_self_attn.attn._kv_scale = scaling_factor
            else:
                raise RuntimeError("Self attention has no KV cache scaling "
                                   "factor attribute!")

    # This function is used to remap the mistral format as
    # used by Mistral and Llama <=2
    def maybe_remap_mistral(
            self, name: str,
            loaded_weight: torch.Tensor) -> Tuple[str, torch.Tensor]:

        def permute(w, n_heads):
            attn_in = self.config.head_dim * n_heads
            attn_out = self.config.hidden_size

            return w.view(n_heads, attn_in // n_heads // 2, 2,
                          attn_out).transpose(1, 2).reshape(attn_in, attn_out)

        mapping = self.mistral_mapping
        modules = name.split(".")

        # rotary embeds should be sliced
        if "wk" in modules:
            loaded_weight = permute(loaded_weight,
                                    self.config.num_key_value_heads)
        elif "wq" in modules:
            loaded_weight = permute(loaded_weight,
                                    self.config.num_attention_heads)

        for item in modules:
            if item in mapping and mapping[item] not in name:
                name = name.replace(item, mapping[item])

        return name, loaded_weight
