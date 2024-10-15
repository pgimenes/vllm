# coding=utf-8
# Copyright 2024 The vLLM team.
# Copyright 2024 Google Inc. HuggingFace Inc. team. All rights reserved.
#
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
from typing import Iterable, List, Optional, Set, Tuple, Union

import torch
from torch import nn
from transformers import Gemma2Config

from vllm.attention import Attention, AttentionMetadata
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, LoRAConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import GeluAndMul
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.linear import (MergedColumnParallelLinear,
                                               MergedDataParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import Sampler, SamplerOutput
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors

from .interfaces import SupportsLoRA
from .utils import _linear_cls_from_config, _qkv_linear_cls_from_config, _gemma_rms_norm_cls_from_config

logger = init_logger(__name__)

from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

class Gemma2MLP(nn.Module):

    def __init__(
        self,
        config: Gemma2Config,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        hidden_activation: str,
        layer_idx: int,
        quant_config: Optional[QuantizationConfig] = None,
        sharding_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        
        prefix = f"model.layers.{layer_idx}.mlp"
        
        # layer info
        layer_num = "".join(filter(str.isdigit, prefix))
        self.is_first_layer = layer_num == "0"
        self.is_last_layer = layer_num == str(config.num_hidden_layers - 1)
        next_layer_num = str(int(layer_num) + 1)

        # get sharding configs
        pre_feedforward_layernorm_type = sharding_config.get(
            f"{prefix.replace('.mlp', '')}.pre_feedforward_layernorm", "replicated")

        self.gate_up_proj_type = sharding_config.get(f"{prefix}.gate_up_proj", "column")
        self.down_proj_type = sharding_config.get(f"{prefix}.down_proj", "row")

        post_feedforward_layernorm_type = sharding_config.get(
            f"{prefix.replace('.mlp', '')}.post_feedforward_layernorm", "replicated")

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
            "bias": False,
            "quant_config": quant_config,
            "prefix": f"{prefix}.gate_up_proj",
        }

        if self.gate_up_proj_type == "data":
            if pre_feedforward_layernorm_type != "data":
                gate_up_proj_kwargs["input_is_parallel"] = False
            if self.down_proj_type != "data":
                gate_up_proj_kwargs["gather_output"] = True

        self.gate_up_proj = gate_up_proj_cls(**gate_up_proj_kwargs)
        
        # down_proj
        down_proj_cls = _linear_cls_from_config(self.down_proj_type)
        down_proj_kwargs = {
            "input_size": intermediate_size,
            "output_size": hidden_size,
            "bias": False,
            "quant_config": quant_config,
            "prefix": f"{prefix}.down_proj",
        }
        if self.down_proj_type == "data":
            if self.gate_up_proj_type != "data":
                down_proj_kwargs["input_is_parallel"] = False
            if self.is_last_layer or post_feedforward_layernorm_type != "data":
                down_proj_kwargs["gather_output"] = True
            
        self.down_proj = down_proj_cls(**down_proj_kwargs)
        if not (hidden_act == hidden_activation == "gelu_pytorch_tanh"):
            raise ValueError(
                "Gemma2 uses `gelu_pytorch_tanh` as the hidden activation "
                "function. Please set `hidden_act` and `hidden_activation` to "
                "`gelu_pytorch_tanh`.")
        self.act_fn = GeluAndMul(approximate="tanh")

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


class Gemma2Attention(nn.Module):

    def __init__(self,
                 layer_idx: int,
                 config: Gemma2Config,
                 hidden_size: int,
                 num_heads: int,
                 num_kv_heads: int,
                 head_dim: int,
                 max_position_embeddings: int,
                 rope_theta: float,
                 cache_config: Optional[CacheConfig] = None,
                 quant_config: Optional[QuantizationConfig] = None,
                 sharding_config: Optional[Dict[str, Any]] = None,
                 attn_logits_soft_cap: Optional[float] = None) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
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
        self.head_dim = head_dim

        # get sharding config for each submodule
        prefix = f"model.layers.{layer_idx}.self_attn"
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

        self.scaling = config.query_pre_attn_scalar**-0.5
        self.rope_theta = rope_theta

        # qkv_proj
        qkv_proj_cls = _qkv_linear_cls_from_config(self.qkv_proj_type)
        qkv_proj_kwargs = {
            "hidden_size": hidden_size,
            "head_size": self.head_dim,
            "total_num_heads": self.total_num_heads,
            "total_num_kv_heads": self.total_num_kv_heads,
            "bias": config.attention_bias,
            "quant_config": quant_config,
            "prefix": f"{prefix}.qkv_proj",
        }
        if self.qkv_proj_type == "data":
            if input_layernorm_type != "data":
                qkv_proj_kwargs["input_is_parallel"] = False

            qkv_proj_kwargs["gather_output"] = True

        self.qkv_proj = qkv_proj_cls(**qkv_proj_kwargs)

        # rotary_emb
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=self.rope_theta,
            is_neox_style=True,
        )

        # attn
        attn_kwargs = {
            "num_heads": self.num_heads,
            "head_size": self.head_dim,
            "scale": self.scaling,
            "num_kv_heads": self.num_kv_heads,
            "cache_config": cache_config,
            "quant_config": quant_config,
            "logits_soft_cap": attn_logits_soft_cap,
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
            "bias": config.attention_bias,
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


class Gemma2DecoderLayer(nn.Module):

    def __init__(
        self,
        layer_idx: int,
        config: Gemma2Config,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        sharding_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size

        # layer info
        prefix = f"model.layers.{layer_idx}"
        layer_num = "".join(filter(str.isdigit, prefix))
        self.is_first_layer = layer_num == "0"
        self.is_last_layer = layer_num == str(config.num_hidden_layers - 1)
        last_layer_num = str(int(layer_num) - 1)
        next_layer_num = str(int(layer_num) + 1)

        # get sharding config for each submodule
        last_layer_pre_feedforward_layernorm_type = sharding_config.get(
            f"{prefix.replace(layer_num, last_layer_num)}.pre_feedforward_layernorm", "replicated")
        last_layer_post_feedforward_layernorm_type = sharding_config.get(
            f"{prefix.replace(layer_num, last_layer_num)}.post_feedforward_layernorm", "replicated")

        self.last_layer_pre_feedforward_layernorm_type = last_layer_pre_feedforward_layernorm_type
        self.last_layer_post_feedforward_layernorm_type = last_layer_post_feedforward_layernorm_type
             
        self.input_layernorm_type = sharding_config.get(f"{prefix}.input_layernorm", "replicated")
        self.attn_qkv_proj_type = sharding_config.get(f"{prefix}.self_attn.qkv_proj", "column")
        self.attn_o_proj_type = sharding_config.get(f"{prefix}.self_attn.o_proj", "row")
        self.mlp_gate_up_proj_type = sharding_config.get(f"{prefix}.mlp.gate_up_proj", "column")
        self.mlp_down_proj_type = sharding_config.get(f"{prefix}.mlp.down_proj", "row")
        self.post_attention_layernorm_type = sharding_config.get(f"{prefix}.post_attention_layernorm", "replicated")
        self.pre_feedforward_layernorm_type = sharding_config.get(f"{prefix}.pre_feedforward_layernorm", "replicated")
        self.post_feedforward_layernorm_type = sharding_config.get(f"{prefix}.post_feedforward_layernorm", "replicated")

        next_layer_input_layernorm_type = sharding_config.get(
            f"{prefix.replace(layer_num, next_layer_num)}.input_layernorm", "replicated")

        # input_layernorm
        input_layernorm_cls = _gemma_rms_norm_cls_from_config(self.input_layernorm_type)
        input_layernorm_kwargs = {
            "hidden_size": config.hidden_size,
            "eps": config.rms_norm_eps,
        }
        if self.input_layernorm_type == "data":
            if self.is_first_layer or last_layer_post_feedforward_layernorm_type != "data":
                input_layernorm_kwargs["input_is_parallel"] = False
            if self.is_first_layer or last_layer_pre_feedforward_layernorm_type != "data":
                input_layernorm_kwargs["residual_is_parallel"] = False
            if self.attn_qkv_proj_type != "data":
                input_layernorm_kwargs["gather_output"] = True
            if self.pre_feedforward_layernorm_type != "data":
                input_layernorm_kwargs["gather_output_residual"] = True
        self.input_layernorm = input_layernorm_cls(**input_layernorm_kwargs)

        # attention
        self.self_attn = Gemma2Attention(
            layer_idx=layer_idx,
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=config.rope_theta,
            cache_config=cache_config,
            quant_config=quant_config,
            attn_logits_soft_cap=config.attn_logit_softcapping,
            sharding_config=sharding_config,
        )

        # post_attention_layernorm
        post_attention_layernorm_cls = _gemma_rms_norm_cls_from_config(self.post_attention_layernorm_type)
        post_attention_layernorm_kwargs = {
            "hidden_size": config.hidden_size,
            "eps": config.rms_norm_eps,
        }
        if self.post_attention_layernorm_type == "data":
            if self.attn_o_proj_type != "data":
                post_attention_layernorm_kwargs["input_is_parallel"] = False
            
            if self.pre_feedforward_layernorm_type != "data":
                post_attention_layernorm_kwargs["gather_output"] = True

        self.post_attention_layernorm = post_attention_layernorm_cls(**post_attention_layernorm_kwargs)

        # pre_feedforward_layernorm
        pre_feedforward_layernorm_cls = _gemma_rms_norm_cls_from_config(self.pre_feedforward_layernorm_type)
        pre_feedforward_layernorm_kwargs = {
            "hidden_size": config.hidden_size,
            "eps": config.rms_norm_eps,
        }
        if self.pre_feedforward_layernorm_type == "data":
            if self.post_attention_layernorm_type != "data":
                pre_feedforward_layernorm_kwargs["input_is_parallel"] = False
            if self.input_layernorm_type != "data":
                pre_feedforward_layernorm_kwargs["residual_is_parallel"] = False
            if self.mlp_gate_up_proj_type != "data":
                pre_feedforward_layernorm_kwargs["gather_output"] = True
            if next_layer_input_layernorm_type != "data":
                pre_feedforward_layernorm_kwargs["gather_output_residual"] = True
            
        self.pre_feedforward_layernorm = pre_feedforward_layernorm_cls(**pre_feedforward_layernorm_kwargs)
        
        # mlp
        self.mlp = Gemma2MLP(
            config=config,
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            hidden_activation=config.hidden_activation,
            quant_config=quant_config,
            layer_idx=layer_idx,
            sharding_config=sharding_config,
        )
        
        # post_feedforward_layernorm
        post_feedforward_layernorm_cls = _gemma_rms_norm_cls_from_config(self.post_feedforward_layernorm_type)
        post_feedforward_layernorm_kwargs = {
            "hidden_size": config.hidden_size,
            "eps": config.rms_norm_eps,
        }
        if self.post_feedforward_layernorm_type == "data":
            if self.mlp_down_proj_type != "data":
                post_feedforward_layernorm_kwargs["input_is_parallel"] = False
            
            if self.is_last_layer or next_layer_input_layernorm_type != "data":
                post_feedforward_layernorm_kwargs["gather_output"] = True

        self.post_feedforward_layernorm = post_feedforward_layernorm_cls(**post_feedforward_layernorm_kwargs)

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
        hidden_states = self.post_attention_layernorm(hidden_states, **kwargs)

        kwargs = {"num_tokens": num_tokens} if self.pre_feedforward_layernorm_type == "data" else {}
        hidden_states, residual = self.pre_feedforward_layernorm(
            hidden_states, residual, **kwargs)
        
        hidden_states = self.mlp(hidden_states)
        
        kwargs = {"num_tokens": num_tokens} if self.post_feedforward_layernorm_type == "data" else {}
        hidden_states = self.post_feedforward_layernorm(hidden_states, **kwargs)

        return hidden_states, residual


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": 0,
        "inputs_embeds": 0,
        "intermediate_tensors": 0,
    })
class Gemma2Model(nn.Module):

    def __init__(
        self,
        config: Gemma2Config,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        sharding_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.config = config

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )
        self.layers = nn.ModuleList([
            Gemma2DecoderLayer(layer_idx, config, cache_config, quant_config, sharding_config)
            for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Normalize the embedding by sqrt(hidden_size)
        # The normalizer's data type should be downcasted to the model's
        # data type such as bfloat16, not float32.
        # See https://github.com/huggingface/transformers/pull/29402
        normalizer = self.config.hidden_size**0.5
        self.register_buffer("normalizer", torch.tensor(normalizer))
        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size))

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
                hidden_states = self.embed_tokens(input_ids)
            hidden_states *= self.normalizer
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

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: Set[str] = set()
        for name, loaded_weight in weights:
            for (param_name, shard_name, shard_id) in stacked_params_mapping:
                if shard_name not in name:
                    continue
                name = name.replace(shard_name, param_name)
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
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)

        unloaded_params = params_dict.keys() - loaded_params
        if unloaded_params:
            logger.warning(
                "Some weights are not initialized from checkpoints: %s",
                unloaded_params)


class Gemma2ForCausalLM(nn.Module, SupportsLoRA, SupportsPP):
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
        "qkv_proj",
        "o_proj",
        "gate_up_proj",
        "down_proj",
    ]
    # Gemma does not apply LoRA to the embedding layer.
    embedding_modules = {}
    embedding_padding_modules = []

    # BitandBytes specific attributes
    default_bitsandbytes_target_modules = [
        ".gate_proj.",
        ".down_proj.",
        ".up_proj.",
        ".q_proj.",
        ".k_proj.",
        ".v_proj.",
        ".o_proj.",
    ]
    # in TP, these weights are partitioned along the column dimension (dim=-1)
    column_parallel_weights_modules = [".down_proj.", ".o_proj."]
    bitsandbytes_stacked_params_mapping = {
        # shard_name, weight_name, index
        "q_proj": ("qkv_proj", 0),
        "k_proj": ("qkv_proj", 1),
        "v_proj": ("qkv_proj", 2),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: Gemma2Config,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        lora_config: Optional[LoRAConfig] = None,
        sharding_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        del lora_config  # Unused.
        super().__init__()
        self.config = config
        # currently all existing Gemma models have `tie_word_embeddings` enabled
        assert config.tie_word_embeddings
        self.quant_config = quant_config
        self.model = Gemma2Model(config, cache_config, quant_config, sharding_config)
        self.logits_processor = LogitsProcessor(
            config.vocab_size, soft_cap=config.final_logit_softcapping)
        self.sampler = Sampler()
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        hidden_states = self.model(input_ids, positions, kv_caches,
                                   attn_metadata, intermediate_tensors)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.model.embed_tokens, hidden_states,
                                       sampling_metadata)
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."]
                           if self.config.tie_word_embeddings else None),
        )
        loader.load_weights(weights)
