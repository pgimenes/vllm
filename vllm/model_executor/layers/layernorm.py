"""Custom normalization layers."""
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Size

from vllm.model_executor.custom_op import CustomOp

from typing import List, Optional, Tuple, Union
_shape_t = Union[int, List[int], Size]

from vllm.distributed import (get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size,
                              split_tensor_along_first_dim,
                              tensor_model_parallel_all_gather)

class LayerNormBase(nn.LayerNorm):
    def __init__(
        self,
        normalized_shape: _shape_t,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        device=None,
        dtype=None,
        prefix: Optional[str] = None,
    ):
        super(LayerNormBase, self).__init__(
            normalized_shape=normalized_shape,
            eps=eps,
            elementwise_affine=elementwise_affine,
            bias=bias,
            device=device,
            dtype=dtype,
        )

        self.prefix = prefix

class ReplicatedLayerNorm(LayerNormBase):
    def forward(self, input_):
        return super(ReplicatedLayerNorm, self).forward(input_)

class DataParallelLayerNorm(LayerNormBase):
    def __init__(
        self,
        normalized_shape: _shape_t,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        device=None,
        dtype=None,
        input_is_parallel: bool = True,
        gather_output: bool = False,
        prefix: Optional[str] = None,
    ):
        super(DataParallelLayerNorm, self).__init__(
            normalized_shape,
            eps=eps,
            elementwise_affine=elementwise_affine,
            bias=bias,
            device=device,
            dtype=dtype,
            prefix=prefix,
        )

        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.world_size = get_tensor_model_parallel_world_size()

        self.input_is_parallel = input_is_parallel
        self.gather_output = gather_output

    def forward(
        self,
        input_,
        num_tokens: Optional[int] = None,
    ):
        # Parallelize feedforward input if necessary
        if self.input_is_parallel:
            input_parallel = input_
        elif not self.input_is_parallel and input_.size(0) < self.world_size:
            input_parallel = input_
        else:
            # Pad with zeros if necessary
            in_size = input_.size(0)
            diff = in_size % self.world_size
            if diff != 0:
                pad_size = self.world_size - diff
                input_ = torch.cat([input_, torch.zeros(pad_size, *input_.shape[1:], device=input_.device)], dim=0)

            splitted = split_tensor_along_first_dim(
                input_, num_partitions=self.tp_size)
            input_parallel = splitted[self.tp_rank].contiguous()

        out = super(DataParallelLayerNorm, self).forward(input_parallel)

        if self.gather_output:
            out = tensor_model_parallel_all_gather(out, dim=0)

            # Truncate output when this has been padded in another layer
            if num_tokens is not None:
                out = out[:num_tokens]

        return out

class RMSNorm(CustomOp):
    """Root mean square normalization.

    Computes x -> w * x / sqrt(E[x^2] + eps) where w is the learned weight.
    Refer to https://arxiv.org/abs/1910.07467
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward_native(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """PyTorch-native implementation equivalent to forward()."""
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        if residual is not None:
            x = x + residual.to(torch.float32)
            residual = x.to(orig_dtype)

        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        x = x.to(orig_dtype) * self.weight
        if residual is None:
            return x
        else:
            return x, residual

    def forward_cuda(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        from vllm import _custom_ops as ops

        if residual is not None:
            ops.fused_add_rms_norm(
                x,
                residual,
                self.weight.data,
                self.variance_epsilon,
            )
            return x, residual
        out = torch.empty_like(x)
        ops.rms_norm(
            out,
            x,
            self.weight.data,
            self.variance_epsilon,
        )
        return out

    def forward_xpu(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        from vllm._ipex_ops import ipex_ops as ops

        if residual is not None:
            ops.fused_add_rms_norm(
                x,
                residual,
                self.weight.data,
                self.variance_epsilon,
            )
            return x, residual
        out = torch.empty_like(x)
        ops.rms_norm(
            out,
            x,
            self.weight.data,
            self.variance_epsilon,
        )
        return out

    def extra_repr(self) -> str:
        s = f"hidden_size={self.weight.data.size(0)}"
        s += f", eps={self.variance_epsilon}"
        return s

class DataParallelRMSNorm(RMSNorm):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        input_is_parallel: bool = True,
        residual_is_parallel: bool = True,
        gather_output: bool = False,
        gather_output_residual: bool = False,
    ) -> None:
        super().__init__(hidden_size, eps)

        self.input_is_parallel = input_is_parallel
        self.residual_is_parallel = residual_is_parallel
        self.gather_output = gather_output
        self.gather_output_residual = gather_output_residual

        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()

    def forward(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        num_tokens: Optional[int] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # Check if input needs to be parallelized
        if self.input_is_parallel:
            x_parallel = x
        elif not self.input_is_parallel and x.size(0) < self.tp_size:
            x_parallel = x
        else:
            # Pad with zeros if necessary
            in_size = x.size(0)
            diff = in_size % self.tp_size
            if diff != 0:
                pad_size = self.tp_size - diff
                x = torch.cat([x, torch.zeros(pad_size, *x.shape[1:], device=x.device)], dim=0)

            splitted = split_tensor_along_first_dim(
                x,
                num_partitions=self.tp_size,
            )
            x_parallel = splitted[self.tp_rank]

        # Check if residual needs to be parallelized
        if residual is None:
            residual_parallel = None
        else:
            if self.residual_is_parallel:
                residual_parallel = residual
            elif not self.residual_is_parallel and residual.size(0) < self.tp_size:
                residual_parallel = residual
            else:
                # Pad with zeros if necessary
                in_size = residual.size(0)
                diff = in_size % self.tp_size
                if diff != 0:
                    pad_size = self.tp_size - diff
                    residual = torch.cat([residual, torch.zeros(pad_size, *residual.shape[1:], device=residual.device)], dim=0)
                    
                splitted = split_tensor_along_first_dim(
                    residual,
                    num_partitions=self.tp_size,
                )
                residual_parallel = splitted[self.tp_rank]

        out = super().forward(x_parallel, residual_parallel)

        if residual is not None:
            out, out_residual = out

        # Gather output if necessary
        if self.gather_output:
            out = tensor_model_parallel_all_gather(out, dim=0)

            # Truncate output when this has been padded in another layer
            if num_tokens is not None:
                out = out[:num_tokens]

        if residual is None:
            return out
        
        else:
            # Gather output residual if necessary
            if self.gather_output_residual:
                out_residual = tensor_model_parallel_all_gather(out_residual, dim=0)
                
                # Truncate output when this has been padded in another layer
                if num_tokens is not None:
                    out_residual = out_residual[:num_tokens]

            return out, out_residual

class GemmaRMSNorm(CustomOp):
    """RMS normalization for Gemma.

    Two differences from the above RMSNorm:
        1. x * (1 + w) instead of x * w.
        2. (x * w).to(orig_dtype) instead of x.to(orig_dtype) * w.
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    @staticmethod
    def forward_static(
        weight: torch.Tensor,
        variance_epsilon: float,
        x: torch.Tensor,
        residual: Optional[torch.Tensor],
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """PyTorch-native implementation equivalent to forward()."""
        orig_dtype = x.dtype
        if residual is not None:
            x = x + residual
            residual = x

        x = x.float()
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + variance_epsilon)
        # Llama does x.to(float16) * w whilst Gemma is (x * w).to(float16)
        # See https://github.com/huggingface/transformers/pull/29402
        x = x * (1.0 + weight.float())
        x = x.to(orig_dtype)
        return x if residual is None else (x, residual)

    def forward_native(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """PyTorch-native implementation equivalent to forward()."""
        return self.forward_static(self.weight.data, self.variance_epsilon, x,
                                   residual)

    def forward_cuda(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if torch.compiler.is_compiling():
            return self.forward_native(x, residual)

        if not getattr(self, "_is_compiled", False):
            self.forward_static = torch.compile(  # type: ignore
                self.forward_static)
            self._is_compiled = True
        return self.forward_native(x, residual)

class DataParallelGemmaRMSNorm(GemmaRMSNorm):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        input_is_parallel: bool = True,
        residual_is_parallel: bool = True,
        gather_output: bool = False,
        gather_output_residual: bool = False,
    ) -> None:
        super().__init__(hidden_size, eps)

        self.input_is_parallel = input_is_parallel
        self.residual_is_parallel = residual_is_parallel
        self.gather_output = gather_output
        self.gather_output_residual = gather_output_residual

        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()

    def forward(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        num_tokens: Optional[int] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # Check if input needs to be parallelized
        if self.input_is_parallel:
            x_parallel = x
        elif not self.input_is_parallel and x.size(0) < self.tp_size:
            x_parallel = x
        else:
            # Pad with zeros if necessary
            in_size = x.size(0)
            diff = in_size % self.tp_size
            if diff != 0:
                pad_size = self.tp_size - diff
                x = torch.cat([x, torch.zeros(pad_size, *x.shape[1:], device=x.device)], dim=0)

            splitted = split_tensor_along_first_dim(
                x,
                num_partitions=self.tp_size,
            )
            x_parallel = splitted[self.tp_rank]

        # Check if residual needs to be parallelized
        if residual is None:
            residual_parallel = None
        else:
            if self.residual_is_parallel:
                residual_parallel = residual
            elif not self.residual_is_parallel and residual.size(0) < self.tp_size:
                residual_parallel = residual
            else:
                # Pad with zeros if necessary
                in_size = residual.size(0)
                diff = in_size % self.tp_size
                if diff != 0:
                    pad_size = self.tp_size - diff
                    residual = torch.cat([residual, torch.zeros(pad_size, *residual.shape[1:], device=residual.device)], dim=0)
                    
                splitted = split_tensor_along_first_dim(
                    residual,
                    num_partitions=self.tp_size,
                )
                residual_parallel = splitted[self.tp_rank]

        out = super().forward(x_parallel, residual_parallel)

        if residual is not None:
            out, out_residual = out

        # Gather output if necessary
        if self.gather_output:
            out = tensor_model_parallel_all_gather(out, dim=0)

            # Truncate output when this has been padded in another layer
            if num_tokens is not None:
                out = out[:num_tokens]

        if residual is None:
            return out
        
        else:
            # Gather output residual if necessary
            if self.gather_output_residual:
                out_residual = tensor_model_parallel_all_gather(out_residual, dim=0)
                
                # Truncate output when this has been padded in another layer
                if num_tokens is not None:
                    out_residual = out_residual[:num_tokens]

            return out, out_residual