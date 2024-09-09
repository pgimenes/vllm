import torch
import torch.nn as nn
from torch import Size

from vllm.distributed import (get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size,
                              split_tensor_along_first_dim,
                              tensor_model_parallel_all_gather)

from typing import List, Optional, Tuple, Union

_shape_t = Union[int, List[int], Size]

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