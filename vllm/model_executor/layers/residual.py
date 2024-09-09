from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from vllm.distributed import (get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size,
                              split_tensor_along_first_dim,
                              tensor_model_parallel_all_gather)

class ResidualBase(nn.Module):
    def __init__(
        self, 
        prefix: str = None,
        hidden_size: int = 0,
    ):
        super(ResidualBase, self).__init__()

        self.prefix = prefix

        # passed as a hint to the model parallelism
        self.hidden_size = hidden_size

    def forward(self, feedforward, residual):
        return feedforward + residual

class ReplicatedResidual(ResidualBase):
    def forward(self, feedforward, residual):
        return super(ReplicatedResidual, self).forward(feedforward, residual)

class DataParallelResidual(ResidualBase):
    def __init__(
        self,
        input_is_parallel: bool = True,
        residual_is_parallel: bool = True,
        gather_output: bool = False,
        prefix: str = None,
        hidden_size: int = 0,
    ):
        super(DataParallelResidual, self).__init__(
            prefix,
            hidden_size,
        )

        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.world_size = get_tensor_model_parallel_world_size()

        self.input_is_parallel = input_is_parallel
        self.residual_is_parallel = residual_is_parallel
        self.gather_output = gather_output

    def forward(
        self,
        feedforward,
        residual,
        num_tokens: Optional[int] = None,
    ):
        # Parallelize feedforward input if necessary
        if self.input_is_parallel:
            feedforward_parallel = feedforward
        elif not self.input_is_parallel and feedforward.size(0) < self.world_size:
            feedforward_parallel = feedforward
        else:
            # Pad with zeros if necessary
            in_size = feedforward.size(0)
            diff = in_size % self.world_size
            if diff != 0:
                pad_size = self.world_size - diff
                feedforward = torch.cat([feedforward, torch.zeros(pad_size, *feedforward.shape[1:], device=feedforward.device)], dim=0)
            
            splitted = split_tensor_along_first_dim(
                feedforward, num_partitions=self.tp_size)
            feedforward_parallel = splitted[self.tp_rank].contiguous()

        # Parallelize residual if necessary
        if self.residual_is_parallel:
            residual_parallel = residual
        elif not self.residual_is_parallel and residual.size(0) < self.world_size:
            residual_parallel = residual
        else:
            # Pad with zeros if necessary
            in_size = residual.size(0)
            diff = in_size % self.world_size
            if diff != 0:
                pad_size = self.world_size - diff
                residual = torch.cat([residual, torch.zeros(pad_size, *residual.shape[1:], device=residual.device)], dim=0)

            splitted = split_tensor_along_first_dim(
                residual, num_partitions=self.tp_size)
            residual_parallel = splitted[self.tp_rank].contiguous()

        out = feedforward_parallel + residual_parallel

        if self.gather_output:
            out = tensor_model_parallel_all_gather(out, dim=0)

            # Truncate output when this has been padded in another layer
            if num_tokens is not None:
                out = out[:num_tokens]

        return out