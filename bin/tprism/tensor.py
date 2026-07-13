"""Tensor classes provided to the computational explanation graph.

These classes wrap the concrete torch tensors backing each switch:
trainable parameters (`TorchTensor`), constant one-hot vectors
(`TorchTensorOnehot`), and index-selection views (`TorchGather`).
Calling an instance returns the current tensor value.
"""

import math

import numpy as np
import torch

import tprism.constraint
from tprism.placeholder import PlaceholderData

from torch import Tensor, dtype
from typing import Any, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from tprism.expl_tensor import SwitchTensorProvider


class TorchTensorBase:
    def __init__(self, shape: Optional[Tuple[int, ...]] = None) -> None:
        # Provide a common shape attribute so callers can access .shape safely.
        self.shape: Optional[Tuple[int, ...]] = shape

    def __call__(self) -> Optional[Tensor]:
        return None


class TorchTensorOnehot(TorchTensorBase):
    def __init__(self, provider: 'SwitchTensorProvider',
                  shape: List[int], value:Any):
        self.shape:Tuple[int, ...] = tuple(shape)
        self.value = value

    def __call__(self):
        v = torch.eye(self.shape[0])[self.value]
        return v


class TorchTensor(TorchTensorBase):
    def __init__(self, provider: 'SwitchTensorProvider', name: str, shape: List[int], dtype: Optional[dtype]=torch.float32, tensor_type: Optional[str]="") -> None:
        if dtype is None:
            dtype = torch.float32
        if tensor_type is None:
            tensor_type = ""
        self.shape:Tuple[int, ...] = tuple(shape)
        self.dtype = dtype
        self.name: str = ""
        if name is None:
            self.name= "tensor%04d" % (np.random.randint(0, 10000),)
        else:
            self.name= name
        self.provider = provider
        self.tensor_type = tensor_type
        ###
        self.constraint_tensor=tprism.constraint.get_constraint_tensor(shape, tensor_type, device=None, dtype=None)
        self.param = None
        if self.constraint_tensor is None:
            param = torch.nn.Parameter(torch.Tensor(*shape), requires_grad=True)
            self.param = param
            provider.add_param(self.name, param, tensor_type)
            self.reset_parameters()
        else:
            param=list(self.constraint_tensor.parameters())[0] #TODO
            provider.add_param(self.name, param, tensor_type)

        ###

    def reset_parameters(self) -> None:
        if self.param is not None:
            if len(self.param.shape) == 2:
                torch.nn.init.kaiming_uniform_(self.param, a=math.sqrt(5))
            else:
                self.param.data.uniform_(-0.1, 0.1)

    def __call__(self):
        if self.constraint_tensor is None:
            return self.param
        else:
            return self.constraint_tensor()


class TorchGather(TorchTensorBase):
    def __init__(self, provider: 'SwitchTensorProvider', var: PlaceholderData|TorchTensorBase, idx: PlaceholderData|TorchTensorBase|int) -> None:
        self.var:PlaceholderData|TorchTensorBase = var
        self.idx:PlaceholderData|TorchTensorBase|int = idx
        self.provider = provider

    def __call__(self):
        if isinstance(self.idx, PlaceholderData):
            idx_embed = self.provider.get_embedding(self.idx)
        else:
            idx_embed = self.idx
        if isinstance(self.var, TorchTensor):
            temp = self.var()
            v = torch.index_select(temp, 0, idx_embed)
        elif isinstance(self.var, TorchTensorBase):
            v = torch.index_select(self.var(), 0, idx_embed)
        elif isinstance(self.var, PlaceholderData):
            v = self.provider.get_embedding(self.var)
            v = v[idx_embed]
        else:
            v = torch.index_select(self.var, 0, idx_embed)
        return v
