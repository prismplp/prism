    

import torch
import torch.nn.functional as F
import re
import numpy as np

import re
import math

import tprism.constraint

from tprism.expl_tensor import PlaceholderGraph, VocabSet, SwitchTensorProvider, TorchTensorBase
from tprism.placeholder import PlaceholderData
from torch import dtype, Tensor
from typing import Any, Dict, List, Tuple, Union, Optional, Type




class TorchTensorOnehot(TorchTensorBase):
    def __init__(self, provider: SwitchTensorProvider,
                  shape: List[int], value:Any):
        self.shape:Tuple[int, ...] = tuple(shape)
        self.value = value

    @staticmethod
    def builder_func(provider: SwitchTensorProvider,
                  shape: List[int], value:Any) -> TorchTensorBase:
        return TorchTensorOnehot(provider, shape, value)

    def __call__(self):
        v = torch.eye(self.shape[0])[self.value]
        return v


class TorchTensor(TorchTensorBase):
    def __init__(self, provider: SwitchTensorProvider, name: str, shape: List[int], dtype: dtype=torch.float32, tensor_type: str="") -> None:
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

    @staticmethod
    def builder_func(provider: SwitchTensorProvider,
                  name: str, shape: List[int], dtype: Optional[dtype]=torch.float32, tensor_type: Optional[str]="") -> TorchTensorBase:
        if tensor_type is None:
            tensor_type = ""
        if dtype is None:
            dtype=torch.float32
        return TorchTensor(provider, name, shape, dtype, tensor_type)

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
    def __init__(self, provider: SwitchTensorProvider, var: PlaceholderData|TorchTensorBase, idx: PlaceholderData|TorchTensorBase|int) -> None:
        self.var:PlaceholderData|TorchTensorBase = var
        self.idx:PlaceholderData|TorchTensorBase|int = idx
        self.provider = provider

    @staticmethod
    def builder_func(provider: SwitchTensorProvider, var: PlaceholderData|TorchTensorBase, idx: PlaceholderData|TorchTensorBase|int) -> TorchTensorBase:
        return TorchGather(provider, var, idx)
    
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


class TorchSwitchTensorProvider(SwitchTensorProvider):
    integer_dtype: Optional[torch.dtype] = None

    def __init__(self) -> None:
        super().__init__()
        self.tensor_onehot_class = TorchTensorOnehot.builder_func
        self.tensor_class = TorchTensor.builder_func
        self.tensor_gather_class = TorchGather.builder_func
        self.integer_dtype = torch.int32
        

    # forward
    def get_loss(self, verbose:bool=False):
        loss={}
        for name, (param,tensor_type) in self.params.items():
            m=re.match(r"^sparse\(([0-9\.]*)\)$", tensor_type)
            if m:
                coeff = float(m.group(1))
                l=coeff*torch.norm(param,1)
                loss["sparse_"+name]=l
            elif tensor_type=="sparse":
                l=torch.norm(param,1)
                loss["sparse_"+name]=l
        return loss
    # forward
    def get_embedding(self, name: Union[str,PlaceholderData], verbose:bool=False)->Optional[TorchTensorBase|Tensor]:
        if verbose:
            print("[INFO] get embedding:", name)
        out = None
        ## TODO:
        if self.input_feed_dict is None:
            if verbose:
                print("[INFO] from tensor_embedding", name)
            if  type(name) is str:
                obj = self.tensor_embedding[name]
                if isinstance(obj, TorchTensorBase):
                    out = obj()
                else:
                    raise Exception("Unknown embedding key", name, type(obj))
            else:
                raise Exception("Unknown embedding key", name)
        elif type(name) is str:
            key = self.tensor_embedding[name]
            if type(key) is PlaceholderData:
                if verbose:
                    print("[INFO] from PlaceholderData", name, "==>", key.name)
                out = self.input_feed_dict[key]
            elif isinstance(key, TorchTensorBase):
                if verbose:
                    print("[INFO] from Tensor", name, "==>")
                out = key()
            else:
                raise Exception("Unknoen embedding type", name, key)
        elif type(name) is PlaceholderData:
            if verbose:
                print("[INFO] from PlaceholderData", name)
            out = self.input_feed_dict[name]
        else:
            raise Exception("Unknoen embedding", name)
        if verbose:
            print(out)
            print(type(out))
            if isinstance(out, torch.Tensor):
                print("sum:", out.sum())
        return out

