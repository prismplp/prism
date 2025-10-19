    

import torch
import torch.nn.functional as F
import json
import re
import numpy as np
from google.protobuf import json_format

from itertools import chain
import collections

import inspect
import importlib
import glob
import os
import re
import pickle
import h5py
import math

import tprism.expl_pb2 as expl_pb2
import tprism.op.base
import tprism.loss.base
import tprism.constraint

from tprism.expl_graph import ComputationalExplGraph
from tprism.expl_tensor import PlaceholderGraph, VocabSet, SwitchTensorProvider
from tprism.loader import OperatorLoader
from tprism.placeholder import PlaceholderData
from numpy import int64
from torch import dtype
from typing import Any, Dict, List, Tuple, Union



 
class TorchTensorBase:
    def __init__(self):
        pass


class TorchTensorOnehot(TorchTensorBase):
    def __init__(self, provider, shape, value):
        self.shape = shape
        self.value = value

    def __call__(self):
        v = torch.eye(self.shape)[self.value]
        return v


class TorchTensor(TorchTensorBase):
    def __init__(self, provider: 'TorchSwitchTensorProvider', name: str, shape: List[Union[int64, int]], dtype: dtype=torch.float32, tensor_type: str="") -> None:
        self.shape = shape
        self.dtype = dtype
        if name is None:
            self.name = "tensor%04d" % (np.random.randint(0, 10000),)
        else:
            self.name = name
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
    def __init__(self, provider: 'TorchSwitchTensorProvider', var: TorchTensor, idx: PlaceholderData) -> None:
        self.var = var
        self.idx = idx
        self.provider = provider

    def __call__(self):
        if isinstance(self.idx, PlaceholderData):
            idx = self.provider.get_embedding(self.idx)
        else:
            idx = self.idx
        if isinstance(self.var, TorchTensor):
            temp = self.var()
            v = torch.index_select(temp, 0, idx)
        elif isinstance(self.var, TorchTensorBase):
            v = torch.index_select(self.var(), 0, idx)
        elif isinstance(self.var, PlaceholderData):
            v = self.provider.get_embedding(self.var)
            v = v[idx]
        else:
            v = torch.index_select(self.var, 0, idx)
        return v


class TorchSwitchTensorProvider(SwitchTensorProvider):
    def __init__(self) -> None:
        super().__init__()
        self.tensor_onehot_class = TorchTensorOnehot
        self.tensor_class = TorchTensor
        self.tensor_gather_class = TorchGather
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
    def get_embedding(self, name: Union[str,PlaceholderData], verbose:bool=False):
        if verbose:
            print("[INFO] get embedding:", name)
        out = None
        ## TODO:
        if self.input_feed_dict is None:
            if verbose:
                print("[INFO] from tensor_embedding", name)
            obj = self.tensor_embedding[name]
            if isinstance(obj, TorchTensorBase):
                out = obj()
            else:
                raise Exception("Unknoen embedding type", name, type(obj))
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
            print("sum:", out.sum())
        return out

