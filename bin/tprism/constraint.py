import torch
import math
import sys
import re

from importlib import import_module

try:
    geotorch = import_module("geotorch")
    print('[geotorch] enabled')
except ModuleNotFoundError:
    print('[geotorch] disabled (Installation: pip install git+https://github.com/Lezcano/geotorch/)')

class BasicTensor(torch.nn.Module):
    def __init__(self, shape,  device=None, dtype=None):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(shape, device=device, dtype=dtype))
        self.reset_parameters()
    def reset_parameters(self) -> None:
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    def forward(self):
        return self.weight

def get_constraint_tensor(shape, tensor_type, device=None, dtype=None):
    if tensor_type=="symmetric":
        return SymmetricTensor(shape)
    elif tensor_type=="skew":
        return SkewTensor(shape)
    elif tensor_type=="sphere":
        return SphereTensor(shape)
    elif tensor_type=="orthogonal":
        return OrthogonalTensor(shape)
    elif tensor_type=="almost_orthogonal":
        return AlmostOrthogonalTensor(shape)
    elif tensor_type=="grassmannian":
        return GrassmannianTensor(shape)
    elif tensor_type[:len("low_rank")]=="low_rank":
        rank=1
        m=re.match(r"^\(([0-9\.]*)\)$", tensor_type[len("low_rank"):])
        if m:
            rank = int(m.group(1))
        return LowRankTensor(shape,rank)

    elif tensor_type[:len("fixed_low_rank")]=="fixed_low_rank":
        rank=1
        m=re.match(r"^\(([0-9\.]*)\)$", tensor_type[len("fixed_low_rank"):])
        if m:
            rank = int(m.group(1))
        return FixedLowRankTensor(shape,rank)
    elif tensor_type=="invertible":
        return InvertibleTensor(shape)
    elif tensor_type=="sln":
        return SLnTensor(shape)
    elif tensor_type=="positive_definite":
        return PositiveDefinite(shape)
    elif tensor_type=="positive_semidefinite":
        return PositiveSemidefinite(shape)
    elif tensor_type[:len("positive_semidefinite_low_rank")]=="positive_semidefinite_low_rank":
        rank=1
        m=re.match(r"^\(([0-9\.]*)\)$", tensor_type[len("positive_semidefinite_low_rank"):])
        if m:
            rank = int(m.group(1))
        return PositiveSemidefiniteLowRank(shape,rank)
    elif tensor_type=="positive_semidefinite_fixed_low_rank":
        rank=1
        m=re.match(r"^\(([0-9\.]*)\)$", tensor_type[len("positive_semidefinite_fixed_low_rank"):])
        if m:
            rank = int(m.group(1))
        return PositiveSemidefiniteFixedLowRank(shape,rank)

    return None

class SymmetricTensor(torch.nn.Module):
    def __init__(self, shape,  device=None, dtype=None):
        super().__init__()
        self.p = BasicTensor(shape,device,dtype)
        geotorch.symmetric(self.p, "weight")
    def forward(self):
        return self.p.weight

class SkewTensor(torch.nn.Module):
    def __init__(self, shape,  device=None, dtype=None):
        super().__init__()
        self.p = BasicTensor(shape,device,dtype)
        geotorch.skew(self.p, "weight")
    def forward(self):
        return self.p.weight

class SphereTensor(torch.nn.Module):
    def __init__(self, shape, radius=1.0, device=None, dtype=None):
        super().__init__()
        self.p = BasicTensor(shape,device,dtype)
        geotorch.sphere(self.p, "weight")
    def forward(self):
        return self.p.weight

class OrthogonalTensor(torch.nn.Module):
    def __init__(self, shape,  device=None, dtype=None):
        super().__init__()
        self.p = BasicTensor(shape,device,dtype)
        geotorch.orthogonal(self.p, "weight")
    def forward(self):
        return self.p.weight

class AlmostOrthogonalTensor(torch.nn.Module):
    def __init__(self, shape,  device=None, dtype=None):
        super().__init__()
        self.p = BasicTensor(shape,device,dtype)
        geotorch.almost_orthogonal(self.p, "weight")
    def forward(self):
        return self.p.weight

class GrassmannianTensor(torch.nn.Module):
    def __init__(self, shape,  device=None, dtype=None):
        super().__init__()
        self.p = BasicTensor(shape,device,dtype)
        geotorch.grassmannian(self.p, "weight")
    def forward(self):
        return self.p.weight


class LowRankTensor(torch.nn.Module):
    def __init__(self, shape, rank, device=None, dtype=None):
        super().__init__()
        self.p = BasicTensor(shape,device,dtype)
        geotorch.low_rank(self.p, "weight", rank)
    def forward(self):
        return self.p.weight


class FixedLowRankTensor(torch.nn.Module):
    def __init__(self, shape, rank, device=None, dtype=None):
        super().__init__()
        self.p = BasicTensor(shape,device,dtype)
        geotorch.fixed_rank(self.p, "weight", rank)
    def forward(self):
        return self.p.weight

class InvertibleTensor(torch.nn.Module):
    def __init__(self, shape, rank, device=None, dtype=None):
        super().__init__()
        self.p = BasicTensor(shape,device,dtype)
        geotorch.invertible(self.p, "weight", rank)
    def forward(self):
        return self.p.weight

#special linear group SL(n, F) 
class SLnTensor(torch.nn.Module):
    def __init__(self, shape, device=None, dtype=None):
        super().__init__()
        self.p = BasicTensor(shape,device,dtype)
        geotorch.sln(self.p, "weight")
    def forward(self):
        return self.p.weight


class PositiveDefinite(torch.nn.Module):
    def __init__(self, shape, device=None, dtype=None):
        super().__init__()
        self.p = BasicTensor(shape,device,dtype)
        geotorch.positive_definite(self.p, "weight")
    def forward(self):
        return self.p.weight

class PositiveSemidefinite(torch.nn.Module):
    def __init__(self, shape, device=None, dtype=None):
        super().__init__()
        self.p = BasicTensor(shape,device,dtype)
        geotorch.positive_semidefinite(self.p, "weight")
    def forward(self):
        return self.p.weight


class PositiveSemidefiniteLowRank(torch.nn.Module):
    def __init__(self, shape, rank, device=None, dtype=None):
        super().__init__()
        self.p = BasicTensor(shape,device,dtype)
        geotorch.positive_semidefinite_low_rank(self.p, "weight", rank)
    def forward(self):
        return self.p.weight

class PositiveSemidefiniteFixedLowRank(torch.nn.Module):
    def __init__(self, shape, rank, device=None, dtype=None):
        super().__init__()
        self.p = BasicTensor(shape,device,dtype)
        geotorch.positive_semidefinite_fixed_rank(self.p, "weight", rank)
    def forward(self):
        return self.p.weight

