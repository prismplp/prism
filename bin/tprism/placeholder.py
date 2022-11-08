""" 
This module contains Placeholder
"""

from numpy import int32, int64, ndarray, str_
from torch import Tensor, dtype
from typing import Any, Optional, Tuple, Type, Union


class PlaceholderData:
    def __init__(self, name: str,
            shape: Tuple[int,...]=(),
            dtype: Optional[dtype]=None,
            ref: Optional[str]=None) -> None:
        self.name = name
        self.reference = ref
        self.shape = shape
        self.dtype = dtype


