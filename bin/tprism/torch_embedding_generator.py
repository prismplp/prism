"""
This module contains EmbeddingGenerators,
which assign a tensor to an atom.
"""
import torch
import json
import numpy as np

import os
import h5py

from tprism.util import TensorInfoMapper
from tprism.placeholder import PlaceholderData
from numpy import ndarray
from torch import Tensor
from typing import Any, Dict, Optional, Sequence, Tuple, TypedDict


EmbeddingData=Dict[str,ndarray]


class CycleEmbeddingEntry(TypedDict):
    tensor: PlaceholderData
    data: Tensor
    id: int

def load_embedding_data(filename: str, key: str, verb: bool) -> EmbeddingData:
    """Load input data supporting .h5/.json format

    Args:
        data_filename_list: list of input file names
    Returns:
        merged input data

    """
    _, ext = os.path.splitext(filename)
    dataset:EmbeddingData = {}
    if ext == ".h5":
        print("[LOAD]", filename)
        dataset = load_embedding_h5(filename, key, verb)
    elif ext == ".json":
        print("[LOAD]", filename)
        dataset = load_embedding_npy(filename, key, verb)
    else:
        print("[ERROR]", filename)
    return dataset

def load_embedding_h5(filename, key, verb)-> EmbeddingData:
    infh:Any = h5py.File(filename, "r")
    dataset:EmbeddingData = {}
    if key in infh:
        for vocab_name in infh[key]:
            rs = infh[key][vocab_name][()]
            dataset[vocab_name] = rs
            if verb:
                print("[LOAD DatasetEmbedding]", vocab_name)
    infh.close()
    return dataset


def load_embedding_npy(filename: str, key: str, verb: bool) -> EmbeddingData:
    fp = open(filename, "r")
    obj = json.load(fp)
    dataset:EmbeddingData = {}
    if key in obj["group"]:
        tensor=np.load(obj["filename"])
        name=obj["name"]
        dataset[name]=tensor
        if verb:
            print("[LOAD DatasetEmbedding]", name)
    return dataset


class BaseEmbeddingGenerator:
    def __init__(self):
        self.feed_verb = False
        self.get_verb  = False
        self.info_verb = False

    def is_embedding(self, vocab_name: str) -> bool:
        return False

    def get_shape(self, vocab_name: str) -> Tuple[int, ...]:
        return ()

    def get_embedding(self, vocab_name: str, shape: Optional[Tuple[int, ...]] = None, node_id = None) -> Optional[PlaceholderData]:
        return None

    def update(self, out_inside: Sequence[Tensor]) -> Tensor:
        return torch.tensor(0.0)
    
    def forward(self, name: str, shape: Sequence[int], node_id: int) -> Tensor:
        return torch.tensor(0.0)
    
    def build_feed(self, feed_dict: Dict[PlaceholderData, Tensor], idx: Optional[ndarray]=None) -> Dict[PlaceholderData, Tensor]:
        return feed_dict


class CycleEmbeddingGenerator(BaseEmbeddingGenerator):
    def __init__(self):
        super().__init__()
        self.embedding: Dict[str, CycleEmbeddingEntry] = {}
        self.tensor_shape: TensorInfoMapper = TensorInfoMapper()

    def load(self, tensor_shape: TensorInfoMapper) -> None:
        self.tensor_shape = tensor_shape

    def get_embedding(
        self,
        vocab_name: str,
        shape: Optional[Tuple[int, ...]] = None,
        node_id: Optional[int] = None,
    ) -> Optional[PlaceholderData]:
        return None

    def forward(self, name: str, shape: Sequence[int], node_id: int) -> Tensor:
        ph_name: str = name + "_cyc"
        shape_tuple: Tuple[int, ...] = tuple(shape)
        if ph_name in self.embedding:
            if self.get_verb:
                print("[GET cycle]>", ph_name, ":", self.embedding[ph_name]["tensor"])
            return torch.tensor(self.embedding[ph_name]["data"])
        else:
            if self.info_verb:
                print("[CREATE cycle]>", ph_name, ":", shape_tuple)
            self.embedding[ph_name] = {
                "tensor": PlaceholderData(
                    name=ph_name, shape=shape_tuple, dtype=torch.float32
                ),
                "data": torch.tensor(np.zeros(shape=shape_tuple, dtype=np.float32)),
                "id": node_id,
            }
            return torch.tensor(self.embedding[ph_name]["data"])

    def build_feed(self, feed_dict: Dict[PlaceholderData, Tensor], idx: Optional[ndarray]=None) -> Dict[PlaceholderData, Tensor]: ## idx is not used
        for ph_name, data in self.embedding.items():
            batch_data = data["data"]
            ph_var = data["tensor"]
            if self.feed_verb:
                print("[INFO: cycle feed]", "node_id:", data["id"], "=>", ph_name)
            feed_dict[ph_var] = torch.Tensor(batch_data)
        return feed_dict

    def update(self, out_inside: Sequence[Tensor]) -> Tensor:
        total_loss: Tensor = torch.tensor(0.0)
        for ph_name, data in self.embedding.items():
            node_id: int = data["id"]
            if self.info_verb:
                print("[INFO: cycle update] node_id:", node_id, "=>", ph_name)
            ##
            o = out_inside[node_id]
            loss = self.embedding[ph_name]["data"] - o
            total_loss += (loss ** 2).sum()
            ##
            self.embedding[ph_name]["data"] = o
            # a=0.5
            # self.embedding[ph_name]["data"]=(1.0-a)*self.embedding[ph_name]["data"]+a*out_inside[node_id]
        return total_loss


# embedding data from data
class EmbeddingGenerator(BaseEmbeddingGenerator):
    """ Generating embedding data from the given tensor atom
    
        1. If a given tensor atom is not shared by this generator, do nothing
        2. A placeholder name is computed by the given tensor atom
        3. If the placeholder already exists, return this placeholder.
        4. Create a placeholder and return it.

    Attributes:
        dataset (Dict[str,tensor]) : loaded dataset which is defined by dictionary from a tensor atom name to an assigned tensor.
        created_ph_var (Dict[str,PlaceholderData]) : assign
    """
    def __init__(self, const_flag:bool=False) -> None:
        super().__init__()
        self.dataset: EmbeddingData = {}
        self.created_ph_var: Dict[str, PlaceholderData] = {}
        self.const_flag=const_flag

    def load(self, filename: str, key: str="train") -> None:
        self.dataset=load_embedding_data(filename, key, self.info_verb)

    def is_embedding(self, vocab_name: str) -> bool:
        return vocab_name in self.dataset

    def get_shape(self, vocab_name: str) -> Tuple[int, ...]:
        return self.dataset[vocab_name].shape

    def get_embedding(self, vocab_name: str, shape: Optional[Tuple[int, ...]] =None, node_id = None) -> Optional[PlaceholderData]:
        if not self.is_embedding(vocab_name):
            if self.info_verb:
                print("[SKIP]>", vocab_name)
            return None
        ph_name = vocab_name + "_ph"
        if ph_name in self.created_ph_var:
            if self.get_verb:
                print("[GET]>", ph_name, ":", self.created_ph_var[ph_name])
            return self.created_ph_var[ph_name]
        else:
            if shape is None:
                shape = self.dataset[vocab_name].shape
            if self.const_flag:
                self.created_ph_var[ph_name] = PlaceholderData(
                    name=ph_name, shape=shape, dtype=torch.float32, ref=vocab_name
                )
            else:
                self.created_ph_var[ph_name] = PlaceholderData(
                    name=ph_name, shape=shape, dtype=torch.float32
                )
            if self.info_verb:
                if self.const_flag:
                    print("[CREATE const]>", ph_name, ":", shape)
                else:
                    print("[CREATE ]>", ph_name, ":", shape, "ref:", vocab_name)
            return self.created_ph_var[ph_name]
        return None
    
    def build_feed(self, feed_dict: Dict[PlaceholderData, Tensor], idx: Optional[ndarray]=None) -> Dict[PlaceholderData, Tensor]:
        for vocab_name, data in self.dataset.items():
            ph_name = vocab_name + "_ph"
            if idx is None or self.const_flag:
                batch_data = data
            else:
                batch_data = data[idx]
            if ph_name in self.created_ph_var:
                ph_var = self.created_ph_var[ph_name]
                feed_dict[ph_var] = torch.Tensor(batch_data)
            if self.feed_verb:
                print("[INFO: feed]", vocab_name, "=>", ph_name)
        return feed_dict
