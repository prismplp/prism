import torch
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

import tprism_module.expl_pb2 as expl_pb2
import tprism_module.op.base
import tprism_module.loss.base
from tprism_module.placeholder import PlaceholderData
from numpy import ndarray
from torch import Tensor
from typing import Dict, Optional, Tuple


class BaseEmbeddingGenerator:
    def is_embedding(self, vocab_name):
        return False

    def get_shape(self, vocab_name):
        return None

    def get_embedding(self, name, shape, node_id):
        return None

    def update(self, out_inside):
        pass


class CycleEmbeddingGenerator(BaseEmbeddingGenerator):
    def __init__(self):
        self.embedding = {}
        self.index_range = {}
        self.tensor_shape = {}
        self.feed_verb = False

    def load(self, options):
        self.index_range = {el.index: el.range for el in options.index_range}
        self.tensor_shape = {
            el.tensor_name: [d for d in el.shape] for el in options.tensor_shape
        }

    def template2shape(self, template):
        return [self.index_range[t] for t in template]

    def get_embedding(self, name, shape, node_id):
        return None

    def forward(self, name, shape, node_id):
        ph_name = name + "_cyc"
        if ph_name in self.embedding:
            print("[GET cycle]>", ph_name, ":", self.embedding[ph_name]["tensor"])
            return torch.tensor(self.embedding[ph_name]["data"])
        else:
            print("[CREATE cycle]>", ph_name, ":", shape)
            self.embedding[ph_name] = {}
            self.embedding[ph_name]["tensor"] = PlaceholderData(
                name=ph_name, shape=shape, dtype=torch.float32
            )
            self.embedding[ph_name]["data"] = torch.tensor(
                np.zeros(shape=shape, dtype=np.float32)
            )
            self.embedding[ph_name]["id"] = node_id
            return torch.tensor(self.embedding[ph_name]["data"])

    def build_feed(self, feed_dict, idx=None):  ## idx is not used
        for ph_name, data in self.embedding.items():
            batch_data = data["data"]
            ph_var = data["tensor"]
            if self.feed_verb:
                print("[INFO: cycle feed]", "node_id:", data["id"], "=>", ph_name)
            feed_dict[ph_var] = torch.Tensor(batch_data)
        return feed_dict

    def update(self, out_inside):
        total_loss = 0
        for ph_name, data in self.embedding.items():
            node_id = data["id"]
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

class DatasetEmbeddingGenerator(BaseEmbeddingGenerator):
    def __init__(self) -> None:
        self.feed_verb = False
        self.dataset = {}
        self.created_ph_var = {}
        self.vocabset_ph_var = None

    def load(self, filename: str, key: str="train") -> None:
        print("[LOAD]", filename)
        infh = h5py.File(filename, "r")
        if key in infh:
            for vocab_name in infh[key]:
                rs = infh[key][vocab_name][()]
                self.dataset[vocab_name] = rs
                print("[LOAD DatasetEmbedding]", vocab_name)
        infh.close()

    def is_embedding(self, vocab_name: str) -> bool:
        return vocab_name in self.dataset

    def get_shape(self, vocab_name: str) -> Tuple[int, int]:
        return self.dataset[vocab_name].shape

    def get_embedding(self, vocab_name: str, shape: None=None) -> PlaceholderData:
        if not self.is_embedding(vocab_name):
            print("[SKIP]>", vocab_name)
            return None
        ph_name = vocab_name + "_ph"
        if ph_name in self.created_ph_var:
            print("[GET]>", ph_name, ":", self.created_ph_var[ph_name])
            return self.created_ph_var[ph_name]
        else:
            if shape is None:
                shape = self.dataset[vocab_name].shape
            self.created_ph_var[ph_name] = PlaceholderData(
                name=ph_name, shape=shape, dtype=torch.float32, ref=vocab_name
            )
            print("[CREATE]>", ph_name, ":", shape, "ref:", vocab_name)
            return self.created_ph_var[ph_name]

    def build_feed(self, feed_dict: Dict[PlaceholderData, Tensor], idx: Optional[ndarray]=None) -> Dict[PlaceholderData, Tensor]:
        for vocab_name, data in self.dataset.items():
            ph_name = vocab_name + "_ph"
            if idx is None:
                batch_data = data
            else:
                batch_data = data[idx]
            if ph_name in self.created_ph_var:
                ph_var = self.created_ph_var[ph_name]
                feed_dict[ph_var] = torch.Tensor(batch_data)
        return feed_dict


# embedding data from data
class ConstEmbeddingGenerator(BaseEmbeddingGenerator):
    def __init__(self):
        self.feed_verb = False
        self.dataset = {}
        self.created_ph_var = {}

    def load(self, filename, key="train"):
        print("[LOAD]", filename)
        infh = h5py.File(filename, "r")
        if key in infh:
            for vocab_name in infh[key]:
                rs = infh[key][vocab_name].value
                self.dataset[vocab_name] = rs
                print("[LOAD ConstEmbedding]", vocab_name)
        infh.close()

    def is_embedding(self, vocab_name):
        return vocab_name in self.dataset

    def get_dataset_shape(self, vocab_name):
        return self.dataset[vocab_name].shape

    def get_embedding(self, vocab_name, shape=None):
        if not self.is_embedding(vocab_name):
            print("[SKIP]>", vocab_name)
            return None
        ph_name = vocab_name + "_ph"
        if ph_name in self.created_ph_var:
            print("[GET]>", ph_name, ":", self.created_ph_var[ph_name])
            return self.created_ph_var[ph_name]
        else:
            if shape is None:
                shape = self.dataset[vocab_name].shape
            self.created_ph_var[ph_name] = PlaceholderData(
                name=ph_name, shape=shape, dtype=torch.float32
            )
            print("[CREATE]>", ph_name, ":", shape)
            return self.created_ph_var[ph_name]

    def build_feed(self, feed_dict, idx=None):
        for vocab_name, data in self.dataset.items():
            ph_name = vocab_name + "_ph"
            if ph_name in self.created_ph_var:
                ph_var = self.created_ph_var[ph_name]
                feed_dict[ph_var] = torch.Tensor(data)
            if self.feed_verb:
                print("[INFO: feed]", vocab_name, "=>", ph_name)
        return feed_dict
