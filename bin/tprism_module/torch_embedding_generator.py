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
from tprism_module.expl_graph import PlaceholderData

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
        ph_name = name + "_cyc"
        if ph_name in self.embedding:
            print("[GET]>", ph_name, ":", self.embedding[ph_name]["tensor"])
            return self.embedding[ph_name]["tensor"]
        else:
            print("[CREATE]>", ph_name, ":", shape)
            self.embedding[ph_name] = {}
            self.embedding[ph_name]["tensor"] = PlaceholderData(
                name=ph_name, shape=shape, dtype=torch.float32
            )
            self.embedding[ph_name]["data"] = np.zeros(shape=shape, dtype=np.float32)
            self.embedding[ph_name]["id"] = node_id
            return self.embedding[ph_name]["tensor"]

    def build_feed(self, feed_dict):
        for ph_name, data in self.embedding.items():
            batch_data = data["data"]
            ph_var = data["tensor"]
            if self.feed_verb:
                print("[INFO: feed]", "node_id:", data["id"], "=>", ph_name)
            feed_dict[ph_var] = batch_data
        return feed_dict

    def update(self, out_inside):
        total_loss = 0
        for ph_name, data in self.embedding.items():
            node_id = data["id"]
            print("[INFO: update] node_id:", node_id, "=>", ph_name)
            ##
            loss = self.embedding[ph_name]["data"] - out_inside[node_id]
            total_loss += np.sum(loss ** 2)
            ##
            self.embedding[ph_name]["data"] = out_inside[node_id]
            # a=0.5
            # self.embedding[ph_name]["data"]=(1.0-a)*self.embedding[ph_name]["data"]+a*out_inside[node_id]
        return total_loss

# embedding data from data
class DatasetEmbeddingGenerator(BaseEmbeddingGenerator):
    def __init__(self):
        self.feed_verb = False
        self.dataset = {}
        self.created_ph_var = {}
        self.vocabset_ph_var = None

    def load(self, filename, key="train"):
        print("[LOAD]", filename)
        infh = h5py.File(filename, "r")
        if key in infh:
            for vocab_name in infh[key]:
                rs = infh[key][vocab_name].value
                self.dataset[vocab_name] = rs
                print("[LOAD VOCAB]", vocab_name)
        infh.close()

    def is_embedding(self, vocab_name):
        return vocab_name in self.dataset

    def get_shape(self, vocab_name):
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
                shape=self.dataset[vocab_name].shape
            self.created_ph_var[ph_name] = PlaceholderData(
                name=ph_name, shape=shape, dtype=torch.float32
            )
            print("[CREATE]>", ph_name, ":", shape)
            return self.created_ph_var[ph_name]

    def build_feed(self, feed_dict, idx):
        for vocab_name, data in self.dataset.items():
            ph_name = vocab_name + "_ph"
            batch_data = data[idx]
            if ph_name in self.created_ph_var:
                ph_var = self.created_ph_var[ph_name]
                feed_dict[ph_var] = batch_data
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
                print("[LOAD VOCAB]", vocab_name)
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
                shape=self.dataset[vocab_name].shape
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
                feed_dict[ph_var] = data
            if self.feed_verb:
                print("[INFO: feed]", vocab_name, "=>", ph_name)
        return feed_dict




