#!/usr/bin/env python

import json
import os
import re
import numpy as np
from numpy import ndarray
from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union

from google.protobuf import json_format
from itertools import chain
import collections
import argparse
import time
import pickle
import h5py

import tprism.expl_pb2 as expl_pb2


def save_embedding_as_h5(filename: str, train_data: Dict[str,ndarray]={} , test_data:Dict[str,ndarray]={}):
    """ save embedding dataset as h5fs format

    Args:
        filename (str): h5 file name
        train_data (Dict[str,ndarray]): the key is the name of a tensor atom and the value is the numpy array associated with the tensor atom for training phase
        test_data (Dict[str,ndarray]): the key is the name of a tensor atom and the value is the numpy array associated with the tensor atom for test phase
    """
    with h5py.File(filename, "w") as fp:
        if train_data is not None:
            train_group = fp.create_group("train")
            for key, val in train_data.items():
                train_group.create_dataset(key, data=val)
        if test_data is not None:
            test_group = fp.create_group("test")
            for key, val in test_data.items():
                test_group.create_dataset(key, data=val)


def to_string_goal(goal):
    """
    s=goal.name
    s+="("
    s+=",".join([str(arg) for arg in goal.args])
    s+=")"
    """
    s = goal.name
    s += ","
    s += ",".join([str(arg) for arg in goal.args])
    return s


class Flags(object):
    """
    integrated information of args(command line aruguments) and flags automatically/manually setted in the T-PRISM program
    """
    def __init__(self, args={}, options=None, with_build=True):
        self.internal_config = dict()
        self.args = args
        if options is not None:
            self.flags = {f.key: f.value for f in options.flags}
        else:
            self.flags={}
        if with_build:
            self.build()

    def __getattr__(self, k):
        if k in self.internal_config:
            return  dict.get(self.internal_config, k)
        elif k in self.args:
            if not getattr(self.args, k, None):
                if k in self.flags:
                    return dict.get(self.flags, k)
            return getattr(self.args, k, None)
        return None
    def __contains__(self, k):
        """
        This function only checks for containment, so it may contain None even if this function returns True
        """
        if k in self.internal_config:
            return True
        elif k in self.args:
            return True
        elif k in self.flags:
            return True
        else:
            return False

    def add(self, k, v):
        self.internal_config[k] = v

    def build(self):
        check_items=[
            ("sgd_minibatch_size", 10, int),
            ("max_iterate", 100, int),
            ("sgd_learning_rate", 0.1, float),
                ]
        check_list_items=[
                "embedding",
                "const_embedding"]

        for k, default_v, vtype in check_items:
            v=getattr(self,k)
            if v is not None and v != "default":
                self.add(k,vtype(v))
            else:
                self.add(k,default_v)
        for k in check_list_items:
            v=getattr(self,k)
            if v is None or v == "default":
                self.add(k,[])


class TensorInfoMapper():
    def __init__(self, options=None,init_dict={}):
        self.shape={}
        self.type={}
        if options is not None:
            self.shape.update({
                el.tensor_name: [d for d in el.shape] for el in options.tensor_shape
            })
            self.type.update({
                el.tensor_name: el.type for el in options.tensor_shape
            })
    def __repr__(self):
      return "TensorInfoMapper("+self.shape.__repr__()+"  "+self.type.__repr__()+")"


def get_goal_dataset(goal_dataset):
    out_idx = []
    for j, goal in enumerate(goal_dataset):
        all_num = goal["dataset"].shape[1]
        all_idx = np.array(list(range(all_num)))
        out_idx.append(all_idx)
    return out_idx


def split_goal_dataset(goal_dataset, valid_ratio=0.1):
    train_idx = []
    valid_idx = []
    for j, goal in enumerate(goal_dataset):
        ph_vars = goal["placeholders"]
        all_num = goal["dataset"].shape[1]
        all_idx = np.array(list(range(all_num)))
        np.random.shuffle(all_idx)
        train_num = int(all_num - valid_ratio * all_num)
        train_idx.append(all_idx[:train_num])
        valid_idx.append(all_idx[train_num:])
    return train_idx, valid_idx


#
# goal_dataset["placeholders"] => ph_vars
# goal_dataset["dataset"]: dataset
# dataset contains indeces: values in the given dataset is coverted into index
def build_goal_dataset(input_data, tensor_provider):
    goal_dataset = []

    def to_index(value, ph_name):
        return tensor_provider.convert_value_to_index(value, ph_name)

    to_index_func = np.vectorize(to_index)
    for d in input_data:
        ph_names = d["placeholders"]
        # TODO: multiple with different placeholders
        ph_vars = [tensor_provider.ph_var[ph_name] for ph_name in ph_names]
        dataset = [None for _ in ph_names]
        goal_data = {"placeholders": ph_vars, "dataset": dataset}
        goal_dataset.append(goal_data)
        for i, ph_name in enumerate(ph_names):
            rec = d["records"]
            if tensor_provider.is_convertable_value(ph_name):
                print("[INFO]", ph_name, "converted!!")
                dataset[i] = to_index_func(rec[:, i], ph_name)
            else:  # goal placeholder
                dataset[i] = rec[:, i]
                print("[WARN] no conversion from values to indices:", ph_name)
                print("goal_placeholder?")
                print(rec.shape)
                print(ph_name)
            print("*")
    for obj in goal_dataset:
        obj["dataset"] = np.array(obj["dataset"])
    return goal_dataset
