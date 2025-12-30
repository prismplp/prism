#!/usr/bin/env python

import numpy as np
from numpy import ndarray
from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union

import h5py

import tprism.expl_pb2 as expl_pb2

import dataclasses

#[ {"goal_id": <int>, "placeholders": <List[str]>, "records": ndarray} ]
@dataclasses.dataclass
class InputData:
    goal_id: int
    placeholders: List[str] = dataclasses.field(default_factory=list)
    records: ndarray = dataclasses.field(default_factory=lambda: np.array([], dtype=np.int32))


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

@dataclasses.dataclass(slots=True)
class Flags:
    # data / dataset
    dataset: Optional[List[str]] = None
    # intermediate data
    input: Optional[str] = None
    # prolog related
    expl_graph: Optional[str] = None
    # model / vocab
    model: Optional[str] = None
    vocab: Optional[str] = None
    # embeddings
    embedding: Optional[List[str]] = None
    const_embedding: Optional[List[str]]  = None

    # graph / output
    output: Optional[str] = None

    # SGD / training (prolog flags)
    sgd_minibatch_size: int = 1
    max_iterate: int = 10
    sgd_learning_rate: float = 0.01
    sgd_loss: str = "base_loss"
    sgd_patience: int = 3

    # others
    cycle: bool = False
    verbose: bool = False
    def __init__(self, args: Any = None, options: Any = None):
        if args is not None:
           args_dict=vars(args)
        if options is not None:
            flags = {f.key: f.value for f in options.flags}
        else:
            flags = {}
        self.build(args_dict, flags)
            
    def __contains__(self, k: str) -> bool:
        return hasattr(self, k) and getattr(self, k) is not None
    def add(self, k: str, v: Any) -> None:
        if hasattr(self, k):
            setattr(self, k, v)
    def build(self, args_dict, flags) -> None:
        for k, v in args_dict.items():
            if v is not None:
                self.add(k, v)
        for k, v in flags.items():
            if v is not None:
                self.add(k, v)

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
def build_goal_dataset(input_data: List[InputData], tensor_provider):
    goal_dataset: List[Dict[str, Any]] = []

    def to_index(value, ph_name):
        return tensor_provider.convert_value_to_index(value, ph_name)

    to_index_func = np.vectorize(to_index)
    for d in input_data:
        ph_names = d.placeholders
        # TODO: multiple with different placeholders
        ph_vars = [tensor_provider.ph_var[ph_name] for ph_name in ph_names]
        dataset: List[Any] = [None for _ in ph_names]
        goal_data = {"placeholders": ph_vars, "dataset": dataset}
        goal_dataset.append(goal_data)
        for i, ph_name in enumerate(ph_names):
            rec = d.records
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
