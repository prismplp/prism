""" This module contains loaders related to tensors, explanationa graphs, operations, and loss functions.
"""

from __future__ import annotations
import json
import re
import numpy as np
from google.protobuf import json_format

import inspect
import importlib
import glob
import os
import re
import pickle
import h5py

import tprism.expl_pb2 as expl_pb2
import tprism.op.base
import tprism.loss.base
from numpy import int32, int64, ndarray, str_
from torch import Tensor, dtype
from torch.nn.parameter import Parameter
from tprism.loss.torch_standard_loss import Ce_pl2, PreferencePair
from tprism.op.torch_standard_op import Sigmoid
from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union

from tprism.placeholder import PlaceholderData
from tprism.torch_embedding_generator import DatasetEmbeddingGenerator


#[ {"goal_id": <int>, "placeholders": <List[str]>, "records": ndarray} ]
InputDataType = List[Dict[str, Union[int, List[str], ndarray]]]


def load_input_data(data_filename_list: List[str]) -> InputDataType:
    """Load input data supporting .h5/.json format

    Input data is a list of object like  [ {"goal_id": <int>, "placeholders": <List[str]>, "records": ndarray} ]

    Args:
        data_filename_list: list of input file names
    Returns:
        merged input data

    
    """
    input_data_list = []
    for filename in data_filename_list:
        _, ext = os.path.splitext(filename)
        if ext == ".h5":
            print("[LOAD]", filename)
            datasets = load_input_h5(filename)
        elif ext == ".json":
            print("[LOAD]", filename)
            datasets = load_input_json(filename)
        elif ext[:5] == ".json":
            print("[LOAD]", filename)
            datasets = load_input_json(filename)
        else:
            print("[ERROR]", data_filename)
        input_data_list.append(datasets)
    return merge_input_data(input_data_list)
    # return input_data_list


def load_input_json(filename: str) -> InputDataType:
    input_data = expl_pb2.PlaceholderData()
    with open(filename, "r") as fp:
        input_data = json_format.Parse(fp.read(), input_data)
    datasets = []
    for g in input_data.goals:
        phs = [ph.name for ph in g.placeholders]
        rs = []
        for r in g.records:
            rs.append([item for item in items])
        dataset = {"goal_id": g.id, "placeholders": phs, "records": rs}
        datasets.append(dataset)
    return datasets


def load_input_h5(filename: str) -> InputDataType:
    infh = h5py.File(filename, "r")
    datasets = []
    for k in infh:
        goal_id = int(k)
        phs = [ph.decode() for ph in infh[k]["data"].attrs.get("placeholders")]
        rs = infh[k]["data"].value
        dataset = {"goal_id": goal_id, "placeholders": phs, "records": rs}
        datasets.append(dataset)
    infh.close()
    return datasets


def merge_input_data(input_data_list: List[InputDataType]) -> InputDataType:
    merged_data = {}
    for datasets in input_data_list:
        for data in datasets:
            goal_id = data["goal_id"]
            if goal_id not in merged_data:
                merged_data[goal_id] = data
            else:
                merged_data[goal_id]["records"].extend(data[goal_id]["records"])
    return list(merged_data.values())


def load_explanation_graph(expl_filename: str, option_filename: str)-> Tuple[Any,Any]:
    """Load an explanation graph and options supporting .json format

    Args:
        expl_filename: explanation graph file names
        option_filename: option file names
    Returns:
        a tupple of graph and option object
    """
 
    graph = expl_pb2.ExplGraph()
    options = expl_pb2.Option()
    print("[LOAD]", expl_filename)
    with open(expl_filename, "r") as fp:
        graph = json_format.Parse(fp.read(), graph)
    # f = open("expl.bin", "rb")
    # graph.ParseFromString(f.read())
    with open(option_filename, "r") as fp:
        options = json_format.Parse(fp.read(), options)
    return graph, options


class OperatorLoader:
    def __init__(self) -> None:
        self.operators = {}
        self.base_module_name = "tprism.op."
        self.module = None

    # a snake case operator name to class name
    def to_class_name(self, snake_str):
        components = snake_str.split("_")
        return "".join(x.title() for x in components)

    # class name to a snake case operator name
    def to_op_name(self, cls_name: str) -> str:
        _underscorer1 = re.compile(r"(.)([A-Z][a-z]+)")
        _underscorer2 = re.compile("([a-z0-9])([A-Z])")
        subbed = _underscorer1.sub(r"\1_\2", cls_name)
        return _underscorer2.sub(r"\1_\2", subbed).lower()

    def get_operator(self, name: str) -> Type[Sigmoid]:
        assert name in self.operators, "%s is not found" % (name)
        cls = self.operators[name]
        assert cls is not None, "%s is not found" % (name)
        return cls

    def load_all(self, path: str) -> None:
        search_path = os.path.dirname(__file__) + "/" + path
        for fpath in glob.glob(search_path + "*.py"):
            print("[LOAD]", fpath)
            name = os.path.basename(os.path.splitext(fpath)[0])
            module_name = self.base_module_name + name
            module = importlib.machinery.SourceFileLoader(
                module_name, fpath
            ).load_module()
            self.load_module(module)

    def load_module(self, module):
        for cls_name, cls in inspect.getmembers(module, inspect.isclass):
            if issubclass(cls, tprism.op.base.BaseOperator):
                print("[IMPORT]", cls_name)
                op_name = self.to_op_name(cls_name)
                self.operators[op_name] = cls


class LossLoader:
    def __init__(self) -> None:
        self.module = None
        self.base_module_name = "tprism.loss."
        self.losses = {}

    # a snake case operator name to class name
    def to_class_name(self, snake_str):
        components = snake_str.split("_")
        return "".join(x.title() for x in components)

    # class name to a snake case operator name
    def to_op_name(self, cls_name: str) -> str:
        _underscorer1 = re.compile(r"(.)([A-Z][a-z]+)")
        _underscorer2 = re.compile("([a-z0-9])([A-Z])")
        subbed = _underscorer1.sub(r"\1_\2", cls_name)
        return _underscorer2.sub(r"\1_\2", subbed).lower()

    def get_loss(self, name: str) -> Union[Type[Ce_pl2], Type[PreferencePair]]:
        if name in self.losses:
            cls = self.losses[name]
            return cls
        else:
            return None

    def load_all(self, path: str) -> None:
        search_path = os.path.dirname(__file__) + "/" + path
        for fpath in glob.glob(search_path + "*.py"):
            print("[LOAD]", fpath)
            name = os.path.basename(os.path.splitext(fpath)[0])
            module_name = self.base_module_name + name
            module = importlib.machinery.SourceFileLoader(
                module_name, fpath
            ).load_module()
            self.load_module(module)

    def load_module(self, module):
        for cls_name, cls in inspect.getmembers(module, inspect.isclass):
            if issubclass(cls, tprism.loss.base.BaseLoss):
                print("[IMPORT]", cls_name)
                op_name = self.to_op_name(cls_name)
                self.losses[op_name] = cls


