""" This module contains loaders related to tensors, explanationa graphs, operations, and loss functions.
"""

from __future__ import annotations
import json
import logging
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
import tprism.parser
from numpy import int32, int64, ndarray, str_
from torch import Tensor, dtype
from torch.nn.parameter import Parameter
from tprism.op.base import BaseOperator
from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union, cast

from tprism.placeholder import PlaceholderData
from tprism.embedding_generator import EmbeddingGenerator
from tprism.util import Flags, TensorInfoMapper, InputData, debug_logger
from tprism.loss import BaseLoss

logger = logging.getLogger(__name__)
module_logger = debug_logger("module")


def load_input_data(data_filename_list: List[str]) -> List[InputData]:
    """Load input data supporting .h5/.json format

    Input data is a list of object like  [ {"goal_id": goal_id: int, "placeholders": [paceholder1: str, ...], "records": ndarray} ]
    
    Args:
        data_filename_list: list of input file names
    Returns:
        merged input data

    
    """
    input_data_list:List[List[InputData]] = []
    for filename in data_filename_list:
        rest_name, ext = os.path.splitext(filename)
        if ext == ".h5":
            logger.info("[LOAD] %s", filename)
            datasets = load_input_h5(filename)
        elif ext == ".json":
            _, ext2 = os.path.splitext(rest_name)
            if ext2==".npy":
                logger.info("[LOAD] %s", filename)
                datasets = load_input_npy(filename)
            else:
                logger.info("[LOAD] %s", filename)
                datasets = load_input_json(filename)
        elif ext[:5] == ".json": # e.g. .json0_1
            logger.info("[LOAD] %s", filename)
            datasets = load_input_json(filename)
        else:
            logger.error("unknown format: %s", filename)
            datasets = []
        input_data_list.append(datasets)
    return merge_input_data(input_data_list)
    # return input_data_list


def load_input_npy(filename: str) -> List[InputData]:
    datasets = []
    with open(filename, "r") as fp:
        obj = json.load(fp)
    for goal_id, o in enumerate(obj):
        rs = np.load(o["filename"])
        dataset = InputData(goal_id,o["placeholders"],rs)
        datasets.append(dataset)
    return datasets

def load_input_json(filename: str) -> List[InputData]:
    input_data = expl_pb2.PlaceholderData()
    
    
    with open(filename, "r") as fp:
        input_data = json_format.Parse(fp.read(), input_data)
    datasets = []
    for g in input_data.goals:
        phs = [ph.name for ph in g.placeholders]
        rs = []
        for r in g.records:
            # TODO: current version only support integer records
            rs.append([int(item) for item in r.items])
        dataset = InputData(g.id, phs, np.array(rs))
        datasets.append(dataset)
    return datasets


def load_input_h5(filename: str) -> List[InputData]:
    infh = h5py.File(filename, "r")
    datasets = []
    for k in infh.keys():
        goal_id = int(k)
        # h5py <= 2.9.0
        #phs = [ph.decode() for ph in infh[k]["data"].attrs.get("placeholders")]
        #rs = infh[k]["data"].value
        # h5py >= 3.0.0
        phs_infh=infh[str(k)+"/data"].attrs.get("placeholders")
        if phs_infh is not None and hasattr(phs_infh, "__iter__"):
            phs = [ph for ph in phs_infh]
        else:
            raise ValueError("placeholders %s is not valid" % (k))
        #phs = [ph for ph in infh[k]["data"].attrs.get("placeholders")]
        rs = infh[str(k)+"/data"][()] # type: ignore
        dataset = InputData(goal_id, phs, np.array(rs))
        datasets.append(dataset)
    infh.close()
    return datasets


def merge_input_data(input_data_list: List[List[InputData]]) ->  List[InputData]:
    merged_data = {}
    for datasets in input_data_list:
        for data in datasets:
            goal_id = data.goal_id
            if goal_id not in merged_data:
                merged_data[goal_id] = data
            else:
                # merged_data[goal_id].records.extend(data.records)
                # Merge numpy arrays along the first axis
                try:
                    merged_data[goal_id].records = np.concatenate(
                        (merged_data[goal_id].records, data.records), axis=0
                    )
                except ValueError:
                    # Fallback to vstack for compatible shapes (e.g., 1-D arrays)
                    merged_data[goal_id].records = np.vstack(
                        (merged_data[goal_id].records, data.records)
                    )
    return list(merged_data.values())


def load_explanation_graph(expl_filename: str, option_filename: Optional[str] =None, args={})-> Tuple[expl_pb2.ExplGraph,TensorInfoMapper,Flags]:
    """Load an explanation graph and options supporting .json format

    Args:
        expl_filename: explanation graph file names
        option_filename: option(flag) file names
    Returns:
        a tupple of graph, Flags, TensorInfoMapper objects
    """
 
    graph = expl_pb2.ExplGraph()
    options = expl_pb2.Option()
    logger.info("[LOAD] %s", expl_filename)
    with open(expl_filename, "r") as fp:
        graph = json_format.Parse(fp.read(), graph)
    # f = open("expl.bin", "rb")
    # graph.ParseFromString(f.read())
    if option_filename is not None:
        logger.info("[LOAD] %s", option_filename)
        with open(option_filename, "r") as fp:
            options = json_format.Parse(fp.read(), options)
    #
    flags = Flags()
    flags.build(args, options)
    tensor_shapes = TensorInfoMapper(options)
    return graph, tensor_shapes, flags


class PluginLoader:
    """Generic plugin loader.

    Scans .py files in a directory and registers every subclass of
    `base_class` found in them under its snake-case class name.
    Subclasses set `base_class` and `base_module_name`.
    """
    base_class: type = object
    base_module_name: str = ""

    def __init__(self) -> None:
        self.plugins: Dict[str, Any] = {}

    # a snake case plugin name to class name
    def to_class_name(self, snake_str: str) -> str:
        components = snake_str.split("_")
        return "".join(x.title() for x in components)

    # class name to a snake case plugin name
    def to_op_name(self, cls_name: str) -> str:
        _underscorer1 = re.compile(r"(.)([A-Z][a-z]+)")
        _underscorer2 = re.compile("([a-z0-9])([A-Z])")
        subbed = _underscorer1.sub(r"\1_\2", cls_name)
        return _underscorer2.sub(r"\1_\2", subbed).lower()

    def load_all(self, path: str) -> None:
        search_path = os.path.dirname(__file__) + "/" + path
        self.load_all_from_search_path(search_path)

    def load_all_from_search_path(self, search_path: str) -> None:
        module_logger.debug("[LOAD] %s", search_path)
        for fpath in glob.glob(os.path.join(search_path, "*.py")):
            name = os.path.basename(os.path.splitext(fpath)[0])
            if name in ("__init__", "base"):
                continue
            self.load(fpath)

    def load(self, fpath: str) -> None:
        module_logger.debug("[LOAD] %s", fpath)
        name = os.path.basename(os.path.splitext(fpath)[0])
        module_name = self.base_module_name + name
        module = importlib.machinery.SourceFileLoader(
            module_name, fpath
        ).load_module()
        self.load_module(module)

    def load_module(self, module) -> None:
        for cls_name, cls in inspect.getmembers(module, inspect.isclass):
            if cls.__module__ != module.__name__:
                continue
            if issubclass(cls, self.base_class) and cls is not self.base_class:
                module_logger.debug("[IMPORT] %s", cls_name)
                op_name = self.to_op_name(cls_name)
                self.plugins[op_name] = cls
            else:
                module_logger.debug("[SKIP] %s (%s is not a subclass of %s)", cls_name, cls, self.base_class)

    def set_cls(self, op_name, cls) -> None:
        self.plugins[op_name] = cls


class OperatorLoader(PluginLoader):
    """
    This class is used to load custom operators
    """
    base_class = BaseOperator
    base_module_name = "tprism.op."

    @property
    def operators(self) -> Dict[str, Type[BaseOperator]]:
        return self.plugins

    def get_operator(self, name: str) -> Type[BaseOperator]:
        assert name in self.plugins, "%s is not found" % (name)
        cls = self.plugins[name]
        assert cls is not None, "%s is not found" % (name)
        return cls


class LossLoader(PluginLoader):
    """
    This class is used to load custom loss functions
    """
    base_class = BaseLoss
    base_module_name = "tprism.loss."

    @property
    def losses(self) -> Dict[str, Type[BaseLoss]]:
        return self.plugins

    def get_loss(self, name: str) -> Tuple[Optional[Type[BaseLoss]], List[str]]:
        """Return the loss class and its parameters for a value of the sgd_loss flag.

        The value is a Prolog-like term parsed by `tprism.parser.parse_term`:
        a loss name (``ce``), or a loss name with arguments (``ce(0.1)``,
        ``ce_pl($placeholder2$)``).  Each argument is passed to the loss class
        as a string; quoted atoms (``'...'``) are unquoted.

        Args:
            name: Value of the sgd_loss flag (``--sgd_loss`` or flags.json).

        Returns:
            A tuple (loss class or None if not found, list of parameters).
        """
        try:
            term = tprism.parser.parse_term(name)
        except SyntaxError as e:
            raise ValueError("cannot parse the loss function %r: %s" % (name, e)) from e
        if isinstance(term, dict) and "name" in term and "args" in term:
            loss_name = term["name"]
            loss_params = [_loss_parameter(arg) for arg in term["args"]]
        elif isinstance(term, str):
            loss_name = term
            loss_params = []
        else:
            raise ValueError("invalid loss function: %r" % (name,))
        if loss_name in self.plugins:
            cls = self.plugins[loss_name]
            return cls, loss_params
        else:
            # base_loss is the default of the sgd_loss flag (no loss function)
            if loss_name != "base_loss":
                logger.warning("unknown loss function: %s (available: %s)",
                               loss_name, ", ".join(sorted(self.plugins)))
            return None, loss_params


def _loss_parameter(arg: Any) -> str:
    """Convert an argument of a parsed loss term into the string given to a loss class."""
    if isinstance(arg, str):
        if len(arg) >= 2 and arg[0] == arg[-1] and arg[0] in ("'", '"'):
            return arg[1:-1]
        return arg
    return tprism.parser.serialize_term(arg)

def check_loss():
    loss_loader = LossLoader()
    loss_loader.load_all("loss/")
    print(loss_loader.losses)
    loss_cls, loss_params = loss_loader.get_loss("ce")
    if loss_cls is not None:
        print(loss_cls, loss_params)
    else:
        print("loss function not found")

def check_operators():
    operator_loader = OperatorLoader()
    operator_loader.load_all("op/")
    print(operator_loader.operators)
    cls = operator_loader.get_operator("sigmoid")
    print(cls)
    
def main():
    check_loss()
    check_operators()

if __name__ == "__main__":
    main()
