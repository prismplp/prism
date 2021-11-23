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

from tprism.expl_graph import ComputationalExplGraph, SwitchTensorProvider
from tprism.expl_graph import PlaceholderGraph, VocabSet
from tprism.loader import OperatorLoader
from tprism.placeholder import PlaceholderData
from numpy import int64
from torch import dtype
from typing import Any, Dict, List, Tuple, Union


class TorchComputationalExplGraph(ComputationalExplGraph, torch.nn.Module):
    """ This class is a concrete explanation graph for pytorch

    Note:
        aaaaa

        ::
        
            {
                "sw_template": [],
                "sw_inside": [],
                "prob_sw_inside": [],
                "node_template": [],
                "node_inside": [],
                "node_scalar_inside": [],
            }

            
    """
    def __init__(self, graph, tensor_provider, cycle_embedding_generator=None):
        torch.nn.Module.__init__(self)

        operator_loader = OperatorLoader()
        operator_loader.load_all("op/torch_")
        goal_template, cycle_node = self.build_explanation_graph_template(
            graph, tensor_provider, operator_loader
        )
        self.operator_loader = operator_loader
        self.goal_template = goal_template
        self.cycle_node = cycle_node
        self.graph = graph
        self.loss = {}
        self.tensor_provider = tensor_provider
        self.cycle_embedding_generator = cycle_embedding_generator

        for name, val in tensor_provider.params.items():
            self.register_parameter(name, val)

    def _distribution_forward(self, name, dist, params, param_template, op):
        if dist == "normal":
            mean = params[0]
            var = params[1]
            scale = torch.sqrt(F.softplus(var))
            q_z = torch.distributions.normal.Normal(mean, scale)
            out_inside = q_z.rsample()
            out_template = param_template[0]
            p_z = torch.distributions.normal.Normal(
                torch.zeros_like(mean), torch.ones_like(var)
            )
            loss_KL = torch.distributions.kl.kl_divergence(q_z, p_z)
            self.loss[name] = loss_KL.sum()
        else:
            out_inside = params[0]
            out_template = param_template[0]
            print("[ERROR] unknown distribution:", dist)
        return out_inside, out_template

    def forward_(self, verbose: bool=False) -> Union[Tuple[List[Dict[str, Union[List[str], List[Tuple[str, str, List[Tuple[str, str]]]], bool]]], Dict[Any, Any]], Tuple[List[Union[Dict[str, Union[List[str], List[Tuple[str, str, List[Tuple[str, str]]]], bool]], Dict[str, Union[List[str], List[Tuple[str, str, Tuple[str, str, List[Union[Tuple[str, str], Tuple[str, int]]]]]], bool]], Dict[str, Union[List[str], List[Tuple[str, str, List[Union[Tuple[str, str], Tuple[str, int]]]]], bool]]]], Dict[Any, Any]]]:
        graph = self.graph
        tensor_provider = self.tensor_provider
        cycle_embedding_generator = self.cycle_embedding_generator
        goal_template = self.goal_template
        cycle_node = self.cycle_node
        operator_loader = self.operator_loader
        self.loss = {}
        # goal_template
        # converting explanation graph to computational graph
        goal_inside = [None] * len(graph.goals)
        for i in range(len(graph.goals)):
            g = graph.goals[i]
            if verbose:
                print(
                    "=== tensor equation (node_id:%d, %s) ==="
                    % (g.node.sorted_id, g.node.goal.name)
                )
            path_inside = []
            path_template = []
            path_batch_flag = False
            for path in g.paths:
                path_data = {
                    "sw_template": [],
                    "sw_inside": [],
                    "prob_sw_inside": [],
                    "node_template": [],
                    "node_inside": [],
                    "node_scalar_inside": [],
                }
                ## build template and inside for switches in the path
                for sw in path.tensor_switches:
                    ph = ("tensor_provider.get_placeholder_name", sw.name)
                    if len(ph) > 0:
                        path_data["sw_template"].append(["b"] + list(sw.values))
                        path_batch_flag = True
                    else:
                        path_data["sw_template"].append(list(sw.values))
                    sw_var = ("tensor_provider.get_embedding", sw.name)
                    path_data["sw_inside"].append(sw_var)
                for sw in path.prob_switches:
                    path_data["prob_sw_inside"].append(("switch", sw.inside))

                ## building template and inside for nodes in the path
                for node in path.nodes:
                    temp_goal = goal_inside[node.sorted_id]
                    temp_goal_id = ("goal_inside", node.sorted_id)
                    if node.sorted_id in cycle_node:
                        name = node.goal.name
                        template = goal_template[node.sorted_id]["template"]
                        shape = goal_template[node.sorted_id]["shape"]
                        temp_goal_inside = (
                            "cycle_embedding_generator",
                            name,
                            shape,
                            node.sorted_id,
                        )
                        temp_goal_template = template
                        path_data["node_inside"].append(temp_goal_inside)
                        path_data["node_template"].append(temp_goal_template)
                    elif temp_goal is None:
                        print("  [ERROR] cycle node is detected")
                        temp_goal = goal_inside[node.sorted_id]
                        print(g.node.sorted_id)
                        print(node)
                        print(node.sorted_id)
                        print(temp_goal)
                        quit()
                    elif len(temp_goal["template"]) > 0:
                        # tensor
                        temp_goal_inside = temp_goal_id
                        temp_goal_template = temp_goal["template"]
                        if temp_goal["batch_flag"]:
                            path_batch_flag = True
                        path_data["node_inside"].append(temp_goal_inside)
                        path_data["node_template"].append(temp_goal_template)
                    else:  # scalar
                        path_data["node_scalar_inside"].append(temp_goal_id)
                ## building template and inside for all elements (switches and nodes) in the path
                sw_node_template = path_data["sw_template"] + path_data["node_template"]
                sw_node_inside = path_data["sw_inside"] + path_data["node_inside"]

                ops = {op.name: op for op in path.operators}
                if "distribution" in ops:
                    op = ops["distribution"]
                    dist = op.values[0]
                    name = g.node.goal.name
                    out_inside, out_template = self._distribution_forward_(
                        name,
                        dist,
                        params=sw_node_inside,
                        param_template=sw_node_template,
                        op=op,
                    )
                else:
                    path_v = sorted(
                        zip(sw_node_template, sw_node_inside), key=lambda x: x[0]
                    )
                    template = [x[0] for x in path_v]
                    inside = [x[1] for x in path_v]
                    # constructing einsum operation using template and inside
                    out_template = self._compute_output_template(template)
                    if len(template) > 0:  # condition for einsum
                        lhs = ",".join(map(lambda x: "".join(x), template))
                        rhs = "".join(out_template)
                        if path_batch_flag:
                            rhs = "b" + rhs
                            out_template = ["b"] + out_template
                        einsum_eq = lhs + "->" + rhs
                        if verbose:
                            print("  index:", einsum_eq)
                            print("  var. :", inside)
                        out_inside = ("torch.einsum", einsum_eq, inside)
                    ## computing operaters
                    for op in path.operators:
                        if verbose:
                            print("  operator:", op.name)
                        cls = operator_loader.get_operator(op.name)
                        op_obj = cls(op.values)
                        out_inside = ("operator", op.name, out_inside)
                        out_template = op_obj.get_output_template(out_template)
                ##
                path_inside.append(out_inside)
                path_template.append(out_template)
                ##
            ##
            path_template_list = self._get_unique_list(path_template)
            if len(path_template_list) == 0:
                goal_inside[i] = {"template": [], "inside": 1, "batch_flag": False}
            else:
                if len(path_template_list) != 1:
                    print("[WARNING] missmatch indices:", path_template_list)
                if len(path_template_list[0]) == 0:
                    goal_inside[i] = {
                        "template": path_template_list[0],
                        "inside": path_inside,
                        "batch_flag": path_batch_flag,
                    }
                else:
                    goal_inside[i] = {
                        "template": path_template_list[0],
                        "inside": path_inside,
                        "batch_flag": path_batch_flag,
                    }
        return goal_inside, self.loss

    def forward(self, verbose=False):
        graph = self.graph
        tensor_provider = self.tensor_provider
        cycle_embedding_generator = self.cycle_embedding_generator
        goal_template = self.goal_template
        cycle_node = self.cycle_node
        operator_loader = self.operator_loader
        self.loss = {}
        # goal_template
        # converting explanation graph to computational graph
        goal_inside = [None] * len(graph.goals)
        for i in range(len(graph.goals)):
            g = graph.goals[i]
            if verbose:
                print(
                    "=== tensor equation (node_id:%d, %s) ==="
                    % (g.node.sorted_id, g.node.goal.name)
                )
            path_inside = []
            path_template = []
            path_batch_flag = False
            for path in g.paths:
                ## build template and inside for switches in the path
                sw_template = []
                sw_inside = []
                for sw in path.tensor_switches:
                    ph = tensor_provider.get_placeholder_name(sw.name)
                    if len(ph) > 0:
                        sw_template.append(["b"] + list(sw.values))
                        path_batch_flag = True
                    else:
                        sw_template.append(list(sw.values))
                    sw_var = tensor_provider.get_embedding(sw.name, verbose)
                    sw_inside.append(sw_var)
                prob_sw_inside = torch.tensor(1.0)
                for sw in path.prob_switches:
                    prob_sw_inside *= sw.inside

                ## building template and inside for nodes in the path
                node_template = []
                node_inside = []
                node_scalar_inside = []
                for node in path.nodes:
                    temp_goal = goal_inside[node.sorted_id]

                    if node.sorted_id in cycle_node:
                        name = node.goal.name
                        template = goal_template[node.sorted_id]["template"]
                        shape = goal_template[node.sorted_id]["shape"]
                        # shape=cycle_embedding_generator.template2shape(template)
                        temp_goal_inside = cycle_embedding_generator.forward(
                            name, shape, node.sorted_id
                        )
                        temp_goal_template = template
                        node_inside.append(temp_goal_inside)
                        node_template.append(temp_goal_template)
                    elif temp_goal is None:
                        print("  [ERROR] cycle node is detected")
                        temp_goal = goal_inside[node.sorted_id]
                        print(g.node.sorted_id)
                        print(node)
                        print(node.sorted_id)
                        print(temp_goal)
                        quit()
                    elif len(temp_goal["template"]) > 0:
                        # tensor
                        temp_goal_inside = temp_goal["inside"]
                        temp_goal_template = temp_goal["template"]
                        if temp_goal["batch_flag"]:
                            path_batch_flag = True
                        node_inside.append(temp_goal_inside)
                        node_template.append(temp_goal_template)
                    else:  # scalar
                        if type(temp_goal["inside"]) is list:
                            a = torch.tensor(temp_goal["inside"])
                            node_scalar_inside.append(torch.squeeze(a))
                        else:
                            node_scalar_inside.append(temp_goal["inside"])
                ## building template and inside for all elements (switches and nodes) in the path
                sw_node_template = sw_template + node_template
                sw_node_inside = sw_inside + node_inside

                ops = {op.name: op for op in path.operators}
                if "distribution" in ops:
                    op = ops["distribution"]
                    dist = op.values[0]
                    name = g.node.goal.name
                    out_inside, out_template = self._distribution_forward(
                        name,
                        dist,
                        params=sw_node_inside,
                        param_template=sw_node_template,
                        op=op,
                    )
                else:
                    path_v = sorted(
                        zip(sw_node_template, sw_node_inside), key=lambda x: x[0]
                    )
                    template = [x[0] for x in path_v]
                    inside = [x[1] for x in path_v]
                    # constructing einsum operation using template and inside
                    out_template = self._compute_output_template(template)
                    # print(template,out_template)
                    out_inside = prob_sw_inside
                    if len(template) > 0:  # condition for einsum
                        lhs = ",".join(map(lambda x: "".join(x), template))
                        rhs = "".join(out_template)
                        if path_batch_flag:
                            rhs = "b" + rhs
                            out_template = ["b"] + out_template
                        einsum_eq = lhs + "->" + rhs
                        if verbose:
                            print("  index:", einsum_eq)
                            print("  var. :", inside)
                        out_inside = torch.einsum(einsum_eq, *inside) * out_inside
                    for scalar_inside in node_scalar_inside:
                        out_inside = scalar_inside * out_inside
                    ## computing operaters
                    for op in path.operators:
                        if verbose:
                            print("  operator:", op.name)
                        cls = operator_loader.get_operator(op.name)
                        # print(">>>",op.values)
                        # print(">>>",cls)
                        op_obj = cls(op.values)
                        out_inside = op_obj.call(out_inside)
                        out_template = op_obj.get_output_template(out_template)

                ##
                path_inside.append(out_inside)
                path_template.append(out_template)
                ##
            ##
            path_template_list = self._get_unique_list(path_template)

            if len(path_template_list) == 0:
                goal_inside[i] = {
                    "template": [],
                    "inside": torch.tensor(1),
                    "batch_flag": False,
                }
            else:
                if len(path_template_list) != 1:
                    print("[WARNING] missmatch indices:", path_template_list)
                if len(path_template_list[0]) == 0: # scalar inside
                    goal_inside[i] = {
                        "template": path_template_list[0],
                        "inside": path_inside[0],
                        "batch_flag": path_batch_flag,
                    }
                else:
                    goal_inside[i] = {
                        "template": path_template_list[0],
                        "inside": torch.sum(torch.stack(path_inside), dim=0),
                        "batch_flag": path_batch_flag,
                    }
        return goal_inside, self.loss


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
    def __init__(self, provider: 'TorchSwitchTensorProvider', name: str, shape: List[Union[int64, int]], dtype: dtype=torch.float32) -> None:
        self.shape = shape
        self.dtype = dtype
        if name is None:
            self.name = "tensor%04d" % (np.random.randint(0, 10000),)
        else:
            self.name = name
        self.provider = provider
        # (np.random.normal(0,1,self.shape), requires_grad=True,dtype=self.dtype)
        param = torch.nn.Parameter(torch.Tensor(*shape), requires_grad=True)
        self.param = param
        provider.add_param(self.name, param)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if len(self.param.shape) == 2:
            torch.nn.init.kaiming_uniform_(self.param, a=math.sqrt(5))
        else:
            self.param.data.uniform_(-0.1, 0.1)

    def __call__(self):
        return self.param


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
        if isinstance(self.var, TorchTensorBase):
            v = torch.index_select(self.var(), 0, idx)
        elif isinstance(self.var, PlaceholderData):
            v = self.provider.get_embedding(self.var)
            v = v[idx]
        else:
            v = torch.index_select(self.var, 0, idx)
        return v


class TorchSwitchTensorProvider(SwitchTensorProvider):
    def __init__(self) -> None:
        self.tensor_onehot_class = TorchTensorOnehot
        self.tensor_class = TorchTensor
        self.tensor_gather_class = TorchGather

        self.integer_dtype = torch.int32
        super().__init__()

    # forward
    def get_embedding(self, name, verbose=False):
        if verbose:
            print("[INFO] get embedding:", name)
        out = None
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

