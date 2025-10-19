"""
This module contains pytorch explanation graphs and pytorch tensors.

This module constructs a computational graph by forward traversing the given explanation graph.

"""


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
import tprism.constraint

from tprism.expl_graph import ComputationalExplGraph, SwitchTensorProvider
from tprism.expl_tensor import PlaceholderGraph, VocabSet
from tprism.loader import OperatorLoader
from tprism.placeholder import PlaceholderData
from numpy import int64
from torch import dtype
from typing import Any, Dict, List, Tuple, Union


class ExplNode:
    """
    A container for intermediate values and dryrun descriptors.
    - If dryrun=False: holds a concrete value (tensor/scalar) in `value`.
    - If dryrun=True: holds a descriptor dict in `desc`.
    """
    def __init__(self, kind: str, dryrun: bool, value: Any = None, desc: Dict[str, Any] = {}) -> None:
        self.kind = kind
        self.dryrun = dryrun
        self.value = value
        self.desc = {} if desc is None else desc

    @staticmethod
    def from_value(kind: str, value: Any) -> "ExplNode":
        return ExplNode(kind=kind, dryrun=False, value=value)

    @staticmethod
    def from_desc(kind: str, desc: Dict[str, Any]) -> "ExplNode":
        return ExplNode(kind=kind, dryrun=True, desc=desc)

    def as_value(self) -> Any:
        if self.dryrun:
            raise RuntimeError("Attempted to access concrete value from a dryrun ExplNode")
        return self.value

    def as_desc(self) -> Dict[str, Any]:
        if self.dryrun:
            return self.desc
        # Optional lightweight descriptor when not in dryrun (used rarely)
        return {"type": self.kind}

    def add_path_scalar_descriptors(self, scalars: List["ExplNode"]) -> None:
        """
        Attach scalar path descriptors (for dryrun pretty output).
        """
        if self.dryrun:
            lst = []
            for s in scalars:
                if isinstance(s, ExplNode):
                    lst.append(s.as_desc())
                else:
                    lst.append(s)
            self.desc["path_scalar"] = lst


class GoalInsideEntry:
    """
    A unified container for goal_inside entries to replace the previous dict-based structure.

    Fields:
      - template: List[str]
      - inside: Any (tensor in non-dryrun, list of path descriptions in dryrun)
      - batch_flag: bool
      - dryrun: bool
      - id, name, args: optional meta information (set in dryrun mode)
    """
    def __init__(self, template: List[str], inside: Any, batch_flag: bool, dryrun: bool) -> None:
        self.template: List[str] = template
        self.inside: Any = inside
        self.batch_flag: bool = batch_flag
        self.dryrun: bool = dryrun
        self.id: Union[int, None] = None
        self.name: Union[str, None] = None
        self.args: Any = None

    def set_meta(self, goal_id: int, name: str, args: Any) -> "GoalInsideEntry":
        self.id = goal_id
        self.name = name
        self.args = args
        return self

    def is_tensor_goal(self) -> bool:
        return len(self.template) > 0

    def is_scalar_goal(self) -> bool:
        return len(self.template) == 0

    def get_scalar_inside(self) -> Any:
        """
        Return a scalar usable in computations (non-dryrun only).
        If inside is a list, convert to tensor and squeeze.
        """
        if self.dryrun:
            # In dryrun, scalar usage is handled by caller constructing descriptors.
            return self.inside
        v = self.inside
        if isinstance(v, list):
            a = torch.tensor(v)
            return torch.squeeze(a)
        return v

    @classmethod
    def merge_paths(
        cls,
        path_template_list: List[List[str]],
        path_inside: List["ExplNode"],
        path_batch_flag: bool,
        dryrun: bool,
    ) -> "GoalInsideEntry":
        """
        Merge paths into a single GoalInsideEntry adhering to the previous behavior.
        Supports lists of ExplNode.
        """
        def normalize_list(items: List["ExplNode"], is_dry: bool) -> List[Any]:
            return [it.as_desc() if is_dry else it.as_value() for it in items]

        if len(path_template_list) == 0:
            template: List[str] = []
            inside_list = normalize_list(path_inside, dryrun)
            inside: Any = inside_list if dryrun else torch.tensor(1)
            # batch_flag must be False for this case (keep original behavior)
            return cls(template, inside, False, dryrun)

        if len(path_template_list) != 1:
            print("[WARNING] missmatch indices:", path_template_list)

        template = path_template_list[0]
        if dryrun:
            inside = normalize_list(path_inside, True)
        else:
            items = normalize_list(path_inside, False)
            if len(template) == 0:
                inside = items[0]
            else:
                inside = torch.sum(torch.stack(items), dim=0)
        return cls(template, inside, path_batch_flag, dryrun)


class TorchComputationalExplGraph(ComputationalExplGraph, torch.nn.Module):
    """ This class is a concrete explanation graph for pytorch

Note:
    A path in the explanation graph is represented as the following format:
        ```
        {
            "sw_template": [],         a list of template: List[str]
            "sw_inside": [],           a list of torch.tensor
            "prob_sw_inside": [],      a list of torch.tensor (scalar) 
            "node_template": [],       a list of template: List[str]
            "node_inside": [],         a list of torch.tensor
            "node_scalar_inside": [],  a list of torch.tensor (scalar)
        }
        ```
Note:
    Each goal in the explanation graph is represented by GoalInsideEntry instead of a dict.
    """
    def __init__(self, graph, tensor_provider, operator_loader, cycle_embedding_generator=None):
        torch.nn.Module.__init__(self)
        ComputationalExplGraph.__init__(self)
        ## setting
        self.operator_loader = None
        self.goal_template = None
        self.cycle_node = None
        self.param_op = None
        self.graph = graph
        self.loss = {}
        self.tensor_provider = tensor_provider
        self.cycle_embedding_generator = cycle_embedding_generator
        """
        if operator_loader is None:
            operator_loader = OperatorLoader()
            operator_loader.load_all("op/torch_")
        """
        self.operator_loader = operator_loader
        ###
        self.build()
        
    def build(self):

        ## call super class        
        goal_template, cycle_node = self.build_explanation_graph_template(
            self.graph, self.tensor_provider, self.operator_loader
        )
        self.goal_template = goal_template
        self.cycle_node = cycle_node
        
        ## setting parameterized operators
        self.param_op = torch.nn.ModuleList()
        for k,v in self.operators.items():
            if issubclass(v.__class__, torch.nn.Module):
                print("Torch parameterized operator:", k)
                self.param_op.append(v)
        ##
        for name, (param,tensor_type) in self.tensor_provider.params.items():
            self.register_parameter(name, param)
        
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

    def make_einsum_args(self, template,out_template,path_batch_flag):
        """
        Example
        template: [["i","j"],["j","k"]]
        out_template: ["i","k"]
        => "ij,jk->ik", out_template
        """
        lhs = ",".join(map(lambda x: "".join(x), template))
        rhs = "".join(out_template)
        if path_batch_flag:
            rhs = "b" + rhs
            out_template = ["b"] + out_template
        einsum_eq = lhs + "->" + rhs
        return einsum_eq, out_template
    
    def make_einsum_args_sublist(self,template,inputs,out_template,path_batch_flag):
        """
        Example
        template: [["i","j"],["j","k"]]
        out_template: ["i","k"]
        => [inputs[0], [0,1], inputs[1], [1,2], [0,2]], out_template
        """
        symbol_set = set([e for sublist in template for e in sublist])
        mapping={s:i for i,s in enumerate(symbol_set)}
        if path_batch_flag:
            out_template = ["b"] + out_template
        sublistform_args=[]
        for v,input_x in zip(template,inputs):
            l=[mapping[el] for el in v]
            sublistform_args.append(input_x)
            sublistform_args.append(l)
        sublistform_args.append([mapping[el] for el in out_template])
        return sublistform_args, out_template

    def _apply_operator(self, op, operator_loader, out_node: ExplNode, out_template, dryrun):
        ## restore operator
        key=str(op.name)+"_"+str(op.values)
        op_obj=None
        if key in self.operators:
            op_obj=self.operators[key]
        else:
            assert True, "unknown operator "+key+" has been found in forward procedure"
        if dryrun:
            desc = {
                "type": "operator",
                "name": op.name,
                "path": out_node.as_desc(),
            }
            out_node = ExplNode.from_desc("operator", desc)
        else:
            out_val = op_obj.call(out_node.as_value())
            out_node = ExplNode.from_value("operator", out_val)
        out_template = op_obj.get_output_template(out_template)
        return out_node, out_template
    
    def forward_path_node(self, path, goal_inside, verbose=False, dryrun=False):
        goal_template = self.goal_template
        cycle_embedding_generator = self.cycle_embedding_generator
        cycle_node=self.cycle_node
        node_template = []
        node_inside = []
        node_scalar_inside = []
        path_batch_flag=False
        for node in path.nodes:
            temp_goal = goal_inside[node.sorted_id]
            if node.sorted_id in cycle_node:
                name = node.goal.name
                template = goal_template[node.sorted_id]["template"]
                shape = goal_template[node.sorted_id]["shape"]
                if dryrun:
                    args = node.goal.args
                    temp_goal_inside = ExplNode.from_desc("goal", {
                        "type":"goal",
                        "from":"cycle_embedding_generator",
                        "name": name,
                        "args": args,
                        "id": node.sorted_id,
                        "shape":shape})
                else:
                    v = cycle_embedding_generator.forward(name, shape, node.sorted_id)
                    temp_goal_inside = ExplNode.from_value("goal", v)
                temp_goal_template = template
                node_inside.append(temp_goal_inside)
                node_template.append(temp_goal_template)
            elif temp_goal is None:
                print("  [ERROR] cycle node is detected")
                temp_goal = goal_inside[node.sorted_id]
                print(node.sorted_id)
                print(node)
                print(node.sorted_id)
                print(temp_goal)
                quit()
            elif len(temp_goal.template) > 0:
                # tensor subgoal
                if dryrun:
                    name = node.goal.name
                    args = node.goal.args
                    temp_goal_inside = ExplNode.from_desc("goal", {
                        "type":"goal",
                        "from":"goal",
                        "name": name,
                        "args": args,
                        "id": node.sorted_id,})
                else:
                    temp_goal_inside = ExplNode.from_value("goal", temp_goal.inside)
                temp_goal_template = temp_goal.template
                if temp_goal.batch_flag:
                    path_batch_flag = True
                node_inside.append(temp_goal_inside)
                node_template.append(temp_goal_template)
            else:  # scalar subgoal
                if dryrun:
                    name = node.goal.name
                    args = node.goal.args
                    temp_goal_inside = ExplNode.from_desc("goal", {
                        "type":"goal",
                        "from":"goal",
                        "name": name,
                        "args": args,
                        "id": node.sorted_id,})
                    node_scalar_inside.append(temp_goal_inside)
                else:
                    node_scalar_inside.append(ExplNode.from_value("goal_scalar", temp_goal.get_scalar_inside()))
        return node_template, node_inside, node_scalar_inside, path_batch_flag

    def forward_path_sw(self, path, verbose=False,verbose_embedding=False, dryrun=False):
        tensor_provider = self.tensor_provider
        sw_template = []
        sw_inside = []
        path_batch_flag=False
        for sw in path.tensor_switches:
            ph = tensor_provider.get_placeholder_name(sw.name)
            if len(ph) > 0:
                sw_template.append(["b"] + list(sw.values))
                path_batch_flag = True
            else:
                sw_template.append(list(sw.values))
            if dryrun:
                sw_var = ExplNode.from_desc("tensor_atom", {
                    "type":"tensor_atom",
                    "from":"tensor_provider.get_embedding",
                    "name":sw.name,})
            else:
                v = tensor_provider.get_embedding(sw.name, verbose_embedding)
                sw_var = ExplNode.from_value("tensor_atom", v)
            sw_inside.append(sw_var)
        prob_sw_inside = []
        if dryrun:
            for sw in path.prob_switches:
                prob_sw_inside.append(ExplNode.from_desc("const", {
                    "type":"const",
                    "name": sw.name,
                    "value": sw.inside,
                    "shape":(),}))
        else:
            for sw in path.prob_switches:
                prob_sw_inside.append(ExplNode.from_value("const", sw.inside))
        return sw_template, sw_inside, prob_sw_inside, path_batch_flag
    
    def forward_path_op(self, path_name, ops, operator_loader,
             sw_node_template, sw_node_inside, node_scalar_inside, prob_sw_inside,path_batch_flag, verbose=False, dryrun=False):
      
        if "distribution" in ops:
            op = ops["distribution"]
            dist = op.values[0]
            if dryrun:
                desc = {
                    "type": "distribution",
                    "name": path_name,
                    "dist_type": op,
                    "path": [x.as_desc() if isinstance(x, ExplNode) else x for x in sw_node_inside],
                }
                out_node = ExplNode.from_desc("distribution", desc)
                out_template= sw_node_template  # TODO: refine if needed
            else:
                params = [x.as_value() if isinstance(x, ExplNode) else x for x in sw_node_inside]
                out_val, out_template = self._distribution_forward(
                    path_name,
                    dist,
                    params=params,
                    param_template=sw_node_template,
                    op=op,
                )
                out_node = ExplNode.from_value("distribution", out_val)
            
        else:  # einsum operator
            path_v = sorted(zip(sw_node_template, sw_node_inside), key=lambda x: x[0])
            template = [x[0] for x in path_v]
            inside_nodes = [x[1] for x in path_v]
            out_template = self._compute_output_template(template)
            if len(template) > 0:
                if dryrun:
                    einsum_eq, out_template = self.make_einsum_args(template,out_template,path_batch_flag)
                    desc = {
                        "type":"einsum",
                        "name":"torch.einsum",
                        "einsum_eq":einsum_eq,
                        "path":[n.as_desc() if isinstance(n, ExplNode) else n for n in inside_nodes],
                    }
                    out_node = ExplNode.from_desc("einsum", desc)
                else:
                    inside_vals = [n.as_value() if isinstance(n, ExplNode) else n for n in inside_nodes]
                    einsum_args, out_template = self.make_einsum_args_sublist(template, inside_vals, out_template, path_batch_flag)
                    out_val = torch.einsum(*einsum_args)
                    out_node = ExplNode.from_value("einsum", out_val)
            else:
                if dryrun:
                    out_node = ExplNode.from_desc("nop", {"type":"nop","name":"nop","path":[n.as_desc() if isinstance(n, ExplNode) else n for n in inside_nodes]})
                else:
                    out_node = ExplNode.from_value("nop", 1)
            if dryrun:
                out_node.add_path_scalar_descriptors(node_scalar_inside + prob_sw_inside)
            else:
                # multiply scalar subgoals and prob switches
                v = out_node.as_value()
                for s in node_scalar_inside:
                    v = s.as_value() * v
                for p in prob_sw_inside:
                    v = p.as_value() * v
                out_node = ExplNode.from_value(out_node.kind, v)
                
            ## computing operaters
            for op_name, op in ops.items():
                if op_name == "distribution":
                    continue
                if verbose:
                    print("  operator:", op_name)
                out_node, out_template = self._apply_operator(
                    op,
                    operator_loader,
                    out_node,
                    out_template,
                    dryrun)
                ##
        return out_node, out_template

    def forward(self, verbose=False,verbose_embedding=False, dryrun=False):
        """
        Args:
            verbose (bool): if true, this function displays an explanation graph with forward computation
            dryrun (bool):  if true, this function outputs information required for calculation as goal_inside instead of computational graph

        Returns:
            Tuple[List[GoalInsideEntry],Dict]: a pair of goal_inside and loss:
                - goal_inside: entries assigned for each goal
                - loss: loss derived from explanation graph: key = loss name and value = loss
        """
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
            for j, path in enumerate(g.paths):
                ## build template and inside for switches in the path
                sw_template, sw_inside, prob_sw_inside, batch_flag = self.forward_path_sw(
                        path, verbose, verbose_embedding, dryrun)
                path_batch_flag = batch_flag or path_batch_flag
                
                ## building template and inside for nodes in the path
                node_template, node_inside, node_scalar_inside, batch_flag = self.forward_path_node(
                        path, goal_inside, verbose, dryrun)
                path_batch_flag = batch_flag or path_batch_flag

                ## building template and inside for all elements (switches and nodes) in the path
                sw_node_template = sw_template + node_template
                sw_node_inside = sw_inside + node_inside

                ops = {op.name: op for op in path.operators}
                
                path_name = g.node.goal.name+str(j)
                out_node,out_template = self.forward_path_op(path_name, ops, operator_loader,
                        sw_node_template, sw_node_inside, node_scalar_inside, prob_sw_inside,
                        path_batch_flag,
                        verbose, dryrun)
                
                path_inside.append(out_node)
                path_template.append(out_template)
                ##
            ### update inside
            path_template_list = self._get_unique_list(path_template)
            entry = GoalInsideEntry.merge_paths(path_template_list, path_inside, path_batch_flag, dryrun)
            if dryrun:
                entry.set_meta(g.node.sorted_id, g.node.goal.name, g.node.goal.args)
            goal_inside[i] = entry

        self.loss.update(tensor_provider.get_loss())
        return goal_inside, self.loss

   