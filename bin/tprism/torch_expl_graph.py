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
from tprism.expl_graph import PlaceholderGraph, VocabSet
from tprism.loader import OperatorLoader
from tprism.placeholder import PlaceholderData
from numpy import int64
from torch import dtype
from typing import Any, Dict, List, Tuple, Union


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
    A goal in the explanation graph is represented as the following format:
        ```
         goal_inside[sorted_id] = {
            "template": path_template,      template: List[str]
            "inside": path_inside,          torch.tensor
            "batch_flag": path_batch_flag,  bool
         }
        ```
    
    Regarding the path template of goals, the T-PRISM's assumption requires that all pathes have equal tensor size.
    However, index symbols need not be equal for all paths.
    So the system display a warning in case of different index symbols and only uses the index symbols in the first path.


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

    def _apply_operator(self,op,operator_loader,out_inside, out_template, dryrun):
        ## restore operator
        key=str(op.name)+"_"+str(op.values)
        if key in self.operators:
            op_obj=self.operators[key]
        else:
            #cls = operator_loader.get_operator(op.name)
            #op_obj = cls(op.values)
            assert True, "unknown operator "+key+" has been found in forward procedure"
        if dryrun:
            out_inside = {
                "type": "operator",
                "name": op.name,
                "path": out_inside}
        else:
            out_inside = op_obj.call(out_inside)
        out_template = op_obj.get_output_template(out_template)
        return out_inside,out_template
    
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
                    temp_goal_inside={
                        "type":"goal",
                        "from":"cycle_embedding_generator",
                        "name": name,
                        "args": args,
                        "id": node.sorted_id,
                        "shape":shape}
                else:
                    temp_goal_inside = cycle_embedding_generator.forward(
                        name, shape, node.sorted_id
                    )
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
            elif len(temp_goal["template"]) > 0:
                # tensor subgoal
                if dryrun:
                    name = node.goal.name
                    args = node.goal.args
                    temp_goal_inside={
                        "type":"goal",
                        "from":"goal",
                        "name": name,
                        "args": args,
                        "id": node.sorted_id,}
                        #"shape":shape}
                else:
                    temp_goal_inside = temp_goal["inside"]
                temp_goal_template = temp_goal["template"]
                if temp_goal["batch_flag"]:
                    path_batch_flag = True
                node_inside.append(temp_goal_inside)
                node_template.append(temp_goal_template)
            else:  # scalar subgoal
                if dryrun:
                    name = node.goal.name
                    args = node.goal.args
                    temp_goal_inside={
                        "type":"goal",
                        "from":"goal",
                        "name": name,
                        "args": args,
                        "id": node.sorted_id,}
                        #"shape":()}
                    node_scalar_inside.append(temp_goal_inside)
                else:
                    if type(temp_goal["inside"]) is list:
                        a = torch.tensor(temp_goal["inside"])
                        node_scalar_inside.append(torch.squeeze(a))
                    else:
                        node_scalar_inside.append(temp_goal["inside"])
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
                #x = tensor_provider.get_embedding(sw.name, verbose_embedding)
                sw_var = {"type":"tensor_atom",
                        "from":"tensor_provider.get_embedding",
                        "name":sw.name,}
                        #"shape":x.shape}
            else:
                sw_var = tensor_provider.get_embedding(sw.name, verbose_embedding)
            sw_inside.append(sw_var)
        prob_sw_inside = []
        if dryrun:
            for sw in path.prob_switches:
                prob_sw_inside.append({
                    "type":"const",
                    "name": sw.name,
                    "value": sw.inside,
                    "shape":(),})
        else:
            for sw in path.prob_switches:
                prob_sw_inside.append(sw.inside)
                """
                prob_sw_inside.append({
                    "type":"const",
                    "name": sw.name,
                    "value": sw.inside,
                    "shape":(),})
                """
        return sw_template, sw_inside, prob_sw_inside, path_batch_flag
    
    def forward_path_op(self, path_name, ops, operator_loader,
             sw_node_template, sw_node_inside, node_scalar_inside, prob_sw_inside,path_batch_flag, verbose=False, dryrun=False):
      
        if "distribution" in ops:
            op = ops["distribution"]
            dist = op.values[0]
            if dryrun:
                #out_inside, out_template = self._distribution_forward_dryrun
                out_inside={
                    "type": "distribution",
                    "name": path_name,
                    "dist_type": op,
                    "path": sw_node_inside}
                out_template= sw_node_template#TODO
            else:
                out_inside, out_template = self._distribution_forward(
                    path_name,
                    dist,
                    params=sw_node_inside,
                    param_template=sw_node_template,
                    op=op,
                )
            
        else:# einsum operator
            path_v = sorted(
                zip(sw_node_template, sw_node_inside), key=lambda x: x[0]
            )
            template = [x[0] for x in path_v]
            inside = [x[1] for x in path_v]
            # constructing einsum operation using template and inside
            out_template = self._compute_output_template(template)
            # print(template,out_template)
            if len(template) > 0:  # condition for einsum
                if verbose:
                    einsum_eq, out_template_v = self.make_einsum_args(template,out_template,path_batch_flag)
                    print("  index:", einsum_eq)
                    print("  var. :", [x.shape for x in inside])
                    #print("  var. :", inside)
                if dryrun:
                    einsum_eq, out_template = self.make_einsum_args(template,out_template,path_batch_flag)
                    out_inside = {
                        "type":"einsum",
                        "name":"torch.einsum",
                        "einsum_eq":einsum_eq,
                        "path": inside}
                else:
                    einsum_args, out_template = self.make_einsum_args_sublist(template,inside,out_template,path_batch_flag)
                    #out_inside = torch.einsum(einsum_eq, *inside) * out_inside
                    out_inside = torch.einsum(*einsum_args)
            else:  # condition for scalar
                if dryrun:
                    out_inside = {
                        "type":"nop",
                        "name":"nop",
                        "path": inside}
                else:
                    out_inside=1 #no subgoal except for operator
            if dryrun:
                out_inside["path_scalar"]=node_scalar_inside+prob_sw_inside
            else:
                # scalar subgoal
                for scalar_inside in node_scalar_inside:
                    out_inside = scalar_inside * out_inside
                # prob(scalar) switch
                for prob_inside in prob_sw_inside:
                    out_inside = prob_inside * out_inside
                
            ## computing operaters
            for op_name, op in ops.items():
                if verbose:
                    print("  operator:", op_name)
                out_inside,out_template=self._apply_operator(
                    op,
                    operator_loader,
                    out_inside,
                    out_template,
                    dryrun)
                ##
        return out_inside,out_template

    def forward(self, verbose=False,verbose_embedding=False, dryrun=False):
        """
        Args:
            verbose (bool): if true, this function displays an explanation graph with forward computation
            dryrun (bool):  if true, this function outputs information required for calculation as goal_inside instead of computational graph

        Returns:
            Tuple[List[Dict],Dict]: a pair of goal_inside and loss:
                - goal_inside: tensors assigned for each goal
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
                out_inside,out_template = self.forward_path_op(path_name, ops, operator_loader,
                        sw_node_template, sw_node_inside, node_scalar_inside, prob_sw_inside,
                        path_batch_flag,
                        verbose, dryrun)
                
                path_inside.append(out_inside)
                path_template.append(out_template)
                ##
            ### update inside
            path_template_list = self._get_unique_list(path_template)
            if len(path_template_list) == 0: # non-tensor/non-probabilistic path
                if dryrun:
                    goal_inside[i] = {
                        "template": [],
                        "inside": path_inside,
                        "batch_flag": False,
                    }
                else:
                    goal_inside[i] = {
                        "template": [],
                        "inside": torch.tensor(1),
                        "batch_flag": False,
                    }
            else:
                if len(path_template_list) != 1:
                    print("[WARNING] missmatch indices:", path_template_list)
                if dryrun:
                    goal_inside[i] = {
                        "template": path_template_list[0],
                        "inside": path_inside,
                        "batch_flag": path_batch_flag,
                    }
                else:
                    if len(path_template_list[0]) == 0: # scalar inside
                        goal_inside[i] = {
                            "template": path_template_list[0],
                            "inside": path_inside[0],
                            "batch_flag": path_batch_flag,
                        }
                    else:
                        temp_inside=torch.sum(torch.stack(path_inside), dim=0)
                        goal_inside[i] = {
                            "template": path_template_list[0],
                            "inside": temp_inside,
                            "batch_flag": path_batch_flag,
                        }
            if dryrun:
                goal_inside[i]["id"]=g.node.sorted_id
                goal_inside[i]["name"]=g.node.goal.name
                goal_inside[i]["args"]=g.node.goal.args

        self.loss.update(tensor_provider.get_loss())
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
    def __init__(self, provider: 'TorchSwitchTensorProvider', name: str, shape: List[Union[int64, int]], dtype: dtype=torch.float32, tensor_type=None) -> None:
        self.shape = shape
        self.dtype = dtype
        if name is None:
            self.name = "tensor%04d" % (np.random.randint(0, 10000),)
        else:
            self.name = name
        self.provider = provider
        self.tensor_type = tensor_type
        ###
        self.constraint_tensor=tprism.constraint.get_constraint_tensor(shape, tensor_type, device=None, dtype=None)
        self.param = None
        if self.constraint_tensor is None:
            param = torch.nn.Parameter(torch.Tensor(*shape), requires_grad=True)
            self.param = param
            provider.add_param(self.name, param, tensor_type)
            self.reset_parameters()
        else:
            param=list(self.constraint_tensor.parameters())[0] #TODO
            provider.add_param(self.name, param, tensor_type)

        ###
    def reset_parameters(self) -> None:
        if len(self.param.shape) == 2:
            torch.nn.init.kaiming_uniform_(self.param, a=math.sqrt(5))
        else:
            self.param.data.uniform_(-0.1, 0.1)

    def __call__(self):
        if self.constraint_tensor is None:
            return self.param
        else:
            return self.constraint_tensor()


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
        if isinstance(self.var, TorchTensor):
            temp = self.var()
            v = torch.index_select(temp, 0, idx)
        elif isinstance(self.var, TorchTensorBase):
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
    def get_loss(self, verbose:bool=False):
        loss={}
        for name, (param,tensor_type) in self.params.items():
            m=re.match(r"^sparse\(([0-9\.]*)\)$", tensor_type)
            if m:
                coeff = float(m.group(1))
                l=coeff*torch.norm(param,1)
                loss["sparse_"+name]=l
            elif tensor_type=="sparse":
                l=torch.norm(param,1)
                loss["sparse_"+name]=l
        return loss
    # forward
    def get_embedding(self, name: Union[str,PlaceholderData], verbose:bool=False):
        if verbose:
            print("[INFO] get embedding:", name)
        out = None
        ## TODO:
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

