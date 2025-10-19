""" This module contains base explanation graph classes that does not depend on pytorch

"""

from __future__ import annotations
import json
import re
import numpy as np

from itertools import chain
import collections

import os
import re
import pickle

from numpy import int32, int64, ndarray, str_

# this is only used for typing 
from torch import Tensor, dtype
from torch.nn.parameter import Parameter 
from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union

from tprism.placeholder import PlaceholderData
from tprism.torch_embedding_generator import BaseEmbeddingGenerator
from tprism.util import TensorInfoMapper
from tprism.loader import OperatorLoader
import tprism.expl_pb2 as expl_pb2
from tprism.loader import InputData
from tprism.expl_tensor import SwitchTensorProvider
from dataclasses import dataclass
from tprism.op.base import BaseOperator

@dataclass
class GoalTemplate:
    """
    Structured goal template.
      - template: index symbols
      - batch_flag: whether 'b' batch dim is present
      - shape: unified output shape per goal (dimension may contain None)
    Backward compatible with dict-style access via __getitem__.
    """
    template: List[str]
    batch_flag: bool
    shape: List[int]

    def __getitem__(self, key: str):
        if key == "template":
            return self.template
        if key == "batch_flag":
            return self.batch_flag
        if key == "shape":
            return self.shape
        raise KeyError(key)

class ComputationalExplGraph:
    """ This class is a base class for a concrete explanation graph.
    This module supports build_explanation_graph_template,
    where consistency of templates, index symbols and shape of tensors, in a given explanation graphs is confirmed.
    The build_explanation_graph_template also computes a list of goal templates (goal means LHS of path).

    Note:
        Goal template

        ::

            a goal template=
            {
                "template": List[str],
                "batch_flag": Boolean,
                "shape": List[int ...],
            }


    """

    def __init__(self):
        self.operators:Dict[str, BaseOperator]={}

    def _get_unique_list(self, seq: List[List[str]]) -> List[List[str]]:
        seen = []
        return [x for x in seq if x not in seen and not seen.append(x)]

    # [['i'], ['i','l', 'j'], ['j','k']] => ['l','k']
    def _compute_output_template(self, template: List[List[str]]) -> List[Union[str, Any]]:
        counter = collections.Counter(chain.from_iterable(template))
        out_template = [k for k, cnt in counter.items() if cnt == 1 and k != "b"]
        return sorted(out_template)

    # [['i'], ['i','l', 'j'], ['j','k']] => ['l','k']
    # [[3], [3, 4, 5], [5,6]] => [4,6]
    def _compute_output_shape(self, out_template: List[Union[str, Any]], sw_node_template: List[List[str]], sw_node_shape: List[Union[Tuple[int, int], List[int], List[Optional[int]], Tuple[int]]]) -> List[Union[int, Any]]:
        symbol_shape = {}
        for template_list, shape_list in zip(sw_node_template, sw_node_shape):
            for t, s in zip(template_list, shape_list):
                if t not in symbol_shape:
                    symbol_shape[t] = s
                elif symbol_shape[t] is None:
                    symbol_shape[t] = s
                else:
                    assert symbol_shape[t] == s, (
                        "index symbol mismatch:"
                        + str(t)
                        + ":"
                        + str(symbol_shape[t])
                        + "!="
                        + str(s)
                    )
        out_shape = []
        for symbol in out_template:
            if symbol in symbol_shape:
                out_shape.append(symbol_shape[symbol])
            else:
                out_shape.append(None)
        return out_shape

    def _unify_shapes(self, path_shapes: List[List[Union[int, Any]]]) -> List[Union[int, Any]]:
        """
        This method is used to unify shapes for all paths
        """
        n = len(path_shapes)
        if n == 0:
            return []
        else:
            m = len(path_shapes[0])
            out_shape = []
            for j in range(m):
                dim = None
                for i in range(n):
                    if path_shapes[i][j] is None:
                        pass
                    elif dim is None:
                        dim = path_shapes[i][j]
                    else:
                        assert path_shapes[i][j] == dim, "shape mismatching"
                out_shape.append(dim)
            return out_shape

    def _apply_operator_template(
        self,
        operator_loader: Optional[OperatorLoader],
        op: expl_pb2.SwIns,
        in_template: List[str],
        in_shape: List[int]
    ) -> Tuple[List[str], List[int]]:
        """
        Applies an operator template to the given input template and shape, restoring or creating the operator as needed.

        Args:
            operator_loader(OperatorLoader): An object responsible for loading operator classes by name.
            op: An operator instance containing 'name' and 'values' attributes.
            in_template: The input template to which the operator will be applied.
            in_shape: The input shape to which the operator will be applied.

        Returns:
            tuple: A tuple (out_template, out_shape) where:
                - out_template: The output template after applying the operator.
                - out_shape: The output shape after applying the operator.

        Side Effects:
            Updates the self.operators dictionary with the operator instance if it does not already exist.

        """
        key: str = str(op.name) + "_" + str(op.values)
        if key in self.operators:
            op_obj = self.operators[key]
        else:
            if operator_loader is None:
                raise ValueError("operator_loader must not be None when applying operator templates.")
            cls = operator_loader.get_operator(op.name)
            op_obj = cls(op.values)
            self.operators[key] = op_obj
        out_template = op_obj.get_output_template(in_template)
        out_shape = op_obj.get_output_shape(in_shape)
        return out_template, out_shape

    def build_explanation_graph_template(
        self,
        graph: expl_pb2.ExplGraph,
        tensor_provider: SwitchTensorProvider,
        operator_loader: Optional[OperatorLoader] = None,
        cycle_node: List[int] = [],
    ) -> Tuple[List[Optional[GoalTemplate]], List[int]]:
        """
        Args:
            graph(expl_pb2.ExplGraph): explanation graph object
            tensor_provider (SwitchTensorProvider): tensor provider
            operator_loader (OperatorLoader): operator loader
            cycle_node (List[int]): a list of sorted_id of cycle nodes

        Returns:
        A tuple containing goal_template and cycle_node
            - goal_template (List[GoalTemplate]): a list of goal templates
            - cycle_node (List[int]): a list of cycle node (if given cycle_node, it is updated)
        """
        # checking template
        goal_template: List[Optional[GoalTemplate]] = [None] * len(graph.goals)
        for i in range(len(graph.goals)):
            g = graph.goals[i]
            path_template = []
            path_shape = []
            path_batch_flag = False
            for path in g.paths:
                ## build template and inside for switches in the path
                sw_template = []
                sw_shape = []
                for sw in path.tensor_switches:
                    ph = tensor_provider.get_placeholder_name(sw.name)
                    sw_obj = tensor_provider.get_switch(sw.name)
                    if len(ph) > 0:
                        sw_template.append(["b"] + list(sw.values))
                        path_batch_flag = True
                        sw_shape.append([None] + list(sw_obj.get_shape()))
                    else:
                        sw_template.append(list(sw.values))
                        sw_shape.append(sw_obj.get_shape())
                ## building template and inside for nodes in the path
                node_template = []
                node_shape = []
                cycle_detected = False
                for node in path.nodes:
                    temp_goal = goal_template[node.sorted_id]
                    if temp_goal is None:
                        # cycle
                        if node.sorted_id not in cycle_node:
                            cycle_node.append(node.sorted_id)
                        cycle_detected = True
                        continue
                    if len(temp_goal.template) > 0:
                        if temp_goal.batch_flag:
                            path_batch_flag = True
                        node_shape.append(temp_goal.shape)
                        node_template.append(temp_goal.template)
                #if cycle_detected:
                #    continue
                sw_node_template = sw_template + node_template
                sw_node_shape = sw_shape + node_shape

                ##########
                ops = [op.name for op in path.operators]
                if "distribution" in ops:
                    # distributino clause
                    print("=== distribution ===")
                    print(sw_node_template)
                    print(sw_node_shape)
                    ##
                    out_template = sw_node_template[0]
                    out_shape = sw_node_shape[0]
                    print(out_template)
                    print(out_shape)
                else:
                    # constructing einsum operation using template and inside
                    out_template = self._compute_output_template(sw_node_template)
                    out_shape = self._compute_output_shape(
                        out_template, sw_node_template, sw_node_shape
                    )
                    if len(sw_node_template) > 0:  # condition for einsum
                        if path_batch_flag:
                            out_template = ["b"] + out_template
                    ## computing operaters
                    for op in path.operators:
                        out_template, out_shape=self._apply_operator_template(
                            operator_loader,
                            op,
                            out_template,
                            out_shape)
                ##########
                path_template.append(out_template)
                path_shape.append(out_shape)
                ##
            ##
            path_template_list = self._get_unique_list(path_template)
            path_shape = self._unify_shapes(path_shape)
            if len(path_template_list) == 0:
                goal_template[i] = GoalTemplate(
                    template=[],
                    batch_flag=False,
                    shape=path_shape,
                )
            else:
                if len(path_template_list) != 1:
                    print("[WARNING] missmatch indices:", path_template_list)
                goal_template[i] = GoalTemplate(
                    template=path_template_list[0],
                    batch_flag=path_batch_flag,
                    shape=path_shape,
                )
        ##
        return goal_template, cycle_node

