"""
This module contains the computational explanation graph.

A computational graph is constructed by forward traversing the given
explanation graph: consistency of templates (index symbols and shapes of
tensors) is verified first, and then each path is evaluated as an einsum
operation with optional operators.
"""

from __future__ import annotations
import collections
import logging

from itertools import chain
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import torch
import torch.nn.functional as F

import tprism.expl_pb2 as expl_pb2
from tprism.embedding_generator import BaseEmbeddingGenerator
from tprism.expl_tensor import SwitchTensorProvider
from tprism.loader import OperatorLoader
from tprism.op.base import BaseOperator
from tprism.tensor_index import (
    TensorIndexRef,
    parse_tensor_index,
    extract_tensor,
    extract_tensor_shape,
)
from tprism.util import debug_logger

logger = logging.getLogger(__name__)
graph_logger = debug_logger("graph")


@dataclass
class GoalTemplate:
    """
    Structured goal template.
      - template: index symbols
      - batch_flag: whether 'b' batch dim is present
      - shape: unified output shape per goal (dimension may contain None)
    Backward compatible with dict-style access via __getitem__.
    """
    template: List[TensorIndexRef]
    batch_flag: bool
    shape: Tuple[Optional[int], ...]


    def __getitem__(self, key: str):
        if key == "template":
            return self.template
        if key == "batch_flag":
            return self.batch_flag
        if key == "shape":
            return self.shape
        raise KeyError(key)


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
      - template: List[TensorIndexRef]
      - inside: Any (tensor in non-dryrun, list of path descriptions in dryrun)
      - batch_flag: bool
      - dryrun: bool
      - id, name, args: optional meta information (set in dryrun mode)
    """
    def __init__(self, template: List[TensorIndexRef], inside: Any, batch_flag: bool, dryrun: bool) -> None:
        self.template: List[TensorIndexRef] = template
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
        path_template_list: List[List[TensorIndexRef]],
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
            template: List[TensorIndexRef] = []
            inside_list = normalize_list(path_inside, dryrun)
            inside: Any = inside_list if dryrun else torch.tensor(1)
            # batch_flag must be False for this case (keep original behavior)
            return cls(template, inside, False, dryrun)

        if len(path_template_list) != 1:
            logger.warning("missmatch indices: %s", path_template_list)

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


class ComputationalExplGraph(torch.nn.Module):
    """ This class is a computational explanation graph for pytorch.

    The method build_explanation_graph_template confirms consistency of
    templates, index symbols and shape of tensors, in a given explanation
    graph and computes a list of goal templates (goal means LHS of path).
    The forward method converts the explanation graph into a computational
    graph.

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
    def __init__(
        self,
        graph: expl_pb2.ExplGraph,
        tensor_provider: SwitchTensorProvider,
        operator_loader: OperatorLoader,
        cycle_embedding_generator: Optional[BaseEmbeddingGenerator] = None,
    ) -> None:
        super().__init__()
        ## setting
        self.operators: Dict[str, BaseOperator] = {}
        self.operator_loader: Optional[OperatorLoader] = operator_loader
        self.goal_template: List[Optional[GoalTemplate]] = []
        self.cycle_node: List[int] = []
        self.param_op: Optional[torch.nn.ModuleList] = None
        self.graph: "expl_pb2.ExplGraph" = graph
        self.loss: Dict[str, torch.Tensor] = {}
        self.tensor_provider: SwitchTensorProvider = tensor_provider
        self.cycle_embedding_generator: Optional[BaseEmbeddingGenerator] = cycle_embedding_generator
        ###
        self.build()

    def _get_unique_symbol_list(self, seq: List[List[TensorIndexRef]]) -> List[List[TensorIndexRef]]:
        seen: List[Tuple[str, ...]] = []
        seen_: List[List[TensorIndexRef]] = []
        for ts in seq:
            symbols = tuple([t.symbol for t in ts])
            if symbols not in seen:
                seen.append(symbols)
                seen_.append(ts)
        return seen_

    # [['i'], ['i','l', 'j'], ['j','k']] => ['l','k']
    def _compute_output_template(self, template: List[List[TensorIndexRef]]) -> List[TensorIndexRef]:
        # skip index
        template_=[]
        for ts in template:
            l=[]
            for t in ts:
                if t.index_type!="index":  # Literal["symbol", "range", "index"]
                    l.append(t.symbol)
            template_.append(l)
        # counting
        counter = collections.Counter(chain.from_iterable(template_))
        out_template_ = [k for k, cnt in counter.items() if cnt == 1 and k != "b"]
        out_template = [TensorIndexRef("symbol", 0, -1, 1, e) for e in sorted(out_template_)]
        return out_template

    # [['i'], ['i','l', 'j'], ['j','k']] => ['l','k']
    # [[3], [3, 4, 5], [5,6]] => [4,6]
    def _compute_output_shape(self, out_template: List[TensorIndexRef], sw_node_template: List[List[TensorIndexRef]], sw_node_shape: List[Tuple[Optional[int], ...]]) -> Tuple[Optional[int],...]:
        symbol_shape = {}
        for template_list, shape_list in zip(sw_node_template, sw_node_shape):
            extracted_shape = extract_tensor_shape(shape_list, template_list)

            for t, s in zip(template_list, extracted_shape):
                if t.symbol not in symbol_shape:
                    symbol_shape[t.symbol] = s
                elif symbol_shape[t.symbol] is None:
                    symbol_shape[t.symbol] = s
                else:
                    assert symbol_shape[t.symbol] == s, (
                        "index symbol mismatch:"
                        + str(t)
                        + ":"
                        + str(symbol_shape[t.symbol])
                        + "!="
                        + str(s)
                    )
        out_shape = []
        for el in out_template:
            symbol = el.symbol
            if symbol in symbol_shape:
                out_shape.append(symbol_shape[symbol])
            else:
                out_shape.append(None)
        return tuple(out_shape)

    def _unify_shapes(self, path_shapes: List[Tuple[Optional[int],...]]) -> Tuple[Optional[int],...]:
        """
        This method is used to unify shapes for all paths
        """
        n = len(path_shapes)
        if n == 0:
            return ()
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
            return tuple(out_shape)

    def _apply_operator_template(
        self,
        operator_loader: Optional[OperatorLoader],
        op: expl_pb2.SwIns,
        in_template: List[TensorIndexRef],
        in_shape: Tuple[Optional[int], ...]
    ) -> Tuple[List[TensorIndexRef], Tuple[Optional[int], ...]]:
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
                sw_template: List[List[TensorIndexRef]] = []
                sw_shape: List[Tuple[Optional[int],...]] = []
                for sw in path.tensor_switches:
                    ph = tensor_provider.get_placeholder_name(sw.name)
                    sw_obj = tensor_provider.get_switch(sw.name)
                    sw_index_list=[parse_tensor_index(el) for el in sw.values]
                    if len(ph) > 0:
                        sw_template.append([TensorIndexRef("symbol", 0, -1, 1, "b")] + sw_index_list)
                        path_batch_flag = True
                        sw_shape.append(tuple([None] + list(sw_obj.get_shape())))
                    else:
                        sw_template.append(sw_index_list)
                        sw_shape.append(sw_obj.get_shape())
                ## building template and inside for nodes in the path
                node_template: List[List[TensorIndexRef]] = []
                node_shape: List[Tuple[Optional[int],...]] = []
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
                        node_shape.append(tuple(temp_goal.shape))
                        node_template.append(temp_goal.template)
                #if cycle_detected:
                #    continue
                sw_node_template = sw_template + node_template
                sw_node_shape = sw_shape + node_shape

                ##########
                ops = [op.name for op in path.operators]
                if "distribution" in ops:
                    # distributino clause
                    graph_logger.debug("=== distribution ===")
                    graph_logger.debug("%s", sw_node_template)
                    graph_logger.debug("%s", sw_node_shape)
                    ##
                    out_template = sw_node_template[0]
                    out_shape = sw_node_shape[0]
                    graph_logger.debug("%s", out_template)
                    graph_logger.debug("%s", out_shape)
                else:
                    # constructing einsum operation using template and inside
                    out_template = self._compute_output_template(sw_node_template)
                    out_shape = self._compute_output_shape(
                        out_template, sw_node_template, sw_node_shape
                    )
                    if len(sw_node_template) > 0:  # condition for einsum
                        if path_batch_flag:
                            out_template = [TensorIndexRef("symbol", 0, -1, 1, "b")] + out_template
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
            path_template_list = self._get_unique_symbol_list(path_template)
            path_shape_ = self._unify_shapes(path_shape)
            if len(path_template_list) == 0:
                goal_template[i] = GoalTemplate(
                    template=[],
                    batch_flag=False,
                    shape=path_shape_,
                )
            else:
                if len(path_template_list) != 1:
                    logger.warning("missmatch indices: %s", path_template_list)
                goal_template[i] = GoalTemplate(
                    template=path_template[0],
                    batch_flag=path_batch_flag,
                    shape=path_shape_,
                )
        ##
        return goal_template, cycle_node

    def build(self) -> None:

        goal_template, cycle_node = self.build_explanation_graph_template(
            self.graph, self.tensor_provider, self.operator_loader
        )
        self.goal_template = goal_template
        self.cycle_node = cycle_node

        ## setting parameterized operators
        self.param_op = torch.nn.ModuleList()
        for k, v in self.operators.items():
            if isinstance(v, torch.nn.Module):
                debug_logger("module").debug("Torch parameterized operator: %s", k)
                self.param_op.append(cast(torch.nn.Module, v))
        ##
        for name, (param, tensor_type) in self.tensor_provider.params.items():
            self.register_parameter(name, param)

    def _distribution_forward(
        self,
        name: str,
        dist: str,
        params: List[torch.Tensor],
        param_template: List[List[TensorIndexRef]],
        op: "expl_pb2.SwIns",
    ) -> Tuple[torch.Tensor, List[TensorIndexRef]]:
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
            logger.error("unknown distribution: %s", dist)
        return out_inside, out_template

    def make_einsum_args(
        self,
        template: List[List[TensorIndexRef]],
        out_template: List[TensorIndexRef],
        path_batch_flag: bool,
    ) -> Tuple[str, List[TensorIndexRef]]:
        """
        Example
        template: [["i","j"],["j","k"]]
        out_template: ["i","k"]
        => "ij,jk->ik", out_template
        """
        lhs = ",".join(map(lambda x: "".join([el.symbol for el in x]), template))
        ## skip index
        rhs_list=[]
        for el in out_template:
            if el.index_type!="index":  # Literal["symbol", "range", "index"]
                rhs_list.append(el.symbol)
        rhs = "".join(rhs_list)
        if path_batch_flag:
            rhs = "b" + rhs
            out_template = [TensorIndexRef("symbol", 0, -1, 1, "b")] + out_template
        einsum_eq = lhs + "->" + rhs
        return einsum_eq, out_template

    def make_einsum_args_sublist(
        self,
        template: List[List[TensorIndexRef]],
        inputs: List[torch.Tensor],
        out_template: List[TensorIndexRef],
        path_batch_flag: bool,
    ) -> Tuple[List[torch.Tensor| List[int]], List[TensorIndexRef]]:
        """
        Example
        template: [["i","j"],["j","k"]]
        out_template: ["i","k"]
        => [inputs[0], [0,1], inputs[1], [1,2], [0,2]], out_template
        """

        symbol_set = set([e.symbol for sublist in template for e in sublist])
        mapping={s:i for i,s in enumerate(symbol_set)}
        if path_batch_flag:
            out_template = [TensorIndexRef("symbol", 0, -1, 1, "b")] + out_template
        sublistform_args: List[torch.Tensor| List[int]] = []
        for v,input_x in zip(template,inputs):
            ## skip index
            l=[]
            for el in v:
                if el.index_type!="index":  # Literal["symbol", "range", "index"]
                    l.append(mapping[el.symbol])
            ##
            slice_x, index_tuple, out_symbols=extract_tensor(input_x, v)
            sublistform_args.append(slice_x)
            sublistform_args.append(l)
        sublistform_args.append([mapping[el.symbol] for el in out_template])
        return sublistform_args, out_template

    def _apply_operator(
        self,
        op: expl_pb2.SwIns,
        operator_loader: Optional["OperatorLoader"],
        out_node: ExplNode,
        out_template: List[TensorIndexRef],
        dryrun: bool,
    ) -> Tuple[ExplNode, List[TensorIndexRef]]:
        ## restore operator
        key=str(op.name)+"_"+str(op.values)
        op_obj:Optional[BaseOperator]=None
        if key in self.operators:
            op_obj=self.operators[key]
        else:
            assert True, "unknown operator "+key+" has been found in forward procedure"
        if op_obj is None:
            raise Exception("Unknown operator: ", key)
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

    def forward_path_node(
        self,
        path: expl_pb2.ExplGraphPath,
        goal_inside: List[Optional[GoalInsideEntry]],
        dryrun: bool = False,
    ) -> Tuple[List[List[TensorIndexRef]], List[ExplNode], List[ExplNode], bool]:
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
                gt=goal_template[node.sorted_id]
                if  gt is None:
                    raise Exception("goal_template[", node.sorted_id,"] is None in cycle node")
                template = gt["template"]
                shape = cast(Tuple[int], gt["shape"])
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
                    if cycle_embedding_generator is None:
                        raise Exception("cycle_embedding_generator is None in cycle node")
                    else:
                        v = cycle_embedding_generator.forward(name, shape, node.sorted_id)
                        temp_goal_inside = ExplNode.from_value("goal", v)
                temp_goal_template = template
                node_inside.append(temp_goal_inside)
                node_template.append(temp_goal_template)
            elif temp_goal is None:
                logger.error(
                    "cycle node is detected: sorted_id=%s node=%s temp_goal=%s",
                    node.sorted_id,
                    node,
                    goal_inside[node.sorted_id],
                )
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

    def forward_path_sw(
        self,
        path: expl_pb2.ExplGraphPath,
        dryrun: bool = False,
    ) -> Tuple[List[List[TensorIndexRef]], List[ExplNode], List[ExplNode], bool]:
        tensor_provider = self.tensor_provider
        sw_template: List[List[TensorIndexRef]] = []
        sw_inside: List[ExplNode] = []
        path_batch_flag=False
        for sw in path.tensor_switches:
            ph = tensor_provider.get_placeholder_name(sw.name)
            sw_index_list=[parse_tensor_index(el) for el in sw.values]
            if len(ph) > 0:
                sw_template.append([TensorIndexRef("symbol", 0, -1, 1, "b")] + sw_index_list)
                path_batch_flag = True
            else:
                sw_template.append(sw_index_list)
            if dryrun:
                sw_var = ExplNode.from_desc("tensor_atom", {
                    "type":"tensor_atom",
                    "from":"tensor_provider.get_embedding",
                    "name":sw.name,})
            else:
                v = tensor_provider.get_embedding(sw.name)
                sw_var = ExplNode.from_value("tensor_atom", v)
            sw_inside.append(sw_var)
        prob_sw_inside: List[ExplNode] = []
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

    def forward_path_op(
        self,
        path_name: str,
        ops: Dict[str, "expl_pb2.SwIns"],
        operator_loader: Optional[OperatorLoader],
        sw_node_template: List[List[TensorIndexRef]],
        sw_node_inside: List[ExplNode],
        node_scalar_inside: List[ExplNode],
        prob_sw_inside: List[ExplNode],
        path_batch_flag: bool,
        dryrun: bool = False,
    ) -> Tuple[ExplNode, List[TensorIndexRef]]:

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
                out_template= sw_node_template[0]  # TODO: refine if needed
            else:
                params = [x.as_value() if isinstance(x, ExplNode) else x for x in sw_node_inside]
                out_val, out_template_temp = self._distribution_forward(
                    path_name,
                    dist,
                    params=params,
                    param_template=sw_node_template,
                    op=op,
                )
                out_template = out_template_temp
                out_node = ExplNode.from_value("distribution", out_val)

        else:  # einsum operator
            path_v = sorted(zip(sw_node_template, sw_node_inside), key=lambda x: x[0])
            template = [x[0] for x in path_v]
            inside_nodes = [x[1] for x in path_v]
            out_template = self._compute_output_template(template)  # out_template is List[str]
            if len(template) > 0:
                if dryrun:
                    #print(">in>",template)
                    #print(">out>",out_template)
                    einsum_eq, out_template_einsum = self.make_einsum_args(
                        template, out_template, path_batch_flag)
                    desc = {
                        "type":"einsum",
                        "name":"torch.einsum",
                        "einsum_eq":einsum_eq,
                        "path":[n.as_desc() if isinstance(n, ExplNode) else n for n in inside_nodes],
                    }
                    out_node = ExplNode.from_desc("einsum", desc)
                    out_template = out_template_einsum
                else:
                    inside_vals = [n.as_value() if isinstance(n, ExplNode) else n for n in inside_nodes]
                    einsum_args, out_template_temp = self.make_einsum_args_sublist(template, inside_vals, out_template, path_batch_flag)
                    try:
                        out_val = torch.einsum(*einsum_args)
                    except Exception as e:
                        logger.error("einsum failed for path: %s", path_name)
                        logger.error("in  > %s", template)
                        logger.error("out_org  > %s", out_template)
                        logger.error("out_list > %s", out_template_temp)
                        logger.error("inside_vals: %s", inside_vals)
                        logger.error("einsum_args: %s", einsum_args)
                        raise e
                    out_node = ExplNode.from_value("einsum", out_val)
                    out_template = out_template_temp
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
                graph_logger.debug("  operator: %s", op_name)
                out_node, out_template = self._apply_operator(
                    op,
                    operator_loader,
                    out_node,
                    out_template,
                    dryrun)
                ##
        return out_node, out_template

    def forward(
        self,
        verbose: bool = False,
        verbose_embedding: bool = False,
        dryrun: bool = False,
    ) -> Tuple[List[Optional[GoalInsideEntry]], Dict[str, torch.Tensor]]:
        """
         Args:
             verbose (bool): deprecated and ignored; verbosity is controlled by the logging level (see `tprism.util.setup_logging`)
             verbose_embedding (bool): deprecated and ignored; same as verbose
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
        goal_inside: List[Optional[GoalInsideEntry]] = [None] * len(graph.goals)
        for i in range(len(graph.goals)):
            g = graph.goals[i]
            graph_logger.debug(
                "=== tensor equation (node_id:%d, %s) ===",
                g.node.sorted_id,
                g.node.goal.name,
            )
            path_inside = []
            path_template: List[List[TensorIndexRef]] = []
            path_batch_flag = False
            for j, path in enumerate(g.paths):
                ## build template and inside for switches in the path
                sw_template, sw_inside, prob_sw_inside, batch_flag = self.forward_path_sw(
                        path, dryrun)
                path_batch_flag = batch_flag or path_batch_flag

                ## building template and inside for nodes in the path
                node_template, node_inside, node_scalar_inside, batch_flag = self.forward_path_node(
                        path, goal_inside, dryrun)
                path_batch_flag = batch_flag or path_batch_flag

                ## building template and inside for all elements (switches and nodes) in the path
                sw_node_template = sw_template + node_template
                sw_node_inside = sw_inside + node_inside

                ops = {op.name: op for op in path.operators}

                path_name = g.node.goal.name+str(j)
                out_node,out_template = self.forward_path_op(path_name, ops, operator_loader,
                        sw_node_template, sw_node_inside, node_scalar_inside, prob_sw_inside,
                        path_batch_flag,
                        dryrun)

                path_inside.append(out_node)
                path_template.append(out_template)
                ##
            ### update inside
            path_template_list = self._get_unique_symbol_list(path_template)
            entry = GoalInsideEntry.merge_paths(path_template_list, path_inside, path_batch_flag, dryrun)
            if dryrun:
                entry.set_meta(g.node.sorted_id, g.node.goal.name, g.node.goal.args)
            goal_inside[i] = entry

        self.loss.update(tensor_provider.get_loss())
        return goal_inside, self.loss
