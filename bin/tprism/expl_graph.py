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
from torch import Tensor, dtype
from torch.nn.parameter import Parameter # this is only used for typing 
from tprism.op.torch_standard_op import Sigmoid
from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union

from tprism.placeholder import PlaceholderData
from tprism.torch_embedding_generator import BaseEmbeddingGenerator
from tprism.util import TensorInfoMapper

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
        self.operators={}

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

    def _apply_operator_template(self,operator_loader, op, in_template, in_shape):
        ## restore operator
        key=str(op.name)+"_"+str(op.values)
        if key in self.operators:
            op_obj=self.operators[key]
        else:
            ## new operator (add new operator)
            cls = operator_loader.get_operator(op.name)
            op_obj = cls(op.values)
            self.operators[key]=op_obj
        ## get operator information
        out_template = op_obj.get_output_template(in_template)
        out_shape = op_obj.get_output_shape(in_shape)
        return out_template, out_shape

    def build_explanation_graph_template(
        self, graph, tensor_provider, operator_loader=None, cycle_node=[]
    ):
        """
        Args:
            graph: explanation graph object
            tensor_provider (SwitchTensorProvider): tensor provider
            operator_loader (OperatorLoader): operator loader
            cycle_node (List[int]): a list of sorted_id of cycle nodes

        Returns:
            A tuple containing goal_template and cycle_node
             - goal_template (List[Dict[str, Any]]): a list of goal templates
             - cycle_node (List[int]): a list of cycle node (if given cycle_node, it is updated)
        """
        # checking template
        goal_template = [None] * len(graph.goals)
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
                    if len(temp_goal["template"]) > 0:
                        if temp_goal["batch_flag"]:
                            path_batch_flag = True
                        node_shape.append(temp_goal["shape"])
                        node_template.append(temp_goal["template"])
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
                goal_template[i] = {
                    "template": [],
                    "batch_flag": False,
                    "shape": path_shape,
                }
            else:
                if len(path_template_list) != 1:
                    print("[WARNING] missmatch indices:", path_template_list)
                goal_template[i] = {
                    "template": path_template_list[0],
                    "batch_flag": path_batch_flag,
                    "shape": path_shape,
                }
        ##
        return goal_template, cycle_node


class SwitchTensor:
    """ This class connect a tensor with a switch
    
    Attributes:
        name (str): switch name
        shape_set (Set[Tuple[int,...]]): a set of tensor shapes associated with this switch
        ph_names (List[str]): generated from the switch name
        vocab_name (str): generated from the switch name
        var_name (str): generated from the switch name
        value (Any): this values is used to represent x in the special tensor atom: get(_,x)

    """
    def __init__(self, sw_name: str) -> None:
        self.value = None
        self.name = sw_name
        self.shape_set = set([])
        self.type_set = set([])
        self.ph_names = self.get_placeholder_name(sw_name)
        self.vocab_name = self.make_vocab_name(sw_name) # update self.value
        self.var_name = self.make_var_name(sw_name)

    def enabled_placeholder(self):
        return len(self.ph_names) == 0

    def add_shape(self, shape: Tuple[int,...]) -> None:
        self.shape_set.add(shape)

    def add_type(self, tensor_type: str) -> None:
        self.type_set.add(tensor_type)

    def get_shape(self) -> Tuple[int,...]:
        assert len(self.shape_set) == 1, (
            self.name + ": shape is not unique:" + str(self.shape_set)
        )
        return list(self.shape_set)[0]

    def get_type(self) -> str:
        assert len(self.type_set) == 1, (
            self.name + ": shape is not unique:" + str(self.type_set)
        )
        return list(self.type_set)[0]


    @staticmethod
    def get_placeholder_name(name: str) -> List[Union[str, Any]]:
        pattern = r"(\$placeholder[0-9]+\$)"
        m = re.finditer(pattern, name)
        names = [el.group(1) for el in m]
        return names
    
    def make_vocab_name(self, name: str) -> str:
        m = re.match(r"^tensor\(get\((.*),([0-9]*)\)\)$", name)
        if m:
            name = "tensor(" + m.group(1) + ")"
            self.value = int(m.group(2))
        pattern = r"\$(placeholder[0-9]+)\$"
        m = re.sub(pattern, "", name)
        return self.make_var_name(m)
    @staticmethod
    def make_var_name(name: str) -> str:
        return re.sub(r"[\[\],\)\(\'$]+", "_", name)


class VocabSet:
    """ This class connect a value object with a vovabrary via placeholders

    vocab -> placeholder ->values

    Attributes:
        vocab_values (Dict[str, List[Any]]): the key is a vocab name, and the value ia a list of values.
        value_index (Dict[Tuple[str,Any],int): the key is a tuple of a vocab name and a value, and the value is an index.

    """

    def __init__(self) -> None:
        # vocab name => a list of values
        self.vocab_values = None
        # vocab name, value => index
        self.value_index = None

    def build_from_ph(self, ph_graph: 'PlaceholderGraph') -> None:
        """ This method builds vocab_values and value_index from PlaceholderGraph.
        """
        vocab_ph = ph_graph.vocab_ph
        ph_values = ph_graph.ph_values #
        vocab_values = {}
        for vocab_name, phs in vocab_ph.items():
            for ph in phs:
                if vocab_name not in vocab_values:
                    vocab_values[vocab_name] = set()
                vocab_values[vocab_name] |= ph_values[ph]
        self.vocab_values = {k: list(v) for k, v in vocab_values.items()}
        self.value_index = self._build_value_index()

    def _build_value_index(self) -> Dict[Tuple[str, int32], int]:
        value_index = {}
        for vocab_name, values in self.vocab_values.items():
            for i, v in enumerate(sorted(values)):
                value_index[(vocab_name, v)] = i
        return value_index

    def get_values_index(self, vocab_name: str, value: Union[int, int32]) -> int:
        key = (vocab_name, value)
        if key in self.value_index:
            return self.value_index[key]
        else:
            return 0

    def get_values(self, vocab_name: str) -> Optional[List[int32]]:
        if vocab_name not in self.vocab_values:
            return None
        return self.vocab_values[vocab_name]


class PlaceholderGraph:
    """ This class build a graph related to placeholders

    vocab <-- sw_info --> placeholder --> values

    Attributes:
        vocab_ph (Dict[str, Set[str]]): vocab_name => a set of nemes of placeholders
        ph_vocab (Dict[str, Set[str]]): placeholder name => a set of the vocabs
        ph_values (Dict[str,Set[Any]]): placeholder name => a set of values
        vocab_shape (Dict[str, Set[Tuple[int,...]]]): vocab_name => a set of shapes

    """

    def __init__(self) -> None:
        self.vocab_ph = None
        self.ph_vocab = None
        self.ph_values = None #Dict[str,Set[Any]]
        self.vocab_shape = None

    def _build_ph_values(self, input_data: List[Dict[str, Union[int, List[str], ndarray]]]) -> None:
        ph_values = {}
        for g in input_data:
            for ph in g["placeholders"]:
                if ph not in ph_values:
                    ph_values[ph] = set()
            placeholders = [ph for ph in g["placeholders"]]
            rt = np.transpose(g["records"])
            for i, item in enumerate(rt):
                ph_values[placeholders[i]] |= set(item)
        self.ph_values = ph_values

    def _build_vocab_ph(self, ph_values: Dict[str, Set[int32]], sw_info: Dict[str, SwitchTensor]) -> None:
        # ph_vocab/vocab_ph: ph_name <== sw_info ==> vocab_name
        # vocab_shape: vocab_name => shape
        ph_vocab = {ph_name: set() for ph_name in ph_values.keys()}
        vocab_ph = {sw.vocab_name: set() for sw in sw_info.values()}
        vocab_shape = {sw.vocab_name: set() for sw in sw_info.values()}
        for sw_name, sw in sw_info.items():
            ## build vocab. shape
            if sw.vocab_name not in vocab_shape:
                vocab_shape[sw.vocab_name] = set()
            vocab_shape[sw.vocab_name] |= sw.shape_set
            ## build ph_vocab/vocab_ph
            ph_list = sw.ph_names
            if len(ph_list) == 1:
                vocab_ph[sw.vocab_name].add(ph_list[0])
                ph_vocab[ph_list[0]].add(sw.vocab_name)
            elif len(ph_list) > 1:
                print("[ERROR] not supprted: one placeholder for one term")
        self.ph_vocab = ph_vocab
        self.vocab_ph = vocab_ph
        self.vocab_shape = vocab_shape
        ##

    def build(self, input_data: List[Dict[str, Union[int, List[str], ndarray]]], sw_info: Dict[str, SwitchTensor]) -> None:
        if input_data is not None:
            self._build_ph_values(input_data)
        else:
            self.ph_values = {}
        self._build_vocab_ph(self.ph_values, sw_info)

VarType = Union[
        Dict[str, Union[str, Tuple[int, int], List[Union[int64, int]]]],
        Dict[str, Union[str, List[int]]],
        Dict[str, Union[str, List[Union[int64, int]]]]
        ]
class SwitchTensorProvider:
    """ This class provides information of switches

    Attributes:
        tensor_embedding (Dict[str, Tensor]): embedding tensor
        sw_info (Dict[str, SwitchTensor]): switch infomation
        ph_graph (PlaceholderGraph): associated placeholder graph
        input_feed_dict (Dict[PlaceholderData, Tensor]): feed_dict to replace a placeholder with a tensor
        params (Dict[str,Tuple[Parameter,str]]): pytorch parameters associated with all switches provided by this provider

    """

    def __init__(self) -> None:
        self.tensor_embedding = None
        self.sw_info = None
        self.ph_graph = None
        self.input_feed_dict = None
        self.params = {}

    def get_embedding(self, name):
        if self.input_feed_dict is None:
            return self.tensor_embedding[name]
        else:
            key = self.tensor_embedding[name]
            return self.input_feed_dict[key]

    def set_embedding(self, name, var):
        self.tensor_embedding[name] = var

    def set_input(self, feed_dict: Dict[PlaceholderData, Tensor]) -> None:
        self.input_feed_dict = feed_dict

    def get_placeholder_name(self, name: str) -> List[Union[str, Any]]:
        """ 
        Args:
            switch name (str): switch name
        Returns:
            placeholder name
        """
        return self.sw_info[name].ph_names

    def get_switch(self, name: str) -> SwitchTensor:
        """ 
        Args:
            switch name (str): switch name
        Returns:
            switch tensor
        """
        return self.sw_info[name]

    def get_placeholder_var_name(self, name: str) -> str:
        """ 
        Args:
            name (str): placeholder name
        Returns:
            placeholder variable name in PlaceholderData.name
        """
        return re.sub(r"\$", "", name)

    def add_param(self, name: str, param: Parameter, tensor_type:str) -> None:
        """ This is called in the initializer of TorchTensor
        Args:
            name (str): Tensor's name
            param (Parameter): Parameter
            tensor_type (str): tensor type like sparse
        """
        self.params[name] = (param,tensor_type)

    def get_param(self, name: str) -> Tuple[Parameter,str]:
        """ 
        Args:
            name (str): Tensor's name
        """
        return self.params[name]

    def convert_value_to_index(self, value: Union[int, int32], ph_name: Union[str, str_]) -> int:
        ph_vocab = self.ph_graph.ph_vocab
        vocab_name = self.ph_graph.ph_vocab[ph_name]
        vocab_name = list(vocab_name)[0]
        index = self.vocab_set.get_values_index(vocab_name, value)
        return index

    def is_convertable_value(self, ph_name: str) -> bool:
        if ph_name in self.ph_graph.ph_vocab:
            return len(self.ph_graph.ph_vocab[ph_name]) > 0
        else:
            return False

    def _build_sw_info(self, graph, tensor_info: TensorInfoMapper) -> Dict[str, SwitchTensor]:
        """ This function builds sw_info from the explanation graph 
        """
        sw_info = {}
        for g in graph.goals:
            for path in g.paths:
                for sw in path.tensor_switches:
                    if sw.name not in sw_info:
                        sw_obj = SwitchTensor(sw.name)
                        sw_info[sw.name] = sw_obj
                    else:
                        sw_obj = sw_info[sw.name]
                    value_list = [el for el in sw.values]
                    if sw.name in tensor_info.shape:
                        shape = tuple(tensor_info.shape[sw.name])
                    sw_obj.add_shape(shape)
                    if sw.name in tensor_info.type:
                        tensor_type = tensor_info.type[sw.name]
                    sw_obj.add_type(tensor_type)
        return sw_info

    def _build_vocab_var_type(self, ph_graph: PlaceholderGraph, vocab_set: VocabSet, embedding_generators: List[BaseEmbeddingGenerator], sw_info: Dict[str,SwitchTensor]) -> Dict[str, VarType]:
        """ This function builds temporal object: vocab_name =>  var_type

        Note:
            vocab_var_type (Dict[str,VarType]):
            Var type is a dictionary like follows:

            ::

                # var_type["type"]=="dataset"
                var_type={
                    "dataset_shape": Tuple[int, ...],
                    "shape": List[int, ...],
                }
                # var_type["type"]=="onehot"
                var_type={
                    "value": int,
                    "shape": List[int, ...],
                }
                # var_type["type"]=="variable"
                var_type={
                    "shape": List[int, ...]
                }

        """
        vocab_var_type = {}
        for vocab_name, shapes in ph_graph.vocab_shape.items():
            values = vocab_set.get_values(vocab_name)
            ##
            if len(shapes) == 1:
                shape = list(shapes)[0]
                if values is not None:
                    # s=[len(values)]+list(shape)
                    s = [max(values) + 1] + list(shape)
                else:
                    s = list(shape)
            else:
                shape = sorted(list(shapes), key=lambda x: len(x), reverse=True)[0]
                s = list(shape)
            ##
            var_type = {}
            dataset_flag = False
            for eg in embedding_generators:
                if eg.is_embedding(vocab_name):
                    dataset_shape = eg.get_shape(vocab_name)
                    var_type["type"] = "dataset"
                    var_type["dataset_shape"] = dataset_shape
                    var_type["shape"] = s
                    dataset_flag = True
            if dataset_flag:
                pass
            elif vocab_name[:14] == "tensor_onehot_":
                m = re.match(r"tensor_onehot_([\d]*)_", vocab_name)
                if m:
                    d = int(m.group(1))
                    if len(s) == 1:
                        var_type["type"] = "onehot"
                        var_type["value"] = d
                        var_type["shape"] = s
                    else:
                        print("[ERROR]")
                else:
                    print("[ERROR]")
            else:
                var_type["type"] = "variable"
                var_type["shape"] = s
                # get tensor type from sw_info
                sw_obj=None
                for k, sw in sw_info.items():
                    if vocab_name==sw.vocab_name:
                        sw_obj=sw
                if sw_obj is not None:
                    var_type["tensor_type"] =sw_obj.get_type()
            vocab_var_type[vocab_name] = var_type
        return vocab_var_type

    def build(
        self,
        graph,
        tensor_shapes,
        input_data,
        flags,
        load_vocab=False,
        embedding_generators=[],
        verbose=False,
    ):
        """
        As a preparation before creating a computational graph, associate switches and tensors
        
         1. building PlaceholderGraph
         2. building/loading VocabSet
         3. associating VocabSet with variable types such as dataset, constant one-hot, and variables.
         4. building placeholders
         5. associating switches with tensors using embedding generators
        
        As a result, vocab_var, ph_var, and tensor_embedding in this class are constructed
        """
        # sw_info: switch name =>SwitchTensor
        sw_info = self._build_sw_info(graph, tensor_shapes)
        #
        ph_graph = PlaceholderGraph()
        ph_graph.build(input_data, sw_info)

        ## build vocab group
        if load_vocab:
            print("[LOAD]", flags.vocab)
            with open(flags.vocab, mode="rb") as f:
                vocab_set = pickle.load(f)
        else:
            vocab_set = VocabSet()
            vocab_set.build_from_ph(ph_graph)
            print("[SAVE]", flags.vocab)
            with open(flags.vocab, mode="wb") as f:
                pickle.dump(vocab_set, f)
        ##
        self.vocab_var_type = self._build_vocab_var_type(
            ph_graph, vocab_set, embedding_generators, sw_info
        )
        self.vocab_set = vocab_set
        self.ph_graph = ph_graph
        self.sw_info = sw_info
        ##
        # build placeholders
        # ph_var    : ph_name => placeholder
        ph_var = {}
        batch_size = flags.sgd_minibatch_size
        for ph_name in ph_graph.ph_values.keys():
            ph_var_name = self.get_placeholder_var_name(ph_name)
            ph_var[ph_name] = PlaceholderData(
                name=ph_var_name, shape=(batch_size,), dtype=self.integer_dtype
            )
        #
        ## assigning tensor variable
        ## vocab_var: vocab_name => variable
        ##
        vocab_var = {}
        for vocab_name, var_type in self.vocab_var_type.items():
            values = vocab_set.get_values(vocab_name)
            if var_type["type"] == "dataset":
                print(
                    ">> dataset >>",
                    vocab_name,
                    ":",
                    var_type["dataset_shape"],
                    "=>",
                    var_type["shape"],
                )
            elif var_type["type"] == "onehot":
                print(">> onehot  >>", vocab_name, ":", var_type["shape"])
                d = var_type["value"]
                var = self.tensor_onehot_class(self, var_type["shape"][0], d)
                vocab_var[vocab_name] = var
            else:
                print(">> variable>>", vocab_name, ":", var_type)
                tensor_type = None
                if "tensor_type" in var_type:
                    tensor_type = var_type["tensor_type"]
                var = self.tensor_class(self, vocab_name, var_type["shape"], tensor_type=tensor_type)
                vocab_var[vocab_name] = var
        # converting PRISM switches to Tensorflow Variables
        # tensor_embedding: sw_name => tensor
        tensor_embedding = {}
        for sw_name, sw in sw_info.items():
            vocab_name = sw.vocab_name
            var_name = sw.var_name
            ph_list = sw.ph_names
            if len(ph_list) == 0:
                dataset_flag = False
                for eg in embedding_generators:
                    if eg.is_embedding(vocab_name):
                        dataset_flag = True
                        # dataset without placeholder
                        shape = list(list(sw.shape_set)[0])
                        if sw.value is None:
                            var = eg.get_embedding(vocab_name, shape)
                            if verbose:
                                print("ph_list==0 and value==none")
                                print((vocab_name, ":", var.shape))
                            tensor_embedding[sw_name] = var
                        else:
                            var = eg.get_embedding(vocab_name)
                            if verbose:
                                print("ph_list==0 and value enbabled")
                                print((vocab_name, ":", var.shape, "=>", shape))
                            index = vocab_set.get_values_index(vocab_name, sw.value)
                            if verbose:
                                print(index, sw.value)
                            #tensor_embedding[sw_name] = var[sw.value]
                            tensor_embedding[sw_name] = self.tensor_gather_class(self, var, sw.value)
                if not dataset_flag:
                    # trainig variable without placeholder
                    var = vocab_var[vocab_name]
                    if verbose:
                        print("ph_list==0 and no dataset")
                        print((vocab_name, ":", var.shape))
                    tensor_embedding[sw_name] = var
            elif len(ph_list) == 1:
                dataset_flag = False
                for eg in embedding_generators:
                    if eg.is_embedding(vocab_name):
                        dataset_flag = True
                        # dataset with placeholder
                        shape = [batch_size] + list(list(sw.shape_set)[0])
                        var = eg.get_embedding(vocab_name)
                        # var = eg.get_embedding(vocab_name, shape)
                        if verbose:
                            print("ph_list==1 and dataset enabled")
                            print((vocab_name, ":", var.shape, "=>", shape))
                        tensor_embedding[sw_name] = var
                if not dataset_flag:
                    # trainig variable with placeholder
                    var = vocab_var[vocab_name]
                    if verbose:
                        print("ph_list==1 and dataset disabled")
                        print((vocab_name, ":", var.shape, "=>", shape))
                    ph = ph_var[ph_list[0]]
                    tensor_embedding[sw_name] = self.tensor_gather_class(self, var, ph)
            else:
                print("[WARM] unknown embedding:", sw_name)
        self.vocab_var = vocab_var
        self.ph_var = ph_var
        self.tensor_embedding = tensor_embedding
        return tensor_embedding
