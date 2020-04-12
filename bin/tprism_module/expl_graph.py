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

import tprism_module.expl_pb2 as expl_pb2
import tprism_module.op.base
import tprism_module.loss.base

class PlaceholderData():
    def __init__(self,name,shape=(),dtype=None,ref=None):
        self.name=name
        self.reference=ref
        self.shape=shape
        self.dtype=dtype



def load_input_data(data_filename_list):
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


def load_input_json(filename):
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


def load_input_h5(filename):
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


def merge_input_data(input_data_list):
    merged_data = {}
    for datasets in input_data_list:
        for data in datasets:
            goal_id = data["goal_id"]
            if goal_id not in merged_data:
                merged_data[goal_id] = data
            else:
                merged_data[goal_id]["records"].extend(data[goal_id]["records"])

    return list(merged_data.values())


def load_explanation_graph(expl_filename, option_filename):
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
    def __init__(self):
        self.operators = {}
        self.base_module_name = "tprism_module.op."
        self.module = None

    # a snake case operator name to class name
    def to_class_name(self, snake_str):
        components = snake_str.split("_")
        return "".join(x.title() for x in components)

    # class name to a snake case operator name
    def to_op_name(self, cls_name):
        _underscorer1 = re.compile(r"(.)([A-Z][a-z]+)")
        _underscorer2 = re.compile("([a-z0-9])([A-Z])")
        subbed = _underscorer1.sub(r"\1_\2", cls_name)
        return _underscorer2.sub(r"\1_\2", subbed).lower()

    def get_operator(self, name):
        assert name in self.operators, "%s is not found" % (name)
        cls = self.operators[name]
        assert cls is not None, "%s is not found" % (name)
        return cls

    def load_all(self, path):
        search_path = os.path.dirname(__file__) + "/" + path
        for fpath in glob.glob(search_path+"*.py"):
            print("[LOAD]", fpath)
            name = os.path.basename(os.path.splitext(fpath)[0])
            module_name = self.base_module_name + name
            module = importlib.machinery.SourceFileLoader(
                module_name, fpath
            ).load_module()
            self.load_module(module)

    def load_module(self, module):
        for cls_name, cls in inspect.getmembers(module, inspect.isclass):
            if issubclass(cls, tprism_module.op.base.BaseOperator):
                print("[IMPORT]", cls_name)
                op_name = self.to_op_name(cls_name)
                self.operators[op_name] = cls


class LossLoader:
    def __init__(self):
        self.module = None
        self.base_module_name = "tprism_module.loss."
        self.losses = {}

    # a snake case operator name to class name
    def to_class_name(self, snake_str):
        components = snake_str.split("_")
        return "".join(x.title() for x in components)

    # class name to a snake case operator name
    def to_op_name(self, cls_name):
        _underscorer1 = re.compile(r"(.)([A-Z][a-z]+)")
        _underscorer2 = re.compile("([a-z0-9])([A-Z])")
        subbed = _underscorer1.sub(r"\1_\2", cls_name)
        return _underscorer2.sub(r"\1_\2", subbed).lower()

    def get_loss(self, name):
        if name in self.losses:
            cls = self.losses[name]
            return cls
        else:
            return None

    def load_all(self, path):
        search_path = os.path.dirname(__file__) + "/" + path
        for fpath in glob.glob(search_path+"*.py"):
            print("[LOAD]", fpath)
            name = os.path.basename(os.path.splitext(fpath)[0])
            module_name = self.base_module_name + name
            module = importlib.machinery.SourceFileLoader(
                module_name, fpath
            ).load_module()
            self.load_module(module)

    def load_module(self, module):
        for cls_name, cls in inspect.getmembers(module, inspect.isclass):
            if issubclass(cls, tprism_module.loss.base.BaseLoss):
                print("[IMPORT]", cls_name)
                op_name = self.to_op_name(cls_name)
                self.losses[op_name] = cls

class ComputationalExplGraph:
    def __init__(self):
        self.path_inside = None
        self.path_template = None

    def get_unique_list(self,seq):
        seen = []
        return [x for x in seq if x not in seen and not seen.append(x)]

    # [['i'], ['i','l', 'j'], ['j','k']] => ['l','k']
    def compute_output_template(self,template):
        counter = collections.Counter(chain.from_iterable(template))
        out_template = [k for k, cnt in counter.items() if cnt == 1 and k != "b"]
        return sorted(out_template)


    # [['i'], ['i','l', 'j'], ['j','k']] => ['l','k']
    # [[3], [3, 4, 5], [5,6]] => [4,6]
    def compute_output_shape(self,out_template, sw_node_template, sw_node_shape):
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


    def unify_shapes(self,path_shapes):
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


    def build_explanation_graph_template(self, graph, tensor_provider, operator_loader=None, cycle_node=[]):
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
                if cycle_detected:
                    continue
                sw_node_template = sw_template + node_template
                sw_node_shape = sw_shape + node_shape
                # constructing einsum operation using template and inside
                out_template = self.compute_output_template(sw_node_template)
                out_shape = self.compute_output_shape(
                    out_template, sw_node_template, sw_node_shape
                )
                if len(sw_node_template) > 0:  # condition for einsum
                    if path_batch_flag:
                        out_template = ["b"] + out_template
                ## computing operaters
                for op in path.operators:
                    cls = operator_loader.get_operator(op.name)
                    op_obj = cls(op.values)
                    out_template = op_obj.get_output_template(out_template)
                path_template.append(out_template)
                path_shape.append(out_shape)
                ##
            ##
            path_template_list = self.get_unique_list(path_template)
            path_shape = self.unify_shapes(path_shape)
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
    def __init__(self, sw_name):
        self.name = sw_name
        self.shape_set = set([])
        self.value =None
        self.ph_names = self.get_placeholder_name(sw_name)
        self.vocab_name = self.make_vocab_name(sw_name)
        self.var_name = self.make_var_name(sw_name)

    def enabled_placeholder(self):
        return len(self.ph_names) == 0

    def add_shape(self, shape):
        self.shape_set.add(shape)

    def get_shape(self):
        assert len(self.shape_set) == 1, (self.name + ": shape is not unique:" + str(self.shape_set))
        return list(self.shape_set)[0]

    def get_placeholder_name(self, name):
        pattern = r"(\$placeholder[0-9]+\$)"
        m = re.finditer(pattern, name)
        names = [el.group(1) for el in m]
        return names

    def make_vocab_name(self, name):
        m=re.match(r'^tensor\(get\((.*),([0-9]*)\)\)$',name)
        if m:
            name="tensor("+m.group(1)+")"
            self.value =int(m.group(2))
        pattern = r"\$(placeholder[0-9]+)\$"
        m = re.sub(pattern, "", name)
        return self.make_var_name(m)

    def make_var_name(self, name):
        return re.sub(r"[\[\],\)\(\'$]+", "_", name)

class VocabSet:
    def __init__(self):
        # vocab name => a list of values
        self.vocab_values = None
        # vocab name, value => index
        self.value_index = None

    def build_from_ph(self, ph_graph):
        vocab_ph=ph_graph.vocab_ph
        ph_values=ph_graph.ph_values
        vocab_values = {}
        for vocab_name, phs in vocab_ph.items():
            for ph in phs:
                if vocab_name not in vocab_values:
                    vocab_values[vocab_name] = set()
                vocab_values[vocab_name] |= ph_values[ph]
        self.vocab_values = {k: list(v) for k, v in vocab_values.items()}
        self.value_index = self._build_value_index()

    def _build_value_index(self):
        value_index = {}
        for vocab_name, values in self.vocab_values.items():
            for i, v in enumerate(sorted(values)):
                value_index[(vocab_name, v)] = i
        return value_index

    def get_values_index(self, vocab_name, value):
        key = (vocab_name, value)
        if key in self.value_index:
            return self.value_index[key]
        else:
            return 0

    def get_values(self, vocab_name):
        if vocab_name not in self.vocab_values:
            return None
        return self.vocab_values[vocab_name]

class PlaceholderGraph:
    def __init__(self):
        self.vocab_ph=None
        self.ph_vocab=None
        self.ph_values=None
        self.vocab_shape=None

    def _build_ph_values(self, input_data):
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

    def _build_vocab_ph(self,ph_values,sw_info):
        # ph_vocab/vocab_ph: ph_name <==> vocab_name
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
        self.ph_vocab=ph_vocab
        self.vocab_ph=vocab_ph
        self.vocab_shape=vocab_shape
        ##
    def build(self,input_data,sw_info):
        if input_data is not None:
            self._build_ph_values(input_data)
        else:
            self.ph_values = {}
        self._build_vocab_ph(self.ph_values,sw_info)


class SwitchTensorProvider:
    def __init__(self):
        self.tensor_embedding=None
        self.sw_info=None
        self.ph_graph=None
        self.input_feed_dict=None
        self.params={}

    def get_embedding(self,name):
        if self.input_feed_dict is None:
            return self.tensor_embedding[name]
        else:
            key=self.tensor_embedding[name]
            return self.input_feed_dict[key]
    
    def set_embedding(self,name,var):
        self.tensor_embedding[name]=var
    
    def set_input(self, feed_dict):
        self.input_feed_dict=feed_dict

    def get_placeholder_name(self, name):
        return self.sw_info[name].ph_names

    def get_switch(self, name):
        return self.sw_info[name]

    def get_placeholder_var_name(self, name):
        return re.sub(r"\$", "", name)

    def add_param(self, name, param):
        self.params[name]=param
    def get_param(self, name):
        return self.params[name]

    def convert_value_to_index(self, value, ph_name):
        ph_vocab=self.ph_graph.ph_vocab
        vocab_name = self.ph_graph.ph_vocab[ph_name]
        vocab_name = list(vocab_name)[0]
        index = self.vocab_set.get_values_index(vocab_name, value)
        return index

    def is_convertable_value(self, ph_name):
        if ph_name in self.ph_graph.ph_vocab:
            return len(self.ph_graph.ph_vocab[ph_name])>0
        else:
            return False

    def _build_sw_info(self,graph,options):
        tensor_shape = {
            el.tensor_name: [d for d in el.shape] for el in options.tensor_shape
        }
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
                    if sw.name in tensor_shape:
                        shape = tuple(tensor_shape[sw.name])
                    sw_obj.add_shape(shape)
        return sw_info

    def _build_vocab_var_type(self,ph_graph,vocab_set,embedding_generators):
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
            var_type={}
            dataset_flag=False
            for eg in embedding_generators:
                if eg.is_embedding(vocab_name):
                    dataset_shape=eg.get_shape(vocab_name)
                    var_type["type"]="dataset"
                    var_type["dataset_shape"]=dataset_shape
                    var_type["shape"]=s
                    dataset_flag=True
            if dataset_flag:
                pass
            elif vocab_name[:14] == "tensor_onehot_":
                m = re.match(r"tensor_onehot_([\d]*)_", vocab_name)
                if m:
                    d = int(m.group(1))
                    if len(s) == 1:
                        var_type["type"]="onehot"
                        var_type["value"]=d
                        var_type["shape"]=s
                    else:
                        print("[ERROR]")
                else:
                    print("[ERROR]")
            else:
                var_type["type"]="variable"
                var_type["shape"]=s
            vocab_var_type[vocab_name]=var_type
        return vocab_var_type

    def build(
        self,
        graph,
        options,
        input_data,
        flags,
        load_embeddings=False,
        embedding_generators=[],
    ):
        # sw_info: switch name =>SwitchTensor
        sw_info = self._build_sw_info(graph,options)
        # 
        ph_graph=PlaceholderGraph()
        ph_graph.build(input_data,sw_info)
        
        ## build vocab group
        if load_embeddings:
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
        self.vocab_var_type=self._build_vocab_var_type(ph_graph,vocab_set,embedding_generators)
        self.vocab_set = vocab_set
        self.ph_graph=ph_graph
        self.sw_info = sw_info
        ##
        # build placeholders
        #ph_var    : ph_name => placeholder
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
        vocab_var={}
        #initializer = tf.contrib.layers.xavier_initializer()
        for vocab_name, var_type in self.vocab_var_type.items():
            values = vocab_set.get_values(vocab_name)
            if var_type["type"]=="dataset":
                print(">> dataset >>", vocab_name, ":",var_type["dataset_shape"],"=>", var_type["shape"])
            elif var_type["type"]=="onehot":
                print(">> onehot  >>", vocab_name, ":", var_type["shape"])
                d=var_type["value"]
                var = self.tensor_onehot_class(self,var_type["shape"][0],d)
                vocab_var[vocab_name] = var
            else:
                print(">> variable>>", vocab_name, ":", var_type["shape"])
                var = self.tensor_class(self,vocab_name,var_type["shape"])
                vocab_var[vocab_name] = var
        # converting PRISM switches to Tensorflow Variables
        # tensor_embedding: sw_name => tensor
        tensor_embedding = {}
        for sw_name, sw in sw_info.items():
            vocab_name = sw.vocab_name
            var_name = sw.var_name
            ph_list = sw.ph_names
            if len(ph_list) == 0:
                dataset_flag=False
                for eg in embedding_generators:
                    if eg.is_embedding(vocab_name):
                        dataset_flag=True
                        # dataset without placeholder
                        shape = list(list(sw.shape_set)[0])
                        if sw.value is None:
                            print("ph_list==0 and value==none")
                            var = eg.get_embedding(vocab_name, shape)
                            tensor_embedding[sw_name] = var
                        else:
                            print("ph_list==0 and value enbabled")
                            var = eg.get_embedding(vocab_name)
                            if verbose:
                                print((vocab_name,":", var.shape,"=>",shape))
                            index=vocab_set.get_values_index(vocab_name, sw.value)
                            if verbose:
                                print(index,sw.value)
                            tensor_embedding[sw_name] = var[sw.value] # TODO
                if not dataset_flag:
                    print("ph_list==0 and no dataset")
                    # trainig variable without placeholder
                    var = vocab_var[vocab_name]
                    tensor_embedding[sw_name] = var
            elif len(ph_list) == 1:
                dataset_flag=False
                for eg in embedding_generators:
                    if eg.is_embedding(vocab_name):
                        dataset_flag=True
                        # dataset with placeholder
                        print("ph_list==1 and dataset enabled")
                        shape = [batch_size] + list(list(sw.shape_set)[0])
                        var = eg.get_embedding(vocab_name)
                        #print(var)
                        #var = eg.get_embedding(vocab_name, shape)
                        tensor_embedding[sw_name] = var
                if not dataset_flag:
                    print("ph_list==1 and dataset disabled")
                    # trainig variable with placeholder
                    var = vocab_var[vocab_name]
                    ph = ph_var[ph_list[0]]
                    tensor_embedding[sw_name] = self.tensor_gather(self,var, ph)
            else:
                print("[WARM] unknown embedding:",sw_name)
        self.vocab_var = vocab_var
        self.ph_var = ph_var
        self.tensor_embedding = tensor_embedding
        return tensor_embedding

