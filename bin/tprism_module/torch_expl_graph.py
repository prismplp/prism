import torch
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

from tprism_module.expl_graph import ComputationalExplGraph,SwitchTensorProvider
from tprism_module.expl_graph import PlaceholderGraph,VocabSet,OperatorLoader
from tprism_module.expl_graph import PlaceholderData

class TorchComputationalExplGraph(ComputationalExplGraph):
    def __init__(self):
        super().__init__()

    def build_explanation_graph(self, graph, tensor_provider, cycle_embedding_generator=None):
        operator_loader = OperatorLoader()
        operator_loader.load_all("op/torch_")
        goal_template, cycle_node = self.build_explanation_graph_template(
            graph, tensor_provider, operator_loader
        )
        self.operator_loader=operator_loader
        self.goal_template=goal_template
        self.cycle_node=cycle_node
        
    def forward(self, graph, tensor_provider, cycle_embedding_generator=None):
        goal_template=self.goal_template
        cycle_node=self.cycle_node
        operator_loader=self.operator_loader
        # goal_template
        # converting explanation graph to computational graph
        goal_inside = [None] * len(graph.goals)
        for i in range(len(graph.goals)):
            g = graph.goals[i]
            if False:
                print(
                    "=== tensor equation (node_id:%d, %s) ==="
                    % (g.node.sorted_id, g.node.goal.name)
                )
            self.path_inside = []
            self.path_template = []
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
                    sw_var = tensor_provider.get_embedding(sw.name)
                    sw_inside.append(sw_var)
                    """
                    tf.add_to_collection(
                        tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(sw_var)
                    )
                    """
                prob_sw_inside = 1.0
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
                        temp_goal_inside = cycle_embedding_generator.get_embedding(
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
                        node_scalar_inside.append(temp_goal["inside"])
                ## building template and inside for all elements (switches and nodes) in the path
                sw_node_template = sw_template + node_template
                sw_node_inside = sw_inside + node_inside
                path_v = sorted(zip(sw_node_template, sw_node_inside), key=lambda x: x[0])
                template = [x[0] for x in path_v]
                inside = [x[1] for x in path_v]
                # constructing einsum operation using template and inside
                out_template = self.compute_output_template(template)
                # print(template,out_template)
                out_inside = prob_sw_inside
                if len(template) > 0:  # condition for einsum
                    lhs = ",".join(map(lambda x: "".join(x), template))
                    rhs = "".join(out_template)
                    if path_batch_flag:
                        rhs = "b" + rhs
                        out_template = ["b"] + out_template
                    einsum_eq = lhs + "->" + rhs
                    if False:
                        print("  index:", einsum_eq)
                        print("  var. :", inside)
                    out_inside = torch.einsum(einsum_eq, *inside) * out_inside
                for scalar_inside in node_scalar_inside:
                    out_inside = scalar_inside * out_inside
                ## computing operaters
                for op in path.operators:
                    if False:
                        print("  operator:", op.name)
                    cls = operator_loader.get_operator(op.name)
                    # print(">>>",op.values)
                    # print(">>>",cls)
                    op_obj = cls(op.values)
                    out_inside = op_obj.call(out_inside)
                    out_template = op_obj.get_output_template(out_template)
                ##
                self.path_inside.append(out_inside)
                self.path_template.append(out_template)
                ##
            ##
            path_template_list = self.get_unique_list(self.path_template)

            if len(path_template_list) == 0:
                goal_inside[i] = {
                    "template": [],
                    "inside": np.array(1),
                    "batch_flag": False,
                }
            else:
                if len(path_template_list) != 1:
                    print("[WARNING] missmatch indices:", path_template_list)
                goal_inside[i] = {
                    "template": path_template_list[0],
                    "inside": torch.sum(torch.stack(self.path_inside), dim=0),
                    "batch_flag": path_batch_flag,
                }
        return goal_inside

class TorchSwitchTensorProvider(SwitchTensorProvider):
    def __init__(self):
        super().__init__()

    def get_embedding(self,name):
        if self.input_feed_dict is None:
            return self.tensor_embedding[name]
        elif type(name) is str:
            key=self.tensor_embedding[name]
            if type(key) is PlaceholderData:
                return torch.tensor(self.input_feed_dict[key])
            else:
                return key
        elif type(name) is PlaceholderData:
            return torch.tensor(self.input_feed_dict[name])
        else:
            raise Exception('Unknoen embedding', name)



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
        ## Forward
        ##
        # build placeholders
        #ph_var    : ph_name => placeholder
        ph_var = {}
        batch_size = flags.sgd_minibatch_size
        for ph_name in ph_graph.ph_values.keys():
            ph_var_name = self.get_placeholder_var_name(ph_name)
            ph_var[ph_name] = PlaceholderData(
                name=ph_var_name, shape=(batch_size,), dtype=torch.int32
            )
        #
        ## assigning tensor variable
        ## vocab_var: vocab_name => variable
        ##
        vocab_var={}
        dtype = torch.float32
        #initializer = tf.contrib.layers.xavier_initializer()
        for vocab_name, var_type in self.vocab_var_type.items():
            values = vocab_set.get_values(vocab_name)
            if var_type["type"]=="dataset":
                print(">> dataset >>", vocab_name, ":",var_type["dataset_shape"],"=>", var_type["shape"])
            elif var_type["type"]=="onehot":
                print(">> onehot  >>", vocab_name, ":", var_type["shape"])
                d=var_type["value"]
                var = torch.eye(var_type["shape"][0])[d]
                vocab_var[vocab_name] = var
            else:
                print(">> variable>>", vocab_name, ":", var_type["shape"])
                """
                var = torch.Tensor(
                    vocab_name, shape=var_type["shape"], initializer=initializer, dtype=dtype
                )
                """
                var = torch.tensor(np.zeros(var_type["shape"],dtype=np.float32), requires_grad=True,dtype=dtype)
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
                            print((vocab_name,":", var.shape,"=>",shape))
                            index=vocab_set.get_values_index(vocab_name, sw.value)
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
                        var = eg.get_embedding(vocab_name, shape)
                        tensor_embedding[sw_name] = var
                if not dataset_flag:
                    print("ph_list==1 and dataset disabled")
                    # trainig variable with placeholder
                    var = vocab_var[vocab_name]
                    ph = ph_var[ph_list[0]]
                    tensor_embedding[sw_name] = torch.gather(var, ph)
            else:
                print("[WARM] unknown embedding:",sw_name)
        self.vocab_var = vocab_var
        self.ph_var = ph_var
        self.tensor_embedding = tensor_embedding
        return tensor_embedding
