#!/usr/bin/env python
import torch
import torch.optim as optim
import json
import os
import re
import numpy as np
from google.protobuf import json_format
from itertools import chain
import collections
import argparse
import time
import pickle

import tprism_module.expl_pb2 as expl_pb2
import tprism_module.expl_graph as expl_graph
import tprism_module.torch_expl_graph as torch_expl_graph
#import tprism_module.draw_graph as draw_graph
import tprism_module.torch_embedding_generator as embed_gen
from  tprism_module.util import  to_string_goal, Flags, build_goal_dataset, split_goal_dataset, get_goal_dataset
from graphviz import Digraph
import re
import numpy as np
import torch
import torch.nn.functional as F
import sklearn.metrics

def make_dot(var, params):
    """ Produces Graphviz representation of PyTorch autograd graph

    Blue nodes are the Variables that require grad, orange are Tensors
    saved for backward in torch.autograd.Function
    Args:
        var: output Variable
        params: dict of (name, Variable) to add names to node that
            require grad (TODO: make optional)
    """
    param_map = {id(v): k for k, v in params.items()}
    print(param_map)

    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()

    def size_to_str(size):
        return '('+(', ').join(['%d'% v for v in size])+')'

    def add_nodes(var):
        if var not in seen:
            if torch.is_tensor(var):
                dot.node(str(id(var)), size_to_str(var.size()), fillcolor='orange')
            elif hasattr(var, 'variable'):
                u = var.variable
                node_name = '%s\n %s' % (param_map.get(id(u)), size_to_str(u.size()))
                dot.node(str(id(var)), node_name, fillcolor='lightblue')
            else:
                dot.node(str(id(var)), str(type(var).__name__))
            seen.add(var)
            if hasattr(var, 'next_functions'):
                for u in var.next_functions:
                    if u[0] is not None:
                        dot.edge(str(id(u[0])), str(id(var)))
                        add_nodes(u[0])
            if hasattr(var, 'saved_tensors'):
                for t in var.saved_tensors:
                    dot.edge(str(id(t)), str(id(var)))
                    add_nodes(t)
    add_nodes(var.grad_fn)
    return dot

  
def display_graph(y,name):
  s=make_dot(y,{})
  s.format = 'png'
  s.render(name)

class TprismEvaluator():
    def __init__(self, goal_dataset = None):
        if goal_dataset is not None:
            self.total_loss = [0.0 for _ in range(len(goal_dataset))]
    def init_batch(self):
        self.output=[]
        self.true_y=[]
        self.pred_y=[]
    def eval_batch(self,loss,output,label,j,num_itr):
        """
            This function is called the last of batch iteration
              loss,output, label: return values of forward
              j: goal index of iteration
              num_itr:: number of iterations
        """
        _loss=loss[j].detach().numpy()
        _o=output[j].detach().numpy()
        if len(output[j].shape)>=2:
            _pred_y=torch.argmax(output[j],dim=1).detach().numpy()
            self.pred_y.extend(_pred_y)
        if label is not None:
            _true_y=label[j].detach().numpy()
            self.true_y.extend(_true_y)
        self.output.extend(_o)
        self.total_loss[j] += np.mean(_loss) / num_itr
    def eval(self,loss,output,label):
        self.total_loss = []
        self.output=[]
        self.true_y=[]
        self.pred_y=[]
        for j in range(len(loss)):
            _loss=loss[j].detach().numpy()
            self.output.append(output[j].detach().numpy())
            if len(output[j].shape)>=2:
                self.pred_y.append(torch.argmax(output[j],dim=1).detach().numpy())
            if label is not None:
                self.true_y.append(label[j].detach().numpy())
            self.total_loss.append(np.mean(_loss))


class TprismModel():
    def __init__(self, flags, options, graph, loss_cls):
        self.graph=graph
        self.flags=flags
        self.options=options
        self.loss_cls=loss_cls

    def build(self,input_data):
        self._build_embedding()
        self._set_data(input_data)
        self._build_explanation_graph()

    def _build_embedding(self):
        embedding_generators = []
        if self.flags.embedding:
            eg = embed_gen.DatasetEmbeddingGenerator()
            eg.load(self.flags.embedding)
            embedding_generators.append(eg)
        if self.flags.const_embedding:
            eg = embed_gen.ConstEmbeddingGenerator()
            eg.load(self.flags.const_embedding)
            embedding_generators.append(eg)
        cycle_embedding_generator = None
        if self.flags.cycle:
            cycle_embedding_generator = embed_gen.CycleEmbeddingGenerator()
            cycle_embedding_generator.load(self.options)
            embedding_generators.append(cycle_embedding_generator)
        self.embedding_generators = embedding_generators
        self.cycle_embedding_generator = cycle_embedding_generator
        
    def _set_data(self,input_data):
        self.tensor_provider = torch_expl_graph.TorchSwitchTensorProvider()
        self.tensor_provider.build(
            self.graph,
            self.options,
            input_data,
            self.flags,
            load_embeddings=False,
            embedding_generators=self.embedding_generators,
        )

    def _build_explanation_graph(self):
        self.comp_expl_graph=torch_expl_graph.TorchComputationalExplGraph(
            self.graph, self.tensor_provider, self.cycle_embedding_generator
        )
    def solve(self,goal_dataset):
        goal_inside=self.goal_inside
        inside = []
        for goal in goal_inside:
            l1 = goal["inside"]
            inside.append(l1)

        init = tf.global_variables_initializer()
        self.sess.run(init)
        feed_dict = {}
        prev_loss = None
        for step in range(self.flags.max_iterate):
            feed_dict = {}
            for embedding_generator in self.embedding_generators:
                if embedding_generator is not None:
                    feed_dict = embedding_generator.build_feed(feed_dict)
            out_inside = self.sess.run(inside, feed_dict=feed_dict)

            loss = 0
            for embedding_generator in self.embedding_generators:
                if embedding_generator is not None:
                    loss = embedding_generator.update(out_inside)
            print("step", step, "loss:", loss)
            if loss < 1.0e-20:
                break
            if prev_loss is not None and not loss < prev_loss:
                pass
            prev_loss = loss
    
    def fit(self, input_data=None, verbose=False):
        if input_data is None:
            return self._fit_no_data(verbose)
        else:
            return self._fit(input_data, verbose)

    def _fit_no_data(self,verbose):
        print("... training phase")
        print("... training variables")
        for key,param in self.comp_expl_graph.state_dict().items():
                print(key,param.shape)
        optimizer=optim.Adam(self.comp_expl_graph.parameters(), self.flags.sgd_learning_rate, weight_decay=0.01)
        print("... initialization")
        loss_cls = self.loss_cls()
        print("... building explanation graph")
        
        best_total_loss=None
        for step in range(self.flags.max_iterate):
            start_t = time.time()
            # train
            #print("... iteration")
            train_evaluator=TprismEvaluator()
            goal_inside = self.comp_expl_graph.forward()
            loss, output, label = loss_cls.call(self.graph, goal_inside, self.tensor_provider)
            optimizer.zero_grad()
            total_loss=torch.sum(torch.stack(loss),dim=0)
            total_loss.backward()
            optimizer.step()
            
            train_evaluator.eval(loss,output,label)
            #display_graph(output[j],'graph_pytorch')
            

            #train_acc=sklearn.metrics.accuracy_score(all_label,all_output)
            train_time = time.time() - start_t
            print(
                ": step",
                step,
                "loss:",
                np.sum(train_evaluator.total_loss)
            )
            print("train time:{0}".format(train_time) + "[sec]")
            if best_total_loss is None or train_evaluator.total_loss < best_total_loss:
                best_total_loss=train_evaluator.total_loss
                self.save(self.flags.model+".best.model")
        self.save(self.flags.model+".last.model")

    def _build_feed(self,ph_vars,dataset,idx):
        feed_dict = {
            ph: dataset[i, idx]
            for i, ph in enumerate(ph_vars)
            }
        return feed_dict
    
    def _set_batch_input(self, goal, train_idx, j, itr):
        batch_size = self.flags.sgd_minibatch_size
        ph_vars = goal["placeholders"]
        dataset = goal["dataset"]
        feed_dict = self._build_feed(ph_vars,dataset,train_idx[j][itr * batch_size : (itr + 1) * batch_size])
        #for k,v in feed_dict.items():
        #    print(k,v.shape)
        for embedding_generator in self.embedding_generators:
            if embedding_generator is not None:
                feed_dict = embedding_generator.build_feed(feed_dict,train_idx[j][itr * batch_size : (itr + 1) * batch_size])
        self.tensor_provider.set_input(feed_dict)
        
    def _fit(self, input_data, verbose):
        print("... training phase")
        goal_dataset = build_goal_dataset(input_data, self.tensor_provider)
        print("... training variables")
        for key,param in self.comp_expl_graph.state_dict().items():
                print(key,param.shape)
        optimizer=optim.Adam(self.comp_expl_graph.parameters(), self.flags.sgd_learning_rate, weight_decay=0.01)
        best_valid_loss = [None for _ in range(len(goal_dataset))]
        stopping_step = 0
        batch_size = self.flags.sgd_minibatch_size
        print("... splitting data")
        train_idx,valid_idx=split_goal_dataset(goal_dataset)
        loss_cls = self.loss_cls()
        print("... starting training")
        for step in range(self.flags.max_iterate):
            start_t = time.time()
            train_evaluator=TprismEvaluator(goal_dataset)
            valid_evaluator=TprismEvaluator(goal_dataset)
            for j, goal in enumerate(goal_dataset):
                np.random.shuffle(train_idx[j])
                # training update
                num_itr = len(train_idx[j]) // batch_size
                train_evaluator.init_batch()
                for itr in range(num_itr):
                    self._set_batch_input(goal,train_idx, j, itr)
                    goal_inside = self.comp_expl_graph.forward()
                    loss, output, label = loss_cls.call(self.graph, goal_inside, self.tensor_provider)
                    #display_graph(output[j],'graph_pytorch')
                    optimizer.zero_grad()
                    loss[j].backward()
                    optimizer.step()
                    train_evaluator.eval_batch(loss,output,label,j,num_itr)
                # validation
                num_itr = len(valid_idx[j]) // batch_size
                valid_evaluator.init_batch()
                for itr in range(num_itr):
                    self._set_batch_input(goal,valid_idx, j, itr)
                    goal_inside = self.comp_expl_graph.forward()
                    loss, output,label = loss_cls.call(self.graph, goal_inside, self.tensor_provider)
                    valid_evaluator.eval_batch(loss,output,label,j,num_itr)
                # conputing accuracies and printing them
                train_acc=sklearn.metrics.accuracy_score(train_evaluator.true_y,train_evaluator.pred_y)
                valid_acc=sklearn.metrics.accuracy_score(valid_evaluator.true_y,valid_evaluator.pred_y)
                print(
                    ": step",
                    step,
                    "train loss:",
                    train_evaluator.total_loss[j],
                    "valid loss:",
                    valid_evaluator.total_loss[j],
                    "train acc:",
                    train_acc,
                    "valid acc:",
                    valid_acc,
                )
                # checking validation loss for early stopping
                if best_valid_loss[j] is None or best_valid_loss[j] > valid_evaluator.total_loss[j]:
                    best_valid_loss[j] = valid_evaluator.total_loss[j]
                    stopping_step = 0
                    self.save(self.flags.model+".best.model")
                else:
                    stopping_step += 1
                    if stopping_step == self.flags.sgd_patience:
                        break
            train_time = time.time() - start_t
            print("train time:{0}".format(train_time) + "[sec]")
        self.save(self.flags.model+".last.model")
    
    def save(self, filename):
        torch.save(self.comp_expl_graph.state_dict(), filename)

    def load(self, filename):
        self.comp_expl_graph.load_state_dict(torch.load(filename))

    def pred(self, input_data=None, verbose=False):
        if input_data is not None:
            return self._pred(input_data,verbose)
        else:
            return self._pred_no_data(verbose)

    def _pred_no_data(self,verbose):
        print("... prediction")
        print("... loaded variables")
        for key,param in self.comp_expl_graph.state_dict().items():
                print(key,param.shape)
        print("... initialization")
        loss_cls = self.loss_cls()
        evaluator=TprismEvaluator()
        
        print("... predicting")
        start_t = time.time()
        goal_inside = self.comp_expl_graph.forward()
        loss, output, label = loss_cls.call(self.graph, goal_inside, self.tensor_provider)
        evaluator.eval(loss,output,label)
        #train_acc=sklearn.metrics.accuracy_score(all_label,all_output)
        print(
            "loss:",
            np.sum(evaluator.total_loss),
        )
        outputs=evaluator.output
        pred_y=evaluator.pred_y
        pred_time = time.time() - start_t
        print("prediction time:{0}".format(pred_time) + "[sec]")
        return pred_y,outputs
        
    def _pred(self, input_data, verbose=False):
        print("... prediction")
        goal_dataset = build_goal_dataset(input_data, self.tensor_provider)
        print("... loaded variables")
        for key,param in self.comp_expl_graph.state_dict().items():
                print(key,param.shape)
        print("... initialization")
        batch_size = self.flags.sgd_minibatch_size
        test_idx=get_goal_dataset(goal_dataset)
        loss_cls = self.loss_cls()
        evaluator=TprismEvaluator(goal_dataset)
        print("... predicting")
        start_t = time.time()
        outputs=[]
        pred_y=[]
        for j, goal in enumerate(goal_dataset):
            # valid
            num_itr = len(test_idx[j]) // batch_size
            evaluator.init_batch()
            for itr in range(num_itr):
                self._set_batch_input(goal,test_idx, j, itr)
                goal_inside = self.comp_expl_graph.forward()
                loss, output,label = loss_cls.call(self.graph, goal_inside, self.tensor_provider)
                evaluator.eval_batch(loss,output,label,j,num_itr)
            ##
            test_acc=sklearn.metrics.accuracy_score(evaluator.true_y,evaluator.pred_y)
            print(
                "loss:",
                evaluator.total_loss[j],
                "acc:",
                test_acc,
            )
            outputs.append(np.array(evaluator.output))
            pred_y.append(evaluator.pred_y)
        pred_time = time.time() - start_t
        print("train time:{0}".format(pred_time) + "[sec]")
        return pred_y,outputs

    def save_draw_graph(self, g, base_name):
        html = draw_graph.show_graph(g)
        fp = open(base_name + ".html", "w")
        fp.write(html)
        dot = draw_graph.tf_to_dot(g)
        dot.render(base_name)

def run_preparing(g, sess, args):
    input_data = expl_graph.load_input_data(args.data)
    graph, options = expl_graph.load_explanation_graph(args.expl_graph, args.flags)
    flags = Flags(args, options)
    flags.update()
    ##
    loss_loader = expl_graph.LossLoader()
    loss_loader.load_all("loss/")
    loss_cls = loss_loader.get_loss(flags.sgd_loss)
    ##
    tensor_provider = torch_expl_graph.TorchSwitchTensorProvider()
    embedding_generators = []
    if flags.embedding:
        eg = embed_gen.DatasetEmbeddingGenerator()
        eg.load(flags.embedding)
        embedding_generators.append(eg)
    if flags.const_embedding:
        eg = embed_gen.ConstEmbeddingGenerator()
        eg.load(flags.const_embedding)
        embedding_generators.append(eg)
    tensor_provider.build(
        graph,
        options,
        input_data,
        flags,
        load_embeddings=False,
        embedding_generators=embedding_generators,
    )


def run_training(args):
    if args.data is not None:
        input_data = expl_graph.load_input_data(args.data)
    else:
        input_data = None
    graph, options = expl_graph.load_explanation_graph(args.expl_graph, args.flags)
    flags = Flags(args, options)
    flags.update()
    ##
    loss_loader = expl_graph.LossLoader()
    loss_loader.load_all("loss/torch*")
    loss_cls = loss_loader.get_loss(flags.sgd_loss)
    ##
    print("... computational graph")
    model = TprismModel(flags,options,graph, loss_cls)
    model.build(input_data)
    start_t = time.time()
    print("... fit")
    if flags.cycle:
        model.solve(goal_dataset)
    elif input_data is not None:
        model.fit(input_data)
        model.pred(input_data)
    else:
        model.fit()
    
    train_time = time.time() - start_t
    print("total training time:{0}".format(train_time) + "[sec]")

def run_test(args):
    if args.data is not None:
        input_data = expl_graph.load_input_data(args.data)
    else:
        input_data = None
    graph, options = expl_graph.load_explanation_graph(args.expl_graph, args.flags)
    flags = Flags(args, options)
    flags.update()
    ##
    loss_loader = expl_graph.LossLoader()
    loss_loader.load_all("loss/torch*")
    loss_cls = loss_loader.get_loss(flags.sgd_loss)
    ##
    print("... computational graph")
    model = TprismModel(flags,options,graph, loss_cls)
    model.build(input_data)
    model.load(flags.model+".best.model")
    start_t = time.time()
    print("... prediction")
    if flags.cycle:
        model.solve(goal_dataset)
    elif input_data is not None:
        pred_y,out=model.pred(input_data)
    else:
        pred_y,out=model.pred()
    train_time = time.time() - start_t
    print("total training time:{0}".format(train_time) + "[sec]")
    print("... output")
    print("[SAVE]", flags.output)
    data={}
    j=0
    for root_obj in graph.root_list:
        for el in root_obj.roots:
            sid=el.sorted_id
            gg=graph.goals[sid].node
            name=gg.goal.name
            #name=to_string_goal(gg.goal)
            data[sid]={"name":name,"data":out[j]}
            j+=1
    fp = open('output.pkl','wb')
    pickle.dump(data,fp)

def main():
    # set random seed
    seed = 1234
    np.random.seed(seed)

    parser = argparse.ArgumentParser()
    parser.add_argument("mode", type=str, help="train/test")
    parser.add_argument("--config", type=str, default=None, help="config json file")

    parser.add_argument(
        "--data", type=str, default=None, nargs="+", help="[from prolog] data json file"
    )
    ## intermediate data
    parser.add_argument(
        "--intermediate_data_prefix",
        "-I",
        type=str,
        default="./",
        help="intermediate data",
    )
    parser.add_argument(
        "--expl_graph",
        type=str,
        default=None,
        help="[from prolog] explanation graph json file",
    )
    parser.add_argument(
        "--flags", type=str, default=None, help="[from prolog] flags json file"
    )
    parser.add_argument("--model", type=str, default=None, help="model file")
    parser.add_argument("--vocab", type=str, default=None, help="model file")
    ##
    parser.add_argument("--embedding", type=str, default=None, help="model file")
    parser.add_argument("--const_embedding", type=str, default=None, help="model file")
    parser.add_argument("--draw_graph", type=str, default=None, help="graph file")

    parser.add_argument(
        "--output", type=str, default="./output.pkl", help="output file"
    )

    parser.add_argument(
        "--gpu",
        type=str,
        default=None,
        help="constraint gpus (default: all) (e.g. --gpu 0,2)",
    )
    parser.add_argument(
        "--cpu", action="store_true", help="cpu mode (calcuration only with cpu)"
    )

    parser.add_argument("--no_verb", action="store_true", help="verb")

    parser.add_argument(
        "--sgd_minibatch_size", type=str, default=None, help="[prolog flag]"
    )
    parser.add_argument("--max_iterate", type=str, default=None, help="[prolog flag]")
    parser.add_argument("--epoch", type=str, default=None, help="[prolog flag]")
    parser.add_argument(
        "--sgd_learning_rate", type=float, default=0.01, help="[prolog flag]"
    )
    parser.add_argument(
        "--sgd_loss",
        type=str,
        default="base_loss",
        help="[prolog flag] nll/preference_pair",
    )
    parser.add_argument("--sgd_patience", type=int, default=3, help="[prolog flag] ")

    parser.add_argument("--cycle", action="store_true", help="cycle")

    args = parser.parse_args()
    # config
    if args.config is None:
        pass
    else:
        print("[LOAD] ", args.config)
        fp = open(args.config, "r")
        config.update(json.load(fp))

    # gpu/cpu
    if args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    elif args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    #
    if args.intermediate_data_prefix is not None:
        if args.expl_graph is None:
            args.expl_graph = args.intermediate_data_prefix + "expl.json"
        if args.flags is None:
            args.flags = args.intermediate_data_prefix + "flags.json"
        if args.model is None:
            args.model = args.intermediate_data_prefix + "model"
        if args.vocab is None:
            args.vocab = args.intermediate_data_prefix + "vocab.pkl"
    ##
    # setup
    seed = 1234
    torch.manual_seed(seed)
    # mode
    if args.mode == "train":
        run_training(args)
    if args.mode == "prepare":
        run_preparing(args)
    if args.mode == "test" or args.mode == "pred":
        run_test(args)
    elif args.mode == "cv":
        run_train_cv(args)
    if args.mode == "show":
        run_display(args)
if __name__ == "__main__":
    main()
