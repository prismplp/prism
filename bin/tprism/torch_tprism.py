#!/usr/bin/env python
"""
This module contains pytorch T-PRISM main 
"""


import torch
import torch.optim as optim
import torch.nn.functional as F
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

import tprism.expl_pb2 as expl_pb2
import tprism.expl_graph as expl_graph
import tprism.torch_expl_graph as torch_expl_graph
import tprism.torch_embedding_generator as embed_gen
from tprism.util import (
    to_string_goal,
    Flags,
    build_goal_dataset,
    split_goal_dataset,
    get_goal_dataset,
)
from tprism.loader import (
    load_input_data,
    load_explanation_graph,
    LossLoader,
)

from tprism.torch_util import draw_graph
import re
import numpy as np
import sklearn.metrics

""" Main module for command line interface

This is called by tprism command (pytorch based tprism)
"""

class TprismEvaluator:
    """Evaluator for pytorch system

    """
    def __init__(self, goal_dataset=None):
        if goal_dataset is not None:
            self.n_goals = len(goal_dataset)
        else:
            self.n_goals = 1
        self.loss_history = [[] for _ in range(self.n_goals)]
        self.loss_dict_history = [[] for _ in range(self.n_goals)]
        self.label = [[] for _ in range(self.n_goals)]
        self.output = [[] for _ in range(self.n_goals)]

    def start_epoch(self):
        self.running_loss = [0.0 for _ in range(self.n_goals)]
        self.running_loss_dict = [{} for _ in range(self.n_goals)]
        self.running_count = [0 for _ in range(self.n_goals)]

    def update(self, loss, loss_dict, j):
        """
            This function is called the last of batch iteration
              loss,output, label: return values of forward
              j: goal index of iteration
              num_itr:: number of iterations
        """
        self.running_loss[j] += loss
        self.running_count[j] += 1
        for k, v in loss_dict.items():
            if k in self.running_loss_dict:
                self.running_loss_dict[j][k] += v
            else:
                self.running_loss_dict[j][k] = v

    def stop_epoch(self, j=0, mean_flag=True):
        if mean_flag:
            self.running_loss[j] /= self.running_count[j]
            for k in self.running_loss_dict[j].keys():
                self.running_loss_dict[j][k] /= self.running_count[j]
        self.loss_history[j].append(self.running_loss[j])
        self.loss_dict_history[j].append(self.running_loss_dict[j])

    def get_dict(self, prefix="train"):
        result = {}
        for j in range(self.n_goals):
            key = "{:s}-loss".format(prefix)
            val = self.running_loss[j]
            result[key] = float(val)
            for k, v in self.running_loss_dict[j].items():
                if k[0] != "*":
                    m = "{:s}-{:s}-loss".format(prefix, k)
                else:
                    m = "*{:s}-{:s}".format(prefix, k[1:])
                result[m] = float(v)
            return result

    def get_msg(self, prefix="train"):
        msg = []
        for key, val in self.get_dict(prefix=prefix).items():
            m = "{:s}: {:.3f}".format(key, val)
            msg.append(m)
        return "  ".join(msg)

    def get_loss(self):
        return self.running_loss

    def update_data(self, output, label, j):
        if type(output[j]) == list:  # preference
            _o = [o.detach().numpy() for o in output[j]]
        else:
            _o = output[j].detach().numpy()
        self.output[j].extend(_o)
        if label is not None:
            _l = label[j].detach().numpy()
            self.label[j].extend(_l)


class TprismModel:
    """T-PRISM model for pytorch

    """
 
    def __init__(self, flags, options, graph, loss_cls):
        self.graph = graph
        self.flags = flags
        self.options = options
        self.loss_cls = loss_cls

    def build(self, input_data, load_embeddings, embedding_key):
        self._build_embedding(embedding_key)
        self._set_data(input_data, load_embeddings)
        self._build_explanation_graph()

    def _build_embedding(self, embedding_key):
        embedding_generators = []
        if self.flags.embedding:
            eg = embed_gen.DatasetEmbeddingGenerator()
            eg.load(self.flags.embedding, key=embedding_key)
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

    def _set_data(self, input_data, load_embeddings):
        self.tensor_provider = torch_expl_graph.TorchSwitchTensorProvider()
        self.tensor_provider.build(
            self.graph,
            self.options,
            input_data,
            self.flags,
            load_embeddings=load_embeddings,
            embedding_generators=self.embedding_generators,
        )

    def _build_explanation_graph(self):
        self.comp_expl_graph = torch_expl_graph.TorchComputationalExplGraph(
            self.graph, self.tensor_provider, self.cycle_embedding_generator
        )

    def solve(self, input_data=None):
        if input_data is None:
            self._solve_no_data()

    def _solve_no_data(self):
        print("... training phase")
        print("... training variables")
        for key, param in self.comp_expl_graph.state_dict().items():
            print(key, param.shape)
        print("... building explanation graph")
        prev_loss = None
        for step in range(self.flags.max_iterate):
            feed_dict = {}
            for embedding_generator in self.embedding_generators:
                if embedding_generator is not None:
                    feed_dict = embedding_generator.build_feed(feed_dict, None)
            self.tensor_provider.set_input(feed_dict)
            # out_inside = self.sess.run(inside, feed_dict=feed_dict)
            goal_inside, loss_list = self.comp_expl_graph.forward(verbose=True)
            inside = []
            for goal in goal_inside:
                l1 = goal["inside"]
                inside.append(l1)
            loss = 0
            for embedding_generator in self.embedding_generators:
                if embedding_generator is not None:
                    loss = embedding_generator.update(inside)
            print("step", step, "loss:", loss)
            if loss < 1.0e-20:
                break
            if prev_loss is not None and not loss < prev_loss:
                pass
            prev_loss = loss
        ##

    def fit(self, input_data=None, verbose=False):
        if input_data is None:
            return self._fit_no_data(verbose)
        else:
            return self._fit(input_data, verbose)

    def _fit_no_data(self, verbose):
        print("... training phase")
        print("... training variables")
        for key, param in self.comp_expl_graph.state_dict().items():
            print(key, param.shape)
        optimizer = optim.Adam(
            self.comp_expl_graph.parameters(),
            self.flags.sgd_learning_rate,
            weight_decay=1.0e-10,
        )
        print("... initialization")
        loss_cls = self.loss_cls()
        print("... building explanation graph")

        best_total_loss = None
        train_evaluator = TprismEvaluator()
        for epoch in range(self.flags.max_iterate):
            start_t = time.time()
            # train
            feed_dict = {}
            for embedding_generator in self.embedding_generators:
                if embedding_generator is not None:
                    feed_dict = embedding_generator.build_feed(feed_dict, None)
            self.tensor_provider.set_input(feed_dict)
            # print("... iteration")
            goal_inside, loss_list = self.comp_expl_graph.forward()
            loss, output, label = loss_cls.call(
                self.graph, goal_inside, self.tensor_provider
            )
            optimizer.zero_grad()
            total_loss = torch.sum(loss, dim=0)
            total_loss.backward()
            optimizer.step()
            metrics = loss_cls.metrics(output, label)
            metrics.update(loss_list)
            train_evaluator.start_epoch()
            train_evaluator.update(total_loss, metrics, 0)
            train_evaluator.stop_epoch()
            # display_graph(output[j],'graph_pytorch')

            # train_acc=sklearn.metrics.accuracy_score(all_label,all_output)
            train_time = time.time() - start_t
            print("[{:4d}] ".format(epoch + 1), train_evaluator.get_msg("train"))
            print("train time:{0}".format(train_time) + "[sec]")
            if (
                best_total_loss is None
                or train_evaluator.running_loss[0] < best_total_loss
            ):
                best_total_loss = train_evaluator.running_loss[0]
                self.save(self.flags.model + ".best.model")
        self.save(self.flags.model + ".last.model")

    def _build_feed(self, ph_vars, dataset, idx, verbose=False):
        if verbose:
            for i, ph in enumerate(ph_vars):
                print("[INFO feed]", ph, ph.name)
                print("[INFO feed]", dataset[i, idx].shape)
        feed_dict = {ph: torch.tensor(dataset[i, idx]) for i, ph in enumerate(ph_vars)}
        return feed_dict

    def _set_batch_input(self, goal, train_idx, j, itr):
        batch_size = self.flags.sgd_minibatch_size
        ph_vars = goal["placeholders"]
        dataset = goal["dataset"]
        batch_idx = train_idx[j][itr * batch_size : (itr + 1) * batch_size]
        feed_dict = self._build_feed(ph_vars, dataset, batch_idx)
        # for k,v in feed_dict.items():
        #    print(k,v.shape)
        for embedding_generator in self.embedding_generators:
            if embedding_generator is not None:
                feed_dict = embedding_generator.build_feed(feed_dict, batch_idx)
        self.tensor_provider.set_input(feed_dict)

    def _fit(self, input_data, verbose):
        print("... training phase")
        goal_dataset = build_goal_dataset(input_data, self.tensor_provider)
        print("... training variables")
        for key, param in self.comp_expl_graph.state_dict().items():
            print(key, param.shape)
        optimizer = optim.Adam(
            self.comp_expl_graph.parameters(),
            self.flags.sgd_learning_rate,
            weight_decay=1.0e-10,
        )
        best_valid_loss = [None for _ in range(len(goal_dataset))]
        patient_count = 0
        batch_size = self.flags.sgd_minibatch_size
        print("... splitting data")
        train_idx, valid_idx = split_goal_dataset(goal_dataset)
        loss_cls = self.loss_cls()
        print("... starting training")
        for epoch in range(self.flags.max_iterate):
            start_t = time.time()
            train_evaluator = TprismEvaluator(goal_dataset)
            valid_evaluator = TprismEvaluator(goal_dataset)
            for j, goal in enumerate(goal_dataset):
                if verbose:
                    print(goal)
                np.random.shuffle(train_idx[j])
                # training update
                num_itr = len(train_idx[j]) // batch_size
                train_evaluator.start_epoch()
                for itr in range(num_itr):
                    self._set_batch_input(goal, train_idx, j, itr)
                    goal_inside, loss_list = self.comp_expl_graph.forward(
                        verbose=verbose
                    )
                    loss, output, label = loss_cls.call(
                        self.graph, goal_inside, self.tensor_provider
                    )
                    if label is not None:
                        metrics = loss_cls.metrics(
                            output[j].detach().numpy(), label[j].detach().numpy()
                        )
                    else:
                        metrics = loss_cls.metrics(output[j].detach().numpy(), None)
                    loss_list.update(metrics)
                    # display_graph(output[j],'graph_pytorch')
                    optimizer.zero_grad()
                    loss[j].backward()
                    optimizer.step()
                    train_evaluator.update(loss[j], loss_list, j)
                # validation
                num_itr = len(valid_idx[j]) // batch_size
                valid_evaluator.start_epoch()
                for itr in range(num_itr):
                    self._set_batch_input(goal, valid_idx, j, itr)
                    goal_inside, loss_list = self.comp_expl_graph.forward()
                    loss, output, label = loss_cls.call(
                        self.graph, goal_inside, self.tensor_provider
                    )
                    if label is not None:
                        metrics = loss_cls.metrics(
                            output[j].detach().numpy(), label[j].detach().numpy()
                        )
                    else:
                        metrics = loss_cls.metrics(output[j].detach().numpy(), None)
                    loss_list.update(metrics)
                    valid_evaluator.update(loss[j], loss_list, j)
                # checking validation loss for early stopping
                if (
                    best_valid_loss[j] is None
                    or best_valid_loss[j] > valid_evaluator.running_loss[j]
                ):
                    best_valid_loss[j] = valid_evaluator.running_loss[j]
                    patient_count = 0
                    check_point_flag = True
                    self.save(self.flags.model + ".best.model")
                else:
                    patient_count += 1
                ckpt_msg = "*" if check_point_flag else ""
                print(
                    "[{:4d}] ".format(epoch + 1),
                    train_evaluator.get_msg("train"),
                    valid_evaluator.get_msg("valid"),
                    "({:2d})".format(patient_count),
                    ckpt_msg,
                )
                if patient_count == self.flags.sgd_patience:
                    break
            train_time = time.time() - start_t
            print("train time:{0}".format(train_time) + "[sec]")
        self.save(self.flags.model + ".last.model")

    def save(self, filename):
        torch.save(self.comp_expl_graph.state_dict(), filename)

    def load(self, filename):
        self.comp_expl_graph.load_state_dict(torch.load(filename))

    def pred(self, input_data=None, verbose=False):
        if input_data is not None:
            return self._pred(input_data, verbose)
        else:
            return self._pred_no_data(verbose)

    def _pred_no_data(self, verbose):
        print("... prediction")
        print("... loaded variables")
        for key, param in self.comp_expl_graph.state_dict().items():
            print(key, param.shape)
        print("... initialization")
        loss_cls = self.loss_cls()
        evaluator = TprismEvaluator()
        ###
        feed_dict = {}
        for embedding_generator in self.embedding_generators:
            if embedding_generator is not None:
                feed_dict = embedding_generator.build_feed(feed_dict, None)
        self.tensor_provider.set_input(feed_dict)
        print("... predicting")
        start_t = time.time()
        goal_inside, loss_list = self.comp_expl_graph.forward()
        loss, output, label = loss_cls.call(
            self.graph, goal_inside, self.tensor_provider
        )
        metrics=loss_cls.metrics(output, label)
        # train_acc=sklearn.metrics.accuracy_score(all_label,all_output)
        #print("loss:", np.sum(evaluator.get_loss())
        print("loss:", torch.sum(loss))
        print("metrics:", metrics)
        pred_time = time.time() - start_t
        print("prediction time:{0}".format(pred_time) + "[sec]")
        if label is None:
            label=label.detach().numpy()
        output=output.detach().numpy()
        return label, output

    def export_computational_graph(self, input_data, verbose=False):
        print("... prediction")
        goal_dataset = build_goal_dataset(input_data, self.tensor_provider)
        print("... loaded variables")
        for key, param in self.comp_expl_graph.state_dict().items():
            print(key, param.shape)
        print("... initialization")
        batch_size = self.flags.sgd_minibatch_size
        test_idx = get_goal_dataset(goal_dataset)
        loss_cls = self.loss_cls()
        evaluator = TprismEvaluator(goal_dataset)
        print("... predicting")
        start_t = time.time()
        outputs = []
        labels = []
        for j, goal in enumerate(goal_dataset):
            # valid
            num_itr = len(test_idx[j]) // batch_size
            evaluator.start_epoch()
            for itr in range(num_itr):
                self._set_batch_input(goal, test_idx, j, itr)
                goal_inside, loss_list = self.comp_expl_graph.forward(dryrun=True)
                for g in goal_inside:
                    print(g)
                    for path in g["inside"]:
                        print("  ", path)

    def _pred(self, input_data, verbose=False):
        print("... prediction")
        goal_dataset = build_goal_dataset(input_data, self.tensor_provider)
        print("... loaded variables")
        for key, param in self.comp_expl_graph.state_dict().items():
            print(key, param.shape)
        print("... initialization")
        batch_size = self.flags.sgd_minibatch_size
        test_idx = get_goal_dataset(goal_dataset)
        loss_cls = self.loss_cls()
        evaluator = TprismEvaluator(goal_dataset)
        print("... predicting")
        start_t = time.time()
        outputs = []
        labels = []
        for j, goal in enumerate(goal_dataset):
            # valid
            num_itr = len(test_idx[j]) // batch_size
            evaluator.start_epoch()
            for itr in range(num_itr):
                self._set_batch_input(goal, test_idx, j, itr)
                goal_inside, loss_list = self.comp_expl_graph.forward()
                loss, output, label = loss_cls.call(
                    self.graph, goal_inside, self.tensor_provider
                )
                evaluator.update(loss[j], loss_list, j)
                _o = output[j].detach().numpy()
                _l = label[j].detach().numpy() if label is not None else None
                metrics = loss_cls.metrics(_o, _l)
                evaluator.update_data(output, label, j)
            ##
            print(evaluator.get_msg("test"))
            # print(evaluator.output[j],evaluator.label[j])
            metrics = loss_cls.metrics(
                np.array(evaluator.output[j]), np.array(evaluator.label[j])
            )
            outputs.append(np.array(evaluator.output[j]))
            labels.append(evaluator.label[j])
        pred_time = time.time() - start_t
        print("test time:{0}".format(pred_time) + "[sec]")
        return labels, outputs

    def save_draw_graph(self, g, base_name):
        html = draw_graph.show_graph(g)
        fp = open(base_name + ".html", "w")
        fp.write(html)
        dot = draw_graph.tf_to_dot(g)
        dot.render(base_name)


def run_preparing(g, sess, args):
    input_data = load_input_data(args.dataset)
    graph, options = load_explanation_graph(args.expl_graph, args.flags)
    flags = Flags(args, options)
    flags.update()
    ##
    loss_loader = LossLoader()
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
    if args.dataset is not None:
        input_data = load_input_data(args.dataset)
    else:
        input_data = None
    graph, options = load_explanation_graph(args.expl_graph, args.flags)
    flags = Flags(args, options)
    flags.update()
    ##
    loss_loader = LossLoader()
    loss_loader.load_all("loss/torch*")
    loss_cls = loss_loader.get_loss(flags.sgd_loss)
    ##
    print("... computational graph")
    model = TprismModel(flags, options, graph, loss_cls)
    model.build(input_data, load_embeddings=False, embedding_key="train")
    start_t = time.time()
    if flags.cycle:
        print("... fit with cycle")
        model.solve()
    elif input_data is not None:
        print("... fit with input data")
        model.export_computational_graph(input_data)
        print("=========")
        model.fit(input_data, verbose=False)
        model.pred(input_data)
    else:
        print("... fit without input")
        model.fit()

    train_time = time.time() - start_t
    print("total training time:{0}".format(train_time) + "[sec]")


def run_test(args):
    if args.dataset is not None:
        input_data = load_input_data(args.dataset)
    else:
        input_data = None
    graph, options = load_explanation_graph(args.expl_graph, args.flags)
    flags = Flags(args, options)
    flags.update()
    ##
    loss_loader = LossLoader()
    loss_loader.load_all("loss/torch*")
    loss_cls = loss_loader.get_loss(flags.sgd_loss)
    ##
    print("... computational graph")
    model = TprismModel(flags, options, graph, loss_cls)
    model.build(input_data, load_embeddings=False, embedding_key="test")
    model.load(flags.model + ".best.model")
    start_t = time.time()
    print("... prediction")
    if flags.cycle:
        model.solve(goal_dataset)
    elif input_data is not None:
        pred_y, out = model.pred(input_data)
    else:
        pred_y, out = model.pred()
    train_time = time.time() - start_t
    print("total training time:{0}".format(train_time) + "[sec]")
    print("... output")
    np.save(flags.output,out)
    print("[SAVE]", flags.output)
    data = {}
    for j, root_obj in enumerate(graph.root_list):
        multi_root = False
        if len(root_obj.roots) > 1:  # preference
            multi_root = True
        for i, el in enumerate(root_obj.roots):
            sid = el.sorted_id
            gg = graph.goals[sid].node
            name = gg.goal.name
            # name=to_string_goal(gg.goal)
            if multi_root:
                data[sid] = {"name": name, "data": out[j][i]}
            else:
                data[sid] = {"name": name, "data": out[j]}

    fp = open("output.pkl", "wb")
    pickle.dump(data, fp)


def main():
    # set random seed
    seed = 1234
    np.random.seed(seed)

    parser = argparse.ArgumentParser()
    parser.add_argument("mode", type=str, help="train/test")
    parser.add_argument("--config", type=str, default=None, help="config json file")

    parser.add_argument(
            "--data", type=str, default=None, nargs="+", help="[from prolog] data json file (deprecated: use --dataset)"
    )
    parser.add_argument(
            "--dataset", type=str, default=None, nargs="+", help="[from prolog] dataset file"
    )
    ## intermediate data
    parser.add_argument(
        "--intermediate_data_prefix",
        type=str,
        default=None,
        help="intermediate data (deprecated: use --input)",
    )
    parser.add_argument(
        "--input",
        "-I",
        type=str,
        default=None,
        help="input intermediate data",
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
    parser.add_argument("--vocab", type=str, default=None, help="vocabrary file")
    ##
    parser.add_argument("--embedding", type=str, default=None, help="embedding file")
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

    # deprecated
    if args.data is not None:
        args.dataset=args.data
    if args.intermediate_data_prefix is not None:
        if args.expl_graph is None:
            args.expl_graph = args.intermediate_data_prefix + "expl.json"
        if args.flags is None:
            args.flags = args.intermediate_data_prefix + "flags.json"
        if args.model is None:
            args.model = args.intermediate_data_prefix + "model"
        if args.vocab is None:
            args.vocab = args.intermediate_data_prefix + "vocab.pkl"
    elif args.input is not None:
        # setting default input data
        sep="."
        if os.path.isdir(args.input):
            sep=""
        if args.expl_graph is None:
            args.expl_graph = args.input + sep + "expl.json"
        if args.flags is None:
            args.flags = args.input + sep + "flags.json"
        if args.model is None:
            args.model = args.input + sep + "model"
        if args.vocab is None:
            args.vocab = args.input + sep + "vocab.pkl"
    #
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
