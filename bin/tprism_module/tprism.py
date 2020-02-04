#!/usr/bin/env python

import tensorflow as tf
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
import tprism_module.draw_graph as draw_graph
import tprism_module.embedding_generator as embed_gen

def to_string_goal(goal):
    """
    s=goal.name
    s+="("
    s+=",".join([str(arg) for arg in goal.args])
    s+=")"
    """
    s=goal.name
    s+=","
    s+=",".join([str(arg) for arg in goal.args])
    return s

class Flags(object):
    def __init__(self, args, options):
        self.internal_config = dict()
        self.args = args
        self.flags = {f.key: f.value for f in options.flags}

    def __getattr__(self, k):
        return (
            dict.get(self.internal_config, k)
            or getattr(self.args, k, None)
            or dict.get(self.flags, k)
        )

    def add(self, k, v):
        self.internal_config[k] = v

    def update(self):
        ##
        batch_size = 10
        if self.sgd_minibatch_size == "default":
            pass
        else:
            batch_size = int(self.sgd_minibatch_size)
        self.sgd_minibatch_size = batch_size
        self.add("sgd_minibatch_size", batch_size)
        ##
        if self.max_iterate == "default":
            self.max_iterate = 100
        else:
            self.max_iterate = int(self.max_iterate)
        ##
        self.sgd_learning_rate = float(self.sgd_learning_rate)


def optimize_solve(sess, goal_dataset, goal_inside, flags, embedding_generators):
    inside = []
    for goal in goal_inside:
        l1 = goal["inside"]
        inside.append(l1)

    init = tf.global_variables_initializer()
    sess.run(init)
    feed_dict = {}
    prev_loss = None
    for step in range(flags.max_iterate):
        feed_dict = {}
        for embedding_generator in embedding_generators:
            if embedding_generator is not None:
                feed_dict = embedding_generator.build_feed(feed_dict)
        out_inside = sess.run(inside, feed_dict=feed_dict)

        loss = 0
        for embedding_generator in embedding_generators:
            if embedding_generator is not None:
                loss = embedding_generator.update(out_inside)
        print("step", step, "loss:", loss)
        if loss < 1.0e-20:
            break
        if prev_loss is not None and not loss < prev_loss:
            pass
        prev_loss = loss
    # for a in out_inside:
    # 	print("~~~~~")
    # 	print(a)
    # 	print(np.sum(a>0))


def optimize_gd(sess, goal_dataset, loss, flags, embedding_generators):
    print("... training phase")
    total_loss = tf.reduce_sum(loss)
    optimizer = tf.train.AdamOptimizer(flags.sgd_learning_rate)
    train = optimizer.minimize(total_loss)

    print("... initialization")
    init = tf.global_variables_initializer()
    sess.run(init)
    print("... initializing generator")
    saver = tf.train.Saver()
    feed_dict = {}
    for embedding_generator in embedding_generators:
        if embedding_generator is not None:
            feed_dict = embedding_generator.build_feed(feed_dict)
    print("starting at", "loss:", sess.run(loss, feed_dict=feed_dict))
    for step in range(flags.max_iterate):
        feed_dict = {}
        for embedding_generator in embedding_generators:
            if embedding_generator is not None:
                feed_dict = embedding_generator.build_feed(feed_dict)
        sess.run(train, feed_dict=feed_dict)
        print("step", step, "loss:", sess.run(total_loss, feed_dict=feed_dict))
    print("[SAVE]", flags.model)
    saver.save(sess, flags.model)


def build_feed(ph_vars,dataset,idx):
    feed_dict = {
        ph: dataset[i, idx]
        for i, ph in enumerate(ph_vars)
        }
    return feed_dict
def split_goal_dataset(goal_dataset,valid_ratio = 0.1):
    train_idx=[]
    valid_idx=[]
    for j, goal in enumerate(goal_dataset):
        ph_vars = goal["placeholders"]
        all_num = goal["dataset"].shape[1]
        all_idx=np.array(list(range(all_num)))
        np.random.shuffle(all_idx)
        train_num = int(all_num - valid_ratio * all_num)
        train_idx.append(all_idx[:train_num])
        valid_idx.append(all_idx[train_num:])
    return train_idx,valid_idx

def optimize(sess, goal_dataset, loss, flags, embedding_generators):
    # optimizer = tf.train.GradientDescentOptimizer(flags.sgd_learning_rate)
    optimizer = tf.train.AdamOptimizer(flags.sgd_learning_rate)
    # train = [optimizer.minimize(l) for l in loss]
    train = []
    for l in loss:
        gradients, variables = zip(*optimizer.compute_gradients(l))
        gradients = [
            None if gradient is None else tf.clip_by_norm(gradient, 5.0)
            for gradient in gradients
        ]
        optimize = optimizer.apply_gradients(zip(gradients, variables))
        train.append(optimize)

    init = tf.global_variables_initializer()
    sess.run(init)
    saver = tf.train.Saver()
    best_valid_loss = [None for _ in range(len(goal_dataset))]
    stopping_step = 0
    batch_size = flags.sgd_minibatch_size
    train_idx,valid_idx=split_goal_dataset(goal_dataset)
    for step in range(flags.max_iterate):
        start_t = time.time()
        total_train_loss = [0.0 for _ in range(len(goal_dataset))]
        total_valid_loss = [0.0 for _ in range(len(goal_dataset))]
        for j, goal in enumerate(goal_dataset):
            # train
            ph_vars = goal["placeholders"]
            np.random.shuffle(train_idx[j])
            num_itr = len(train_idx[j]) // batch_size
            if not flags.no_verb:
                progbar = tf.keras.utils.Progbar(num_itr)
            ## one epoch
            for itr in range(num_itr):
                feed_dict = build_feed(ph_vars,goal["dataset"],train_idx[j][itr * batch_size : (itr + 1) * batch_size])
                for embedding_generator in embedding_generators:
                    if embedding_generator is not None:
                        feed_dict = embedding_generator.build_feed(feed_dict,train_idx[j][itr * batch_size : (itr + 1) * batch_size])
                batch_loss, _ = sess.run([loss[j], train[j]], feed_dict=feed_dict)
                if not flags.no_verb:
                    bl = np.mean(batch_loss)
                    progbar.update(itr, values=[("loss", bl)])
                total_train_loss[j] += np.mean(batch_loss) / num_itr
            # valid
            num_itr = len(valid_idx[j]) // batch_size
            for itr in range(num_itr):
                feed_dict = build_feed(ph_vars,goal["dataset"],valid_idx[j][itr * batch_size : (itr + 1) * batch_size])
                for embedding_generator in embedding_generators:
                    if embedding_generator is not None:
                        feed_dict = embedding_generator.build_feed(feed_dict,valid_idx[j][itr * batch_size : (itr + 1) * batch_size])
                batch_loss, _ = sess.run([loss[j], train[j]], feed_dict=feed_dict)
                total_valid_loss[j] += np.mean(batch_loss) / num_itr
            #
            print(
                ": step",
                step,
                "train loss:",
                total_train_loss[j],
                "valid loss:",
                total_valid_loss[j],
            )
            #
            if best_valid_loss[j] is None or best_valid_loss[j] > total_valid_loss[j]:
                best_valid_loss[j] = total_valid_loss[j]
                stopping_step = 0
            else:
                stopping_step += 1
                if stopping_step == flags.sgd_patience:
                    print("[SAVE]", flags.model)
                    saver.save(sess, flags.model)
                    return
        train_time = time.time() - start_t
        print("train time:{0}".format(train_time) + "[sec]")
    print("[SAVE]", flags.model)
    saver.save(sess, flags.model)

def pred(sess, goal_dataset, loss, output, flags, embedding_generators):
    total_goal_inside=None
    start_t = time.time()
    if goal_dataset is not None:
        ### dataset is given (minibatch)
        batch_size = flags.sgd_minibatch_size
        total_loss = [[] for _ in range(len(goal_dataset))]
        total_output = [[] for _ in range(len(goal_dataset))]
        for j, goal in enumerate(goal_dataset):
            ph_vars = goal["placeholders"]
            dataset = goal["dataset"]
            num = dataset.shape[1]
            num_itr = (num + batch_size - 1) // batch_size
            if not flags.no_verb:
                progbar = tf.keras.utils.Progbar(num_itr)
            idx = list(range(num))
            for itr in range(num_itr):
                temp_idx = idx[itr * batch_size : (itr + 1) * batch_size]
                if len(temp_idx) < batch_size:
                    padding_idx = np.zeros((batch_size,), dtype=np.int32)
                    padding_idx[: len(temp_idx)] = temp_idx
                    temp_idx=padding_idx
                feed_dict = build_feed(ph_vars,dataset,temp_idx)
                for embedding_generator in embedding_generators:
                    if embedding_generator is not None:
                        feed_dict = embedding_generator.build_feed(feed_dict,temp_idx)
                batch_loss, batch_output = sess.run(
                    [loss[j], output[j]], feed_dict=feed_dict
                )
                if not flags.no_verb:
                    progbar.update(itr)
                # print(batch_output.shape)
                # batch_output=np.transpose(batch_output)
                total_loss[j].extend(batch_loss[: len(temp_idx)])
                total_output[j].extend(batch_output[: len(temp_idx)])
            print("loss:", np.mean(total_loss[j]))
            print("output:", np.array(total_output[j]).shape)
    else:
        feed_dict = {}
        total_loss = []
        total_output = []
        if embedding_generator:
            feed_dict = embedding_generator.build_feed(feed_dict)
        for j in range(len(loss)):
            j_loss, j_output = sess.run([loss[j], output[j]], feed_dict=feed_dict)
            total_loss.append(j_loss)
            total_output.append(j_output)
            ###
        print("loss:", np.mean(total_loss))
        print("output:", np.array(total_output).shape)
        total_goal_inside = []
        for g in goal_inside:
            g_inside = sess.run([g['inside']], feed_dict=feed_dict)
            total_goal_inside.append(g_inside[0])
    test_time = time.time() - start_t
    print("test time:{0}".format(test_time) + "[sec]")
    return total_output, total_loss, total_goal_inside
    


def save_draw_graph(g, base_name):
    html = draw_graph.show_graph(g)
    fp = open(base_name + ".html", "w")
    fp.write(html)
    dot = draw_graph.tf_to_dot(g)
    dot.render(base_name)


#
#goal_dataset["placeholders"] => ph_vars
#goal_dataset["dataset"]: dataset
# dataset contains indeces: values in the given dataset is coverted into index
def build_goal_dataset(input_data, tensor_provider):
    goal_dataset = []

    def to_index(value, ph_name):
        return tensor_provider.convert_value_to_index(value, ph_name)

    to_index_func = np.vectorize(to_index)
    for d in input_data:
        ph_names = d["placeholders"]
        # TODO: multiple with different placeholders
        ph_vars = [tensor_provider.ph_var[ph_name] for ph_name in ph_names]
        dataset = [None for _ in ph_names]
        goal_data = {"placeholders": ph_vars, "dataset": dataset}
        goal_dataset.append(goal_data)
        for i, ph_name in enumerate(ph_names):
            rec = d["records"]
            if tensor_provider.is_convertable_value(ph_name):
                dataset[i] = to_index_func(rec[:, i], ph_name)
            else:  # goal placeholder
                dataset[i] = rec[:, i]
                print("[WARN] no conversion from values to indices")
                print("goal_placeholder?")
                print(rec.shape)
                print(ph_name)
            print("*")
    for obj in goal_dataset:
        obj["dataset"] = np.array(obj["dataset"])
    return goal_dataset


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
    tensor_provider = expl_graph.SwitchTensorProvider()
    embedding_generators = []
    if flags.embedding:
        eg = embed_gen.DatasetEmbeddingGenerator()
        eg.load(flags.embedding)
        embedding_generators.append(eg)
    if flags.const_embedding:
        eg = embed_gen.ConstEmbeddingGenerator()
        eg.load(flags.const_embedding)
        embedding_generators.append(eg)
    tensor_embedding = tensor_provider.build(
        graph,
        options,
        input_data,
        flags,
        load_embeddings=False,
        embedding_generators=embedding_generators,
    )


def run_training(g, sess, args):
    if args.data is not None:
        input_data = expl_graph.load_input_data(args.data)
    else:
        input_data = None
    graph, options = expl_graph.load_explanation_graph(args.expl_graph, args.flags)
    flags = Flags(args, options)
    flags.update()
    ##
    loss_loader = expl_graph.LossLoader()
    loss_loader.load_all("loss/")
    loss_cls = loss_loader.get_loss(flags.sgd_loss)
    ##
    tensor_provider = expl_graph.SwitchTensorProvider()
    embedding_generators = []
    if flags.embedding:
        eg = embed_gen.DatasetEmbeddingGenerator()
        eg.load(flags.embedding)
        embedding_generators.append(eg)
    if flags.const_embedding:
        eg = embed_gen.ConstEmbeddingGenerator()
        eg.load(flags.const_embedding)
        embedding_generators.append(eg)
    cycle_embedding_generator = None
    if flags.cycle:
        cycle_embedding_generator = embed_gen.CycleEmbeddingGenerator()
        cycle_embedding_generator.load(options)
        embedding_generators.append(cycle_embedding_generator)

    tensor_embedding = tensor_provider.build(
        graph,
        options,
        input_data,
        flags,
        load_embeddings=False,
        embedding_generators=embedding_generators,
    )
    comp_expl_graph=expl_graph.ComputationalExplGraph()
    goal_inside = comp_expl_graph.build_explanation_graph(
        graph, tensor_provider, cycle_embedding_generator
    )
    if input_data is not None:
        goal_dataset = build_goal_dataset(input_data, tensor_provider)
    else:
        goal_dataset = None

    if flags.draw_graph:
        save_draw_graph(g, "test")

    loss, output = loss_cls().call(graph, goal_inside, tensor_provider)
    if loss:
        with tf.name_scope("summary"):
            tf.summary.scalar("loss", loss)
            merged = tf.summary.merge_all()
            writer = tf.summary.FileWriter("./tf_logs", sess.graph)
    ##
    print("traing start")
    vars_to_train = tf.trainable_variables()
    for var in vars_to_train:
        print("train var:", var.name, var.shape)
    ##
    start_t = time.time()
    if flags.cycle:
        optimize_solve(
            sess,
            goal_dataset,
            goal_inside,
            flags,
            embedding_generators,
        )
    elif goal_dataset is not None:
        optimize(
            sess,
            goal_dataset,
            loss,
            flags,
            embedding_generators,
        )
    else:
        optimize_gd(
            sess,
            goal_dataset,
            loss,
            flags,
            embedding_generators,
        )
    train_time = time.time() - start_t
    print("traing time:{0}".format(train_time) + "[sec]")


def run_test(g, sess, args):
    if args.data is not None:
        input_data = expl_graph.load_input_data(args.data)
    else:
        input_data = None
    graph, options = expl_graph.load_explanation_graph(args.expl_graph, args.flags)
    flags = Flags(args, options)
    flags.update()
    ##
    loss_loader = expl_graph.LossLoader()
    loss_loader.load_all("loss/")
    loss_cls = loss_loader.get_loss(flags.sgd_loss)
    ##
    tensor_provider = expl_graph.SwitchTensorProvider()
    embedding_generators = []
    if flags.embedding:
        eg = embed_gen.DatasetEmbeddingGenerator()
        eg.load(flags.embedding, key="test")
        embedding_generators.append(eg)
    if flags.const_embedding:
        eg = embed_gen.ConstEmbeddingGenerator()
        eg.load(flags.const_embedding, key="test")
        embedding_generators.append(eg)
    cycle_embedding_generator = None
    if flags.cycle:
        cycle_embedding_generator.load(options)
        embedding_generators.append(cycle_embedding_generator)
    tensor_embedding = tensor_provider.build(
        graph,
        options,
        input_data,
        flags,
        load_embeddings=True,
        embedding_generators=embedding_generators,
    )
    comp_expl_graph=expl_graph.ComputationalExplGraph()
    goal_inside = comp_expl_graph.build_explanation_graph(
        graph, tensor_provider, cycle_embedding_generator
    )
    if input_data is not None:
        goal_dataset = build_goal_dataset(input_data, tensor_provider)
    else:
        goal_dataset = None
    if flags.draw_graph:
        save_draw_graph(g, "test")
    loss, output = loss_cls().call(graph, goal_inside, tensor_provider)
    saver = tf.train.Saver()
    saver.restore(sess, flags.model)

    if flags.cycle:
        optimize_solve(
            sess,
            goal_dataset,
            goal_inside,
            flags,
            embedding_generators,
        )
    total_output,total_loss,total_goal_inside=pred(sess, goal_dataset, loss, output, flags, embedding_generators)
    print("[SAVE]", flags.output)
    np.save(flags.output, total_output)
    if total_goal_inside is not None:
        data={}
        for g_info,g in zip(graph.goals, total_goal_inside):
            gg=g_info.node.goal
            name=to_string_goal(gg)
            data[g_info.node.id]={"name":name,"data":g}
        fp = open('output.pkl','wb')
        pickle.dump(data,fp)


    """
    ###
    print("[SAVE]", flags.output)
    np.save(flags.output, total_output)
    data={}
    for g_info,g in zip(graph.goals, total_goal_inside):
        gg=g_info.node.goal
        name=to_string_goal(gg)
        data[g_info.node.id]={"name":name,"data":g}
    fp = open('output.pkl','wb')
    pickle.dump(data,fp)
    """

def run_display(args):
    #
    if args.expl_graph is None:
        args.expl_graph = args.intermediate_data_prefix + "expl.json"
    if args.flags is None:
        args.flags = args.intermediate_data_prefix + "flags.json"
    if args.model is None:
        args.model = args.intermediate_data_prefix + "model.ckpt"
    if args.vocab is None:
        args.vocab = args.intermediate_data_prefix + "vocab.pkl"
    fp = open(args.vocab, "rb")
    obj = pickle.load(fp)
    print(obj.vocab_group)
    ##

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
        "--output", type=str, default="./output.npy", help="output file"
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
            args.model = args.intermediate_data_prefix + "model.ckpt"
        if args.vocab is None:
            args.vocab = args.intermediate_data_prefix + "vocab.pkl"
    ##
    # setup
    g = tf.Graph()
    with g.as_default():
        seed = 1234
        tf.set_random_seed(seed)
        with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
            # mode
            if args.mode == "train":
                run_training(g, sess, args)
            if args.mode == "prepare":
                run_preparing(g, sess, args)
            if args.mode == "test":
                run_test(g, sess, args)
            elif args.mode == "cv":
                run_train_cv(g, sess, args)
            if args.mode == "show":
                run_display(args)
if __name__ == "__main__":
    main()
