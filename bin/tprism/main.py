#!/usr/bin/env python
"""T-PRISM command line interface.

This module provides the command-line entrypoint called by the tprism
command: argument parsing and the train/prepare/test mode runners.
The model itself lives in `tprism.model`.
"""


from typing import List, Optional, Tuple, cast
import argparse
import json
import logging
import os
import pickle
import time

import numpy as np
import torch

import tprism.embedding_generator as embed_gen
import tprism.expl_pb2 as expl_pb2
from tprism.expl_tensor import SwitchTensorProvider
from tprism.loader import (
    load_input_data,
    load_explanation_graph,
    LossLoader,
)
from tprism.loss.base import BaseLoss
from tprism.model import TprismModel
from tprism.util import (
    DEBUG_CATEGORIES,
    Flags,
    InputData,
    TensorInfoMapper,
    setup_logging,
)

# fall back to a package-level name when executed as `python -m tprism.main`
# so that messages are handled by the "tprism" logger configured in setup_logging
logger = logging.getLogger(__name__ if __name__ != "__main__" else "tprism.main")


def _setup(
    args: argparse.Namespace,
) -> Tuple[Optional[List[InputData]], "expl_pb2.ExplGraph", TensorInfoMapper, Flags, Optional[BaseLoss]]:
    """Load input data, the explanation graph, and the loss object.

    This is the common preparation shared by all CLI modes.

    Args:
        args: Parsed CLI arguments.

    Returns:
        A tuple (input_data, graph, tensor_shapes, flags, loss_obj).
    """
    if args.dataset is not None:
        input_data = load_input_data(args.dataset)
    else:
        input_data = None
    graph, tensor_shapes, flags = load_explanation_graph(args.expl_graph, args.flags, args)
    ##
    loss_loader = LossLoader()
    loss_loader.load_all("loss/")
    loss_cls, loss_params = loss_loader.get_loss(cast(str, flags.sgd_loss))
    loss_obj = None
    if loss_cls is not None:
        loss_obj = loss_cls(loss_params)
    return input_data, graph, tensor_shapes, flags, loss_obj


def run_preparing(args: argparse.Namespace) -> None:
    """Run preparation steps.

    Sets up embeddings/tensors and, if needed, vocabulary for the model.

    Args:
        args: Parsed CLI arguments.
    """
    input_data, graph, tensor_shapes, flags, loss_obj = _setup(args)
    ##
    tensor_provider = SwitchTensorProvider()
    embedding_generators: List[embed_gen.BaseEmbeddingGenerator] = []
    for embedding_filename in cast(list, flags.embedding):
        eg = embed_gen.EmbeddingGenerator()
        eg.load(embedding_filename)
        embedding_generators.append(eg)
    for embedding_filename in cast(list, flags.const_embedding):
        eg = embed_gen.EmbeddingGenerator(const_flag=True)
        eg.load(embedding_filename)
        embedding_generators.append(eg)
    tensor_provider.build(
        graph,
        tensor_shapes,
        input_data,
        flags,
        load_vocab=False,
        embedding_generators=embedding_generators,
    )


def run_training(args: argparse.Namespace) -> None:
    """Run training and optionally validation/saving.

    Args:
        args: Parsed CLI arguments.
    """
    input_data, graph, tensor_shapes, flags, loss_obj = _setup(args)
    ##
    logger.info("... computational graph")
    model = TprismModel(flags, tensor_shapes, graph, loss_obj=loss_obj)
    model.build(input_data, load_vocab=False, embedding_key="train")
    start_t = time.time()
    if flags.cycle:
        logger.info("... fit with cycle")
        model.solve()
    elif input_data is not None:
        logger.info("... fit with input data")
        #model.export_computational_graph(input_data)
        model.fit(input_data)
        model.pred(input_data)
    else:
        logger.info("... fit without input")
        model.fit()

    train_time = time.time() - start_t
    logger.info("total training time:%s[sec]", train_time)


def run_test(args: argparse.Namespace) -> None:
    """Run inference with a trained model and save outputs.

    Args:
        args: Parsed CLI arguments.
    """
    input_data, graph, tensor_shapes, flags, loss_obj = _setup(args)
    ##
    logger.info("... computational graph")
    model = TprismModel(flags, tensor_shapes, graph, loss_obj=loss_obj)
    model.build(input_data, load_vocab=True, embedding_key="test")
    if flags.model is not None:
        model.load(flags.model + ".best.model")
    start_t = time.time()
    logger.info("... prediction")
    if flags.cycle:
        model.solve()
    elif input_data is not None:
        #model.export_computational_graph(input_data)
        pred_y, out = model.pred(input_data)
    else:
        #model.export_computational_graph()
        pred_y, out = model.pred()
    train_time = time.time() - start_t
    logger.info("total training time:%s[sec]", train_time)
    logger.info("... output")
    np.save(cast(str, flags.output), out)
    logger.info("[SAVE] %s", flags.output)
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


def main() -> None:
    """CLI entrypoint.

    Parses arguments and dispatches to train/prepare/test modes.
    """
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
    parser.add_argument("--embedding", type=str,nargs="+", default=[], help="embedding file")
    parser.add_argument("--const_embedding", type=str,nargs="+", default=[], help="model file")
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

    parser.add_argument(
        "--no_verb", action="store_true", help="quiet mode: show only warnings and errors"
    )

    parser.add_argument(
        "--sgd_minibatch_size", type=str, default=None, help="[prolog flag]"
    )
    parser.add_argument("--max_iterate", type=str, default=None, help="[prolog flag]")
    parser.add_argument("--epoch", type=str, default=None, help="alias of --max_iterate")
    parser.add_argument(
        "--sgd_learning_rate", type=float, default=None, help="[prolog flag]"
    )
    parser.add_argument(
        "--sgd_loss",
        type=str,
        default=None,
        help="[prolog flag] nll/preference_pair",
    )
    parser.add_argument("--sgd_patience", type=int, default=None, help="[prolog flag] ")
    parser.add_argument(
        "--sgd_valid_ratio",
        type=float,
        default=None,
        help="[prolog flag] ratio of validation data used for early stopping (0 disables validation)",
    )
    parser.add_argument(
        "--sgd_goal_valid_ratio",
        type=float,
        default=None,
        help="[prolog flag] goal-level split: ratio of whole goals held out for validation (0 disables)",
    )
    parser.add_argument(
        "--sgd_optimizer",
        type=str,
        default=None,
        help="[prolog flag] sgd/adadelta/adam",
    )
    parser.add_argument(
        "--sgd_weight_decay",
        type=float,
        default=None,
        help="[prolog flag] weight decay",
    )
    parser.add_argument(
        "--sgd_adam_beta", type=float, default=None, help="[prolog flag]"
    )
    parser.add_argument(
        "--sgd_adam_gamma", type=float, default=None, help="[prolog flag]"
    )
    parser.add_argument(
        "--sgd_adam_epsilon", type=float, default=None, help="[prolog flag]"
    )
    parser.add_argument(
        "--sgd_adadelta_gamma", type=float, default=None, help="[prolog flag]"
    )
    parser.add_argument(
        "--sgd_adadelta_epsilon", type=float, default=None, help="[prolog flag]"
    )

    parser.add_argument("--cycle", action="store_true", help="cycle")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="verbose mode: show all debug messages"
    )
    parser.add_argument(
        "--debug_verb",
        type=str,
        nargs="+",
        default=None,
        choices=DEBUG_CATEGORIES,
        metavar="CATEGORY",
        help="show only debug messages of the given functional categories"
        " (multiple allowed): " + " ".join(DEBUG_CATEGORIES),
    )

    args = parser.parse_args()
    # config
    if args.config is not None:
        with open(args.config, "r") as fp:
            args.__dict__.update(json.load(fp))

    # logging (after --config so that a config file can set verbose/no_verb/debug_verb)
    setup_logging(verbose=args.verbose, quiet=args.no_verb, debug=args.debug_verb)
    if args.config is not None:
        logger.info("[LOAD] %s", args.config)

    # gpu/cpu
    if args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    elif args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # aliases
    if args.max_iterate is None and args.epoch is not None:
        args.max_iterate = args.epoch

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


if __name__ == "__main__":
    main()
