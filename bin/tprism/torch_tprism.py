#!/usr/bin/env python
"""PyTorch T-PRISM main module.

This module provides the PyTorch implementation of T-PRISM including the
model wrapper, training/evaluation loops, and command-line entrypoints.
All public functions and methods include Google-style docstrings.
"""


from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, cast
import torch
import torch.optim as optim
import json
import os
import numpy as np
import argparse
import time
import pickle

import tprism.expl_pb2 as expl_pb2
import tprism.torch_expl_graph as torch_expl_graph
import tprism.torch_embedding_generator as embed_gen
import tprism.torch_expl_tensor as torch_expl_tensor
from tprism.placeholder import PlaceholderData
from torch import Tensor


from tprism.util import (
    to_string_goal,
    Flags,
    build_goal_dataset,
    split_goal_dataset,
    get_goal_dataset,
    InputData,
    TensorInfoMapper,
)
from tprism.loader import (
    load_input_data,
    load_explanation_graph,
    LossLoader,
    OperatorLoader,
)
from tprism.loss.base import BaseLoss
import numpy as np
from numpy import ndarray


""" Main module for command line interface

This is called by tprism command (pytorch based tprism)
"""

def _check_and_detach(output: Any) -> ndarray|List[ndarray]|None:
    """Detach and convert tensors to NumPy arrays.

    Detaches any `torch.Tensor` or a list of tensors from the autograd graph
    and converts them to NumPy arrays. Non-tensor inputs are returned
    unchanged.

    Args:
        output: A `torch.Tensor`, a list of tensors, an `np.ndarray`, or any
            other value.

    Returns:
        Any: If a tensor, returns a NumPy array. If a list of tensors, returns
        a list of NumPy arrays. Otherwise, returns the input unchanged.
    """
    if output is None:
        return None
    elif type(output) is list:
        output = [o.detach().numpy() if type(o) != np.ndarray else o for o in output]
    elif type(output) is torch.Tensor:
        output = output.detach().numpy()
    return output

class TprismEvaluator:
    """Lightweight metrics accumulator for PyTorch training/evaluation.

    Tracks running losses, metric dictionaries, labels, and outputs per goal,
    and exposes helpers to format and retrieve the latest values.
    """

    def __init__(self, goal_dataset: Optional[Sequence[Any]] = None) -> None:
        if goal_dataset is not None:
            self.n_goals = len(goal_dataset)
        else:
            self.n_goals = 1
        self.loss_history:List[List[Any]] = [[] for _ in range(self.n_goals)]
        self.loss_dict_history:List[List[Any]] = [[] for _ in range(self.n_goals)]
        self.label:List[List[Any]] = [[] for _ in range(self.n_goals)]
        self.output:List[List[Any]] = [[] for _ in range(self.n_goals)]

    def start_epoch(self) -> None:
        """Initialize running counters at the beginning of an epoch."""
        self.running_loss:List[Any] = [0.0 for _ in range(self.n_goals)]
        self.running_loss_dict:List[Dict[str,Any]] = [{} for _ in range(self.n_goals)]
        self.running_count:List[int] = [0 for _ in range(self.n_goals)]
    
    def update(self, loss: Any, loss_dict: Dict[str, Any], j: int) -> None:
        """Accumulate loss and metrics for a batch.

        Args:
            loss: Loss value for the batch (tensor or numeric).
            loss_dict: Mapping from metric name to value for the batch.
            j: Goal index to update.
        """
        loss=_check_and_detach(loss)
        self.running_loss[j] += loss
        self.running_count[j] += 1
        for k, v in loss_dict.items():
            v=_check_and_detach(v)
            if k in self.running_loss_dict:
                self.running_loss_dict[j][k] += v
            else:
                self.running_loss_dict[j][k] = v

    def stop_epoch(self, j: int = 0, mean_flag: bool = True) -> None:
        """Average running values and push to history at epoch end.

        Args:
            j: Goal index.
            mean_flag: Whether to average accumulated values over batches.
        """
        if mean_flag:
            self.running_loss[j] /= self.running_count[j]
            for k in self.running_loss_dict[j].keys():
                self.running_loss_dict[j][k] /= self.running_count[j]
        self.loss_history[j].append(self.running_loss[j])
        self.loss_dict_history[j].append(self.running_loss_dict[j])

    def get_dict(self, prefix: str = "train") -> Dict[str, float]:
        """Get the latest loss/metrics as a dictionary.

        Args:
            prefix: Prefix for output keys (e.g., "train" or "valid").

        Returns:
            Dict[str, float]: Mapping of metric names to their latest values.
        """
        result = {}
        for j in range(self.n_goals):
            key = "{:s}-loss".format(prefix)
            if self.n_goals > 1:
                key = "goal{:d}-{:s}-loss".format(j,prefix)
            val = self.running_loss[j]
            result[key] = float(val)
            for k, v in self.running_loss_dict[j].items():
                if k[0] != "*":
                    m = "{:s}-{:s}-loss".format(prefix, k)
                else:
                    m = "*{:s}-{:s}".format(prefix, k[1:])
                result[m] = float(v)
        return result

    def get_msg(self, prefix: str = "train") -> str:
        """Format losses/metrics as a single-line message.

        Args:
            prefix: Prefix for output keys (e.g., "train" or "valid").

        Returns:
            str: Message like "key: value" joined by double spaces.
        """
        msg = []
        loss_dict=self.get_dict(prefix=prefix)
        if loss_dict is not None:
            for key, val in loss_dict.items():
                m = "{:s}: {:.3f}".format(key, val)
                msg.append(m)
            return "  ".join(msg)
        else:
           return ""

    def get_loss(self) -> List[float]:
        """Return the latest per-goal loss.

        Returns:
            List[float]: Per-goal running loss values.
        """
        return self.running_loss

    def update_data(self, output: Sequence[Any]|Tensor, label: Optional[Sequence[Any]]|Tensor, j: int) -> None:
        """Append predictions and labels to accumulation buffers.

        Args:
            output: Model outputs (sequence indexed by goal).
            label: Ground-truth labels (sequence indexed by goal) or None.
            j: Goal index.
        """
        _o = cast(Sequence[Any], _check_and_detach(output[j]))
        self.output[j].extend(_o)
        if label is not None:
            _l = cast(Sequence[Any],_check_and_detach(label[j]))
            self.label[j].extend(_l)



class TprismModel:
    """T-PRISM model implemented with PyTorch with training/inference wrappers.

    Args:
        flags: Runtime flags/hyperparameters (`tprism.util.Flags`).
        tensor_shapes: Mapping of tensor names to shapes.
        graph: Explanation graph object.
        loss_cls: Loss class (subclass of `BaseLoss`). If None, `BaseLoss` is used.
    """

    def __init__(
        self,
        flags: Flags,
        tensor_shapes: TensorInfoMapper,
        graph: expl_pb2.ExplGraph,
        loss_cls: Optional[Type[BaseLoss]],
    ) -> None:
        self.graph = graph
        self.flags = flags
        self.tensor_shapes = tensor_shapes
        if loss_cls is None:
            self.loss_cls = BaseLoss
        else:
            self.loss_cls = loss_cls
        self.operator_loader = OperatorLoader()
        self.operator_loader.load_all("op/torch_")

    def build(self, input_data: Optional[List[InputData]], load_vocab: bool, embedding_key: str, verbose: bool=False) -> None:
        """Initialize embeddings, data provider, and computational graph.

        Args:
            input_data: Input dataset or None.
            load_vocab: Whether to load vocabulary.
            embedding_key: Embedding key (e.g., "train" or "test").
            verbose: Whether to print verbose output.
        """
        self._build_embedding(embedding_key)
        self._set_data(input_data, load_vocab, verbose)
        self._build_explanation_graph()

    def _build_embedding(self, embedding_key: str) -> None:
        """Construct and load embedding generators.

        Args:
            embedding_key: Embedding key used when loading variable embeddings.
        """
        embedding_generators:List[embed_gen.BaseEmbeddingGenerator] = []
        if self.flags.embedding is not None:
            for embedding_filename in self.flags.embedding:
                eg = embed_gen.EmbeddingGenerator()
                eg.load(embedding_filename, key=embedding_key)
                embedding_generators.append(eg)
        if self.flags.const_embedding is not None:
            for embedding_filename in self.flags.const_embedding:
                eg = embed_gen.EmbeddingGenerator(const_flag=True)
                eg.load(embedding_filename)
                embedding_generators.append(eg)
        cycle_embedding_generator = None
        if self.flags.cycle:
            cycle_embedding_generator = embed_gen.CycleEmbeddingGenerator()
            cycle_embedding_generator.load(self.tensor_shapes)
            embedding_generators.append(cycle_embedding_generator)
        self.embedding_generators = embedding_generators
        self.cycle_embedding_generator = cycle_embedding_generator

    def _set_data(self, input_data: Optional[List[InputData]], load_vocab: bool, verbose: bool) -> None:
        """Build tensor provider and set input data/vocabulary.

        Args:
            input_data: Input dataset or None.
            load_vocab: Whether to load vocabulary.
        """
        self.tensor_provider = torch_expl_tensor.TorchSwitchTensorProvider()
        self.tensor_provider.build(
            self.graph,
            self.tensor_shapes,
            input_data,
            self.flags,
            load_vocab=load_vocab,
            embedding_generators=self.embedding_generators,
            verbose=verbose,
        )

    def _build_explanation_graph(self) -> None:
        """Build the PyTorch computational explanation graph."""
        self.comp_expl_graph = torch_expl_graph.TorchComputationalExplGraph(
            self.graph, self.tensor_provider, self.operator_loader, self.cycle_embedding_generator
        )

    def solve(self, input_data: Optional[List[InputData]] = None)->None:
        """Optimization loop for cyclic settings (embedding adjustment).

        Args:
            input_data: Input dataset. Currently unsupported; use None.

        Returns:
            Optional[Any]: Currently returns None (side effects only).
        """
        if input_data is None:
            self._solve_no_data()
        else:
            print("solver with input data is not implemented")
            

    def _solve_no_data(self) -> None:
        """Run simple optimization for cycles without input data."""
        print("... training phase")
        print("... training variables")
        for key, param in self.comp_expl_graph.state_dict().items():
            print(key, param.shape)
        print("... building explanation graph")
        prev_loss = None
        for step in range(self.flags.max_iterate):
            feed_dict:Dict[PlaceholderData, Tensor] = {}
            for embedding_generator in self.embedding_generators:
                if embedding_generator is not None:
                    feed_dict = embedding_generator.build_feed(feed_dict, None)
            self.tensor_provider.set_input(feed_dict)
            # out_inside = self.sess.run(inside, feed_dict=feed_dict)
            goal_inside, loss_list = self.comp_expl_graph.forward(verbose=True)
            inside = []
            for goal in goal_inside:
                if goal is None:
                    raise RuntimeError("goal_inside contains None")
                l1 = goal.inside
                inside.append(l1)
            loss = Tensor(0)
            for embedding_generator in self.embedding_generators:
                if embedding_generator is not None:
                    loss = embedding_generator.update(inside)
            print("step", step, "loss:", loss)
            if loss is None:
                raise RuntimeError("loss is None")
            if loss < 1.0e-20:
                break
            if prev_loss is not None and not loss < prev_loss:
                pass
            prev_loss = loss
        ##
        return None

    def fit(self, input_data: Optional[List[InputData]] = None, verbose: bool = False) -> Any:
        """Train the model.

        Args:
            input_data: Dataset or None for no-data training.
            verbose: Whether to print verbose logs.

        Returns:
            TprismEvaluator | Tuple[TprismEvaluator, TprismEvaluator]:
            Training evaluator only (no-data) or a tuple of (train, valid).
        """
        if input_data is None:
            return self._fit_no_data(verbose)
        else:
            return self._fit(input_data, verbose)

    def _fit_no_data(self, verbose: bool) -> TprismEvaluator:
        """Train without input data and return training metrics.

        Args:
            verbose: Whether to print verbose logs.

        Returns:
            TprismEvaluator: Training evaluator with recorded metrics.
        """
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
            feed_dict:Dict[PlaceholderData, Tensor] = {}
            for embedding_generator in self.embedding_generators:
                if embedding_generator is not None:
                    feed_dict = embedding_generator.build_feed(feed_dict, None)
            self.tensor_provider.set_input(feed_dict)
            # print("... iteration")
            goal_inside, loss_list = self.comp_expl_graph.forward()
            if goal_inside is not None:
                loss, output, label = loss_cls.call(
                    self.graph, goal_inside, self.tensor_provider
                )
                optimizer.zero_grad()
                if loss is None:
                    raise RuntimeError("loss is None")
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
                    if self.flags.model is not None:
                        print("... saving best model to", self.flags.model + ".best.model")
                        self.save(self.flags.model + ".best.model")
            else:
                raise RuntimeError("goal_inside is None")
        if self.flags.model is not None:
            print("... saving last model to", self.flags.model + ".last.model")
            self.save(self.flags.model + ".last.model")
        return train_evaluator

    def _build_feed(
        self,
        ph_vars: Sequence[Any],
        dataset: Any,
        idx: Sequence[int]|ndarray,
        verbose: bool = True,
    ) -> Dict[Any, torch.Tensor]:
        """Build a batch feed dict for placeholders.

        Args:
            ph_vars: Placeholder variables.
            dataset: Dataset supporting slicing `dataset[i, idx]`.
            idx: Indices to slice the dataset.
            verbose: Whether to print input shapes.

        Returns:
            Dict[Any, torch.Tensor]: Mapping from placeholder to tensors.
        """
        if verbose:
            for i, ph in enumerate(ph_vars):
                print("[INFO feed]", ph, ph.name)
                print("[INFO feed]", dataset[i, idx].shape)
        feed_dict = {ph: torch.tensor(dataset[i, idx]) for i, ph in enumerate(ph_vars)}
        return feed_dict

    def _set_batch_input(
        self,
        goal: Dict[str, Any],
        train_idx: Sequence[Sequence[int]],
        j: int,
        itr: int,
    ) -> None:
        """Set the specified batch inputs into the tensor provider.

        Args:
            goal: Goal object containing placeholders and dataset.
            train_idx: Indices for train/valid/test split per goal.
            j: Goal index.
            itr: Batch iteration index.
        """
        batch_size = self.flags.sgd_minibatch_size
        ph_vars = goal["placeholders"]
        dataset = goal["dataset"]
        batch_idx = np.array(train_idx[j][itr * batch_size : (itr + 1) * batch_size])
        feed_dict = self._build_feed(ph_vars, dataset, batch_idx)
        # for k,v in feed_dict.items():
        #    print(k,v.shape)
        for embedding_generator in self.embedding_generators:
            if embedding_generator is not None:
                feed_dict = embedding_generator.build_feed(feed_dict, batch_idx)
        self.tensor_provider.set_input(feed_dict)

    def _fit(self, input_data: List[InputData], verbose: bool) -> Tuple[TprismEvaluator, TprismEvaluator]:
        """Train with input data, performing train/validation loops.

        Args:
            input_data: Training dataset.
            verbose: Whether to print verbose logs.

        Returns:
            Tuple[TprismEvaluator, TprismEvaluator]: (train_evaluator, valid_evaluator).
        """
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
        train_evaluator = TprismEvaluator(goal_dataset)
        valid_evaluator = TprismEvaluator(goal_dataset)
        check_point_flag= False
        for epoch in range(self.flags.max_iterate):
            start_t = time.time()
            train_evaluator.start_epoch()
            valid_evaluator.start_epoch()
            for j, goal in enumerate(goal_dataset):
                if verbose:
                    print(goal)
                np.random.shuffle(train_idx[j])
                # training update
                num_itr = len(train_idx[j]) // batch_size
                for itr in range(num_itr):
                    self._set_batch_input(goal, train_idx, j, itr)
                    goal_inside, loss_list = self.comp_expl_graph.forward(
                        verbose=verbose
                    )
                    if goal_inside is not None:
                        loss, output, label = loss_cls.call(
                            self.graph, goal_inside, self.tensor_provider
                        )
                        if output is None or loss is None:
                            raise RuntimeError("output/loss is None in training")
                        if label is not None:
                            metrics = loss_cls.metrics(
                                _check_and_detach(output[j]), 
                                _check_and_detach(label[j])
                            )
                        else:
                            metrics = loss_cls.metrics(_check_and_detach(output[j]), None)
                        loss_list.update(metrics)
                        # display_graph(output[j],'graph_pytorch')
                        optimizer.zero_grad()
                        loss[j].backward()
                        optimizer.step()
                        train_evaluator.update(loss[j], loss_list, j)
                    else:
                        raise RuntimeError("goal_inside is None")
                # validation
                num_itr = len(valid_idx[j]) // batch_size
                for itr in range(num_itr):
                    self._set_batch_input(goal, valid_idx, j, itr)
                    goal_inside, loss_list = self.comp_expl_graph.forward()
                    loss, output, label = loss_cls.call(
                        self.graph, goal_inside, self.tensor_provider
                    )
                    if output is None or loss is None:
                        raise RuntimeError("output/loss is None in training")
                    if label is not None:
                        metrics = loss_cls.metrics(
                            _check_and_detach(output[j]), 
                            _check_and_detach(label[j])
                        )
                    else:
                        metrics = loss_cls.metrics(_check_and_detach(output[j]), None)
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
                    if self.flags.model is not None:
                        print(
                            "... saving best model to",
                            self.flags.model + ".best.model",
                        )
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
            train_evaluator.stop_epoch()
            valid_evaluator.stop_epoch()
        if self.flags.model is not None:
            print("... saving last model to", self.flags.model + ".last.model")
            self.save(self.flags.model + ".last.model")
        return train_evaluator, valid_evaluator

    def save(self, filename: str) -> None:
        """Save current model parameters to a file.

        Args:
            filename: Path to save the model parameters.
        """
        torch.save(self.comp_expl_graph.state_dict(), filename)

    def load(self, filename: str) -> None:
        """Load model parameters from a file and set them.

        Args:
            filename: Path to a saved model parameters file.
        """
        if os.path.isfile(filename):
            self.comp_expl_graph.load_state_dict(torch.load(filename))
        else:
            print("[SKIP] skip loading")

    def pred(self, input_data: Optional[List[InputData]] = None, verbose: bool = False) -> Tuple[Any, Any]:
        """Run prediction (inference).

        Args:
            input_data: Dataset or None for no-data inference.
            verbose: Whether to print verbose logs.

        Returns:
            Tuple[Any, Any]: (labels, outputs), usually converted to NumPy.
        """
        if input_data is not None:
            return self._pred(input_data, verbose)
        else:
            return self._pred_no_data(verbose)
        
    def _pred_no_data(self, verbose: bool) -> Tuple[Optional[Any], Optional[Any]]:
        """Infer without input data.

        Args:
            verbose: Whether to print verbose logs.

        Returns:
            Tuple[Optional[Any], Optional[Any]]: (labels, outputs) detached and converted.
        """
        print("... prediction")
        print("... loaded variables")
        for key, param in self.comp_expl_graph.state_dict().items():
            print(key, param.shape)
        print("... initialization")
        loss_cls = self.loss_cls()
        evaluator = TprismEvaluator()
        ###
        feed_dict: Dict[PlaceholderData, Tensor] = {}
        for embedding_generator in self.embedding_generators:
            if embedding_generator is not None:
                feed_dict = embedding_generator.build_feed(feed_dict, None)
        self.tensor_provider.set_input(feed_dict)
        print("... predicting")
        start_t = time.time()
        goal_inside, loss_list = self.comp_expl_graph.forward(verbose=verbose,verbose_embedding=verbose)
        loss, output, label = loss_cls.call(
            self.graph, goal_inside, self.tensor_provider
        )
        metrics=loss_cls.metrics(output, label)
        pred_time = time.time() - start_t
        print("prediction time:{0}".format(pred_time) + "[sec]")
        # train_acc=sklearn.metrics.accuracy_score(all_label,all_output)
        #print("loss:", np.sum(evaluator.get_loss())
        if loss is not None:
            print("loss:", torch.sum(loss))
        print("metrics:", metrics)
        # 
        label_  = _check_and_detach(label)
        output_ = _check_and_detach(output)
        return label_, output_

    def export_computational_graph(self, input_data: Optional[List[InputData]] = None, verbose: bool = False) -> None:
        """Debug helper: print computational graph paths to stdout.

        Args:
            input_data: Dataset to build batches from, or None.
            verbose: Whether to print verbose logs.
        """
        print("... prediction")
        goal_dataset=None
        if input_data:
            goal_dataset = build_goal_dataset(input_data, self.tensor_provider)
        print("... loaded variables")
        for key, param in self.comp_expl_graph.state_dict().items():
            print(key, param.shape)
        print("... initialization")
        loss_cls = self.loss_cls()
        if input_data:
            batch_size = self.flags.sgd_minibatch_size
            test_idx = get_goal_dataset(goal_dataset)
        print("... exporting")
        if input_data and goal_dataset is not None:
            for j, goal in enumerate(goal_dataset):
                # valid
                num_itr = len(test_idx[j]) // batch_size
                for itr in range(num_itr):
                    self._set_batch_input(goal, test_idx, j, itr)
                    goal_inside, loss_list = self.comp_expl_graph.forward(dryrun=True)
                    for g in goal_inside:
                        if g is not None:
                            print(g)
                            for path in g.inside:
                                print("  ", path)
        else:
            goal_inside, loss_list = self.comp_expl_graph.forward(dryrun=True)
            for g in goal_inside:
                if g is not None:
                    print(g)
                    for path in g.inside:
                        print("  ", path)

    def _pred(self, input_data: List[InputData], verbose: bool) -> Tuple[List[Any], List[Any]]:
        """Infer with input data.

        Args:
            input_data: Dataset for prediction.
            verbose: Whether to print verbose logs.

        Returns:
            Tuple[List[Any], List[Any]]: (labels, outputs) per goal.
        """
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
            if len(test_idx[j])%batch_size==0:
                num_itr = len(test_idx[j]) // batch_size
            else:
                num_itr = len(test_idx[j]) // batch_size + 1
            evaluator.start_epoch()
            for itr in range(num_itr):
                self._set_batch_input(goal, test_idx, j, itr)
                goal_inside, loss_list = self.comp_expl_graph.forward(verbose=verbose, verbose_embedding=verbose)
                loss, output, label = loss_cls.call(
                    self.graph, goal_inside, self.tensor_provider
                )
                if loss is not None:
                    evaluator.update(loss[j], loss_list, j)
                if output is not None:
                    evaluator.update_data(output, label, j)
                else:
                    raise RuntimeError("output is None in prediction")
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


def run_preparing(args: argparse.Namespace) -> None:
    """Run preparation steps.

    Sets up embeddings/tensors and, if needed, vocabulary for the model.

    Args:
        args: Parsed CLI arguments.
    """
    if args.dataset is not None:
        input_data = load_input_data(args.dataset)
    else:
        input_data = None
    graph, tensor_shapes, flags = load_explanation_graph(args.expl_graph, args.flags, args)
    ##
    loss_loader = LossLoader()
    loss_loader.load_all("loss/")
    loss_cls = loss_loader.get_loss(cast(str,flags.sgd_loss))
    ##
    tensor_provider = torch_expl_tensor.TorchSwitchTensorProvider()
    embedding_generators: List[embed_gen.BaseEmbeddingGenerator] = []
    for embedding_filename in cast(list,flags.embedding):
        eg = embed_gen.EmbeddingGenerator()
        eg.load(embedding_filename )
        embedding_generators.append(eg)
    for embedding_filename in cast(list,flags.const_embedding):
        eg = embed_gen.EmbeddingGenerator(const_flag=True)
        eg.load(embedding_filename )
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
    if args.dataset is not None:
        input_data = load_input_data(args.dataset)
    else:
        input_data = None
    graph, tensor_shapes, flags = load_explanation_graph(args.expl_graph, args.flags, args)
    ##
    loss_loader = LossLoader()
    loss_loader.load_all("loss/torch*")
    loss_cls = loss_loader.get_loss(cast(str,flags.sgd_loss))
    ##
    print("... computational graph")
    model = TprismModel(flags, tensor_shapes, graph, loss_cls)
    model.build(input_data, load_vocab=False, embedding_key="train", verbose=False)
    start_t = time.time()
    if flags.cycle:
        print("... fit with cycle")
        model.solve()
    elif input_data is not None:
        print("... fit with input data")
        #model.export_computational_graph(input_data)
        model.fit(input_data, verbose=False)
        model.pred(input_data)
    else:
        print("... fit without input")
        model.fit(verbose=False)

    train_time = time.time() - start_t
    print("total training time:{0}".format(train_time) + "[sec]")


def run_test(args: argparse.Namespace) -> None:
    """Run inference with a trained model and save outputs.

    Args:
        args: Parsed CLI arguments.
    """
    if args.dataset is not None:
        input_data = load_input_data(args.dataset)
    else:
        input_data = None
    graph, tensor_shapes, flags = load_explanation_graph(args.expl_graph, args.flags, args)
    ##
    loss_loader = LossLoader()
    loss_loader.load_all("loss/torch*")
    loss_cls = loss_loader.get_loss(cast(str,flags.sgd_loss))
    ##
    print("... computational graph")
    model = TprismModel(flags, tensor_shapes, graph, loss_cls)
    model.build(input_data, load_vocab=True, embedding_key="test", verbose=False)
    if flags.model is not None:
        model.load(flags.model + ".best.model")
    start_t = time.time()
    print("... prediction")
    if flags.cycle:
        model.solve()
    elif input_data is not None:
        #model.export_computational_graph(input_data)
        pred_y, out = model.pred(input_data,verbose=args.verbose)
    else:
        #model.export_computational_graph()
        pred_y, out = model.pred(verbose=args.verbose)
    train_time = time.time() - start_t
    print("total training time:{0}".format(train_time) + "[sec]")
    print("... output")
    np.save(cast(str,flags.output),out)
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
    parser.add_argument("--verbose", action="store_true", help="verbose")

    args = parser.parse_args()
    # config
    if args.config is None:
        pass
    else:
        print("[LOAD] ", args.config)
        fp = open(args.config, "r")
        args.__dict__.update(json.load(fp))

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


if __name__ == "__main__":
    main()
