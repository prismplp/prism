"""PyTorch T-PRISM model module.

This module provides the T-PRISM model wrapper (`TprismModel`) and the
training/evaluation loops. All public functions and methods include
Google-style docstrings.
"""


from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, cast
import logging
import os
import time

import numpy as np
from numpy import ndarray
import torch
import torch.optim as optim
from torch import Tensor

import tprism.expl_pb2 as expl_pb2
import tprism.embedding_generator as embed_gen
from tprism.expl_graph import ComputationalExplGraph
from tprism.expl_tensor import SwitchTensorProvider
from tprism.placeholder import PlaceholderData
from tprism.util import (
    Flags,
    build_goal_dataset,
    split_goals,
    split_goal_dataset,
    get_goal_dataset,
    debug_logger,
    set_log_level,
    InputData,
    TensorInfoMapper,
)
from tprism.loader import OperatorLoader
from tprism.loss.base import BaseLoss

logger = logging.getLogger(__name__)
feed_logger = debug_logger("feed")
minibatch_logger = debug_logger("minibatch")
param_logger = debug_logger("param")


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
        output = output.detach().numpy(force = True)
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
            if k in self.running_loss_dict[j]:
                self.running_loss_dict[j][k] += v
            else:
                self.running_loss_dict[j][k] = v

    def stop_epoch(self, j: int = 0, mean_flag: bool = True) -> None:
        """Average running values and push to history at epoch end.

        Args:
            j: Goal index.
            mean_flag: Whether to average accumulated values over batches.
        """
        if mean_flag and self.running_count[j] > 0:
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
            # goals not evaluated in this epoch (e.g. held out by the
            # goal-level split) have no batches to average
            if self.running_count[j] == 0:
                continue
            key = "{:s}-loss".format(prefix)
            if self.n_goals > 1:
                key = "goal{:d}-{:s}-loss".format(j,prefix)
            val = self.running_loss[j]
            result[key] = float(val) / self.running_count[j]
            for k, v in self.running_loss_dict[j].items():
                if k[0] != "*":
                    m = "{:s}-{:s}-loss".format(prefix, k)
                else:
                    m = "*{:s}-{:s}".format(prefix, k[1:])
                result[m] = float(v) / self.running_count[j]
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
        log_level: Optional log level for the tprism package, as a `logging`
            constant or name ("debug", "info", "warning", "error", ...). The
            default (None) keeps the current configuration (INFO for library
            use). See `tprism.util.set_log_level`.
    """

    def __init__(
        self,
        flags: Flags,
        tensor_shapes: TensorInfoMapper,
        graph: expl_pb2.ExplGraph,
        loss_cls: Optional[Type[BaseLoss]]=None,
        loss_obj: Optional[BaseLoss]=None,
        log_level: Optional[int|str]=None,
    ) -> None:
        if log_level is not None:
            set_log_level(log_level)
        self.graph = graph
        self.flags = flags
        self.tensor_shapes = tensor_shapes
        if loss_obj is None:
            if loss_cls is None:
                self.loss_obj = BaseLoss()
            else:
                self.loss_obj = loss_cls()
        else:
            self.loss_obj = loss_obj
        self.operator_loader = OperatorLoader()
        self.operator_loader.load_all("op/")

    def build(self, input_data: Optional[List[InputData]], load_vocab: bool, embedding_key: str, verbose: bool=False) -> None:
        """Initialize embeddings, data provider, and computational graph.

        Args:
            input_data: Input dataset or None.
            load_vocab: Whether to load vocabulary.
            embedding_key: Embedding key (e.g., "train" or "test").
            verbose: Deprecated and ignored; verbosity is controlled by the
                logging level (see `tprism.util.setup_logging`).
        """
        self._build_embedding(embedding_key)
        self._set_data(input_data, load_vocab)
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

    def _set_data(self, input_data: Optional[List[InputData]], load_vocab: bool) -> None:
        """Build tensor provider and set input data/vocabulary.

        Args:
            input_data: Input dataset or None.
            load_vocab: Whether to load vocabulary.
        """
        self.tensor_provider = SwitchTensorProvider()
        self.tensor_provider.build(
            self.graph,
            self.tensor_shapes,
            input_data,
            self.flags,
            load_vocab=load_vocab,
            embedding_generators=self.embedding_generators,
        )

    def _build_explanation_graph(self) -> None:
        """Build the PyTorch computational explanation graph."""
        self.comp_expl_graph = ComputationalExplGraph(
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
            logger.error("solver with input data is not implemented")


    def _solve_no_data(self) -> None:
        """Run simple optimization for cycles without input data."""
        logger.info("... training phase")
        logger.info("... training variables")
        for key, param in self.comp_expl_graph.state_dict().items():
            param_logger.debug("%s %s", key, param.shape)
        logger.info("... building explanation graph")
        prev_loss = None
        for step in range(self.flags.max_iterate):
            feed_dict:Dict[PlaceholderData, Tensor] = {}
            for embedding_generator in self.embedding_generators:
                if embedding_generator is not None:
                    feed_dict = embedding_generator.build_feed(feed_dict, None)
            self.tensor_provider.set_input(feed_dict)
            # out_inside = self.sess.run(inside, feed_dict=feed_dict)
            goal_inside, loss_list = self.comp_expl_graph.forward()
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
            logger.info("step %d loss: %s", step, loss)
            if loss is None:
                raise RuntimeError("loss is None")
            if loss < 1.0e-20:
                break
            if prev_loss is not None and not loss < prev_loss:
                pass
            prev_loss = loss
        ##
        return None

    def _build_optimizer(self) -> optim.Optimizer:
        """Create the optimizer selected by the sgd_optimizer flag.

        Honors the prism flags sgd_optimizer, sgd_learning_rate,
        sgd_weight_decay, sgd_adam_beta/gamma/epsilon, and
        sgd_adadelta_gamma/epsilon.

        Returns:
            optim.Optimizer: A configured optimizer over the model parameters.
        """
        params = self.comp_expl_graph.parameters()
        name = self.flags.sgd_optimizer
        lr = self.flags.sgd_learning_rate
        weight_decay = self.flags.sgd_weight_decay
        if name == "adam":
            return optim.Adam(
                params,
                lr,
                betas=(self.flags.sgd_adam_beta, self.flags.sgd_adam_gamma),
                eps=self.flags.sgd_adam_epsilon,
                weight_decay=weight_decay,
            )
        elif name == "adadelta":
            return optim.Adadelta(
                params,
                lr,
                rho=self.flags.sgd_adadelta_gamma,
                eps=self.flags.sgd_adadelta_epsilon,
                weight_decay=weight_decay,
            )
        elif name == "sgd":
            return optim.SGD(params, lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"unknown sgd_optimizer: {name}")

    def fit(self, input_data: Optional[List[InputData]] = None, verbose: bool = False) -> Any:
        """Train the model.

        Args:
            input_data: Dataset or None for no-data training.
            verbose: Deprecated and ignored; verbosity is controlled by the
                logging level (see `tprism.util.setup_logging`).

        Returns:
            TprismEvaluator | Tuple[TprismEvaluator, TprismEvaluator]:
            Training evaluator only (no validation goals) or a tuple of
            (train, valid) evaluators.
        """
        if input_data is None:
            return self._fit_no_data()
        else:
            return self._fit(input_data)

    def _goal_metrics(
        self,
        loss: Tensor,
        output: Optional[Tensor],
        label: Optional[Tensor],
        sel: Optional[Tensor],
    ) -> Dict[str, Any]:
        """Compute loss metrics, restricted to the given goals when possible.

        Args:
            loss: Per-goal loss returned by the loss plugin.
            output: Loss plugin output.
            label: Loss plugin label (or None).
            sel: Goal indices to restrict to, or None for all goals.

        Returns:
            Dict[str, Any]: Metrics from the loss plugin. When `sel` is
            given but output/label are not aligned with the per-goal loss,
            an empty dict is returned.
        """
        if sel is None:
            return self.loss_obj.metrics(
                _check_and_detach(output), _check_and_detach(label)
            )
        if not isinstance(output, Tensor) or output.shape[0] != loss.shape[0]:
            return {}
        sub_label = None
        if isinstance(label, Tensor) and label.shape[0] == loss.shape[0]:
            sub_label = _check_and_detach(label[sel])
        return self.loss_obj.metrics(_check_and_detach(output[sel]), sub_label)

    def _fit_no_data(self) -> TprismEvaluator | Tuple[TprismEvaluator, TprismEvaluator]:
        """Train without input data and return training metrics.

        As in `_fit`, a goal-level split (`split_goals` with the
        `sgd_goal_valid_ratio` flag) can hold out whole root goals for
        validation. Each epoch runs a single forward pass; when
        validation goals exist, only the training goals' per-goal losses
        `loss[j]` are summed for the gradient (held-out goals never
        contribute), the held-out goals' losses are evaluated every
        epoch, and early stopping (`sgd_patience`) is applied to the
        validation loss. The split requires a per-goal loss (one entry
        per rank_root); a loss reduced to a scalar disables validation.

        Returns:
            TprismEvaluator | Tuple[TprismEvaluator, TprismEvaluator]:
            Training evaluator only (no validation goals) or a tuple of
            (train, valid) evaluators.
        """
        logger.info("... training phase")
        logger.info("... training variables")
        for key, param in self.comp_expl_graph.state_dict().items():
            param_logger.debug("%s %s", key, param.shape)
        optimizer = self._build_optimizer()
        logger.info("... building explanation graph")

        logger.info("... splitting data")
        # goal-level split (coarse): whole root goals held out for
        # validation, as in _fit
        train_goal_idx, valid_goal_idx = split_goals(
            self.graph.root_list, valid_ratio=self.flags.sgd_goal_valid_ratio
        )
        has_valid = len(valid_goal_idx) > 0
        if has_valid:
            if len(train_goal_idx) == 0:
                raise ValueError(
                    "sgd_goal_valid_ratio=%s leaves no training goals"
                    % (self.flags.sgd_goal_valid_ratio,)
                )
            logger.info(
                "... goal-level split: %d training goals / %d held-out goals %s",
                len(train_goal_idx),
                len(valid_goal_idx),
                sorted(valid_goal_idx.tolist()),
            )
        else:
            logger.info("... no validation data: early stopping is disabled")
        train_sel = torch.as_tensor(train_goal_idx, dtype=torch.long)
        valid_sel = torch.as_tensor(valid_goal_idx, dtype=torch.long)

        best_total_loss = None
        best_valid_loss: Optional[Any] = None
        patient_count = 0
        train_evaluator = TprismEvaluator()
        valid_evaluator = TprismEvaluator()
        for epoch in range(self.flags.max_iterate):
            start_t = time.time()
            # train
            feed_dict:Dict[PlaceholderData, Tensor] = {}
            for embedding_generator in self.embedding_generators:
                if embedding_generator is not None:
                    feed_dict = embedding_generator.build_feed(feed_dict, None)
            self.tensor_provider.set_input(feed_dict)
            # print("... iteration")
            train_evaluator.start_epoch()
            improved = False
            goal_inside, loss_list = self.comp_expl_graph.forward()
            if goal_inside is None:
                raise RuntimeError("goal_inside is None")
            loss, output, label = self.loss_obj.call(
                self.graph, goal_inside, self.tensor_provider
            )
            if loss is None:
                raise RuntimeError("loss is None")
            if has_valid and loss.dim() == 0:
                logger.warning(
                    "... loss is a scalar (not per-goal):"
                    " goal-level validation is disabled"
                )
                has_valid = False
            optimizer.zero_grad()
            if has_valid:
                # only the training goals contribute to the gradient
                total_loss = torch.sum(loss[train_sel], dim=0)
            else:
                total_loss = torch.sum(loss, dim=0)
            total_loss.backward()
            optimizer.step()
            metrics = self._goal_metrics(
                loss, output, label, train_sel if has_valid else None
            )
            metrics.update(loss_list)
            train_evaluator.update(total_loss, metrics, 0)
            # display_graph(output[j],'graph_pytorch')
            if has_valid:
                # goal-level validation: held-out goals are never trained on
                valid_loss = torch.sum(loss[valid_sel], dim=0).detach()
                valid_metrics = self._goal_metrics(loss, output, label, valid_sel)
                valid_evaluator.start_epoch()
                valid_evaluator.update(valid_loss, valid_metrics, 0)
                # checking improvement of the validation loss
                if (
                    best_valid_loss is None
                    or valid_evaluator.running_loss[0] < best_valid_loss
                ):
                    best_valid_loss = valid_evaluator.running_loss[0]
                    improved = True
                logger.info(
                    "[%4d]  %s %s (%2d) %s",
                    epoch + 1,
                    train_evaluator.get_msg("train"),
                    valid_evaluator.get_msg("valid"),
                    patient_count,
                    "*" if improved else "",
                )
                train_evaluator.stop_epoch()
                valid_evaluator.stop_epoch()
            else:
                logger.info("[%4d]  %s", epoch + 1, train_evaluator.get_msg("train"))
                train_evaluator.stop_epoch()
            train_time = time.time() - start_t
            logger.info("train time:%s[sec]", train_time)
            # checkpointing and early stopping
            if has_valid:
                if improved:
                    patient_count = 0
                    if self.flags.model is not None:
                        logger.info("... saving best model to %s", self.flags.model + ".best.model")
                        self.save(self.flags.model + ".best.model")
                else:
                    patient_count += 1
                if patient_count >= self.flags.sgd_patience:
                    logger.info(
                        "... early stopping: no improvement in validation loss"
                        " for %d epochs",
                        patient_count,
                    )
                    break
            elif (
                best_total_loss is None
                or train_evaluator.running_loss[0] < best_total_loss
            ):
                best_total_loss = train_evaluator.running_loss[0]
                if self.flags.model is not None:
                    logger.info("... saving best model to %s", self.flags.model + ".best.model")
                    self.save(self.flags.model + ".best.model")
        if self.flags.model is not None:
            logger.info("... saving last model to %s", self.flags.model + ".last.model")
            self.save(self.flags.model + ".last.model")
        if has_valid:
            return train_evaluator, valid_evaluator
        return train_evaluator

    def _build_feed(
        self,
        ph_vars: Sequence[Any],
        dataset: Any,
        idx: Sequence[int]|ndarray,
    ) -> Dict[Any, torch.Tensor]:
        """Build a batch feed dict for placeholders.

        Args:
            ph_vars: Placeholder variables.
            dataset: Dataset supporting slicing `dataset[i, idx]`.
            idx: Indices to slice the dataset.

        Returns:
            Dict[Any, torch.Tensor]: Mapping from placeholder to tensors.
        """
        if feed_logger.isEnabledFor(logging.DEBUG):
            for i, ph in enumerate(ph_vars):
                feed_logger.debug("[feed] %s %s: shape=%s", ph, ph.name, dataset[i, idx].shape)
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

    def _evaluate_batch(self, j: int) -> Tuple[Tensor, Dict[str, Any]]:
        """Forward the current batch and compute loss/metrics for goal j.

        The batch inputs must be set beforehand via `_set_batch_input`.

        Args:
            j: Goal index.

        Returns:
            Tuple[Tensor, Dict[str, Any]]: (per-goal loss tensor, metrics dict).
        """
        goal_inside, loss_list = self.comp_expl_graph.forward()
        if goal_inside is None:
            raise RuntimeError("goal_inside is None")
        loss, output, label = self.loss_obj.call(
            self.graph, goal_inside, self.tensor_provider
        )
        if output is None or loss is None:
            raise RuntimeError("output/loss is None in training")
        if label is not None:
            metrics = self.loss_obj.metrics(
                _check_and_detach(output[j]),
                _check_and_detach(label[j])
            )
        else:
            metrics = self.loss_obj.metrics(_check_and_detach(output[j]), None)
        loss_list.update(metrics)
        return loss, loss_list

    def _validate_goal(
        self,
        goal_dataset: Sequence[Dict[str, Any]],
        valid_idx: Sequence[Sequence[int]],
        j: int,
        valid_evaluator: "TprismEvaluator",
    ) -> int:
        """Run validation batches for goal j and return the number of batches.

        Args:
            goal_dataset: All goals.
            valid_idx: Validation indices per goal.
            j: Goal index.
            valid_evaluator: Evaluator receiving the validation metrics.

        Returns:
            int: Number of validation batches run (0 if not enough records).
        """
        goal = goal_dataset[j]
        batch_size = self.flags.sgd_minibatch_size
        num_itr = len(valid_idx[j]) // batch_size
        for itr in range(num_itr):
            self._set_batch_input(goal, valid_idx, j, itr)
            loss, loss_list = self._evaluate_batch(j)
            valid_evaluator.update(loss[j], loss_list, j)
        return num_itr

    def _check_improvement(
        self,
        j: int,
        valid_evaluator: "TprismEvaluator",
        best_valid_loss: List[Optional[Any]],
    ) -> bool:
        """Update the per-goal best validation loss; return True if improved."""
        if (
            best_valid_loss[j] is None
            or best_valid_loss[j] > valid_evaluator.running_loss[j]
        ):
            best_valid_loss[j] = valid_evaluator.running_loss[j]
            return True
        return False

    def _fit(self, input_data: List[InputData]) -> Tuple[TprismEvaluator, TprismEvaluator]:
        """Train with input data, performing train/validation loops.

        Args:
            input_data: Training dataset.

        Returns:
            Tuple[TprismEvaluator, TprismEvaluator]: (train_evaluator, valid_evaluator).
        """
        logger.info("... training phase")
        goal_dataset = build_goal_dataset(input_data, self.tensor_provider)
        logger.info("... training variables")
        for key, param in self.comp_expl_graph.state_dict().items():
            param_logger.debug("%s %s", key, param.shape)
        optimizer = self._build_optimizer()
        best_valid_loss: List[Optional[Any]] = [None for _ in range(len(goal_dataset))]
        best_train_loss: Optional[float] = None
        patient_count = 0
        batch_size = self.flags.sgd_minibatch_size
        logger.info("... splitting data")
        # goal-level split first (coarse): whole goals held out for validation
        train_goal_idx, valid_goal_idx = split_goals(
            goal_dataset, valid_ratio=self.flags.sgd_goal_valid_ratio
        )
        print("train goal:",train_goal_idx)
        print("valid goal:",valid_goal_idx)
        if len(valid_goal_idx) > 0:
            logger.info(
                "... goal-level split: %d training goals / %d held-out goals %s",
                len(train_goal_idx),
                len(valid_goal_idx),
                sorted(valid_goal_idx.tolist()),
            )
        # record-level split within each training goal
        train_idx, valid_idx = split_goal_dataset(
            goal_dataset,
            valid_ratio=self.flags.sgd_valid_ratio,
            valid_goals=valid_goal_idx,
        )
        # early stopping is enabled only when validation batches are available
        has_valid = any(
            len(valid_idx[j]) // batch_size > 0 for j in range(len(goal_dataset))
        )
        if not has_valid:
            logger.info("... no validation data: early stopping is disabled")
        logger.info("... starting training")
        train_evaluator = TprismEvaluator(goal_dataset)
        valid_evaluator = TprismEvaluator(goal_dataset)
        early_stop = False
        for epoch in range(self.flags.max_iterate):
            start_t = time.time()
            train_evaluator.start_epoch()
            valid_evaluator.start_epoch()
            improved = False
            for j in train_goal_idx:
                goal = goal_dataset[j]
                minibatch_logger.debug("%s", goal)
                np.random.shuffle(train_idx[j])
                # training update
                num_itr = len(train_idx[j]) // batch_size
                for itr in range(num_itr):
                    self._set_batch_input(goal, train_idx, j, itr)
                    loss, loss_list = self._evaluate_batch(j)
                    # display_graph(output[j],'graph_pytorch')
                    optimizer.zero_grad()
                    loss[j].backward()
                    optimizer.step()
                    train_evaluator.update(loss[j], loss_list, j)
                # record-level validation within a training goal
                num_itr = self._validate_goal(goal_dataset, valid_idx, j, valid_evaluator)
                # checking improvement of the validation loss
                if num_itr > 0:
                    if self._check_improvement(j, valid_evaluator, best_valid_loss):
                        improved = True
                    logger.info(
                        "[%4d]  %s %s (%2d) %s",
                        epoch + 1,
                        train_evaluator.get_msg("train"),
                        valid_evaluator.get_msg("valid"),
                        patient_count,
                        "*" if improved else "",
                    )
                else:
                    logger.info(
                        "[%4d]  %s",
                        epoch + 1,
                        train_evaluator.get_msg("train"),
                    )
            # goal-level validation: held-out goals are never trained on
            for j in valid_goal_idx:
                minibatch_logger.debug("%s", goal_dataset[j])
                num_itr = self._validate_goal(goal_dataset, valid_idx, j, valid_evaluator)
                if num_itr > 0:
                    if self._check_improvement(j, valid_evaluator, best_valid_loss):
                        improved = True
                    logger.info(
                        "[%4d]  %s (%2d) %s",
                        epoch + 1,
                        valid_evaluator.get_msg("valid"),
                        patient_count,
                        "*" if improved else "",
                    )
            # checkpointing and early stopping (epoch level)
            if has_valid:
                if improved:
                    patient_count = 0
                    if self.flags.model is not None:
                        logger.info(
                            "... saving best model to %s",
                            self.flags.model + ".best.model",
                        )
                        self.save(self.flags.model + ".best.model")
                else:
                    patient_count += 1
                if patient_count >= self.flags.sgd_patience:
                    logger.info(
                        "... early stopping: no improvement in validation loss"
                        " for %d epochs",
                        patient_count,
                    )
                    early_stop = True
            else:
                # no validation data: save the best model based on the training loss
                total_train_loss = float(np.sum(train_evaluator.get_loss()))
                if best_train_loss is None or total_train_loss < best_train_loss:
                    best_train_loss = total_train_loss
                    if self.flags.model is not None:
                        logger.info(
                            "... saving best model to %s",
                            self.flags.model + ".best.model",
                        )
                        self.save(self.flags.model + ".best.model")
            train_time = time.time() - start_t
            logger.info("train time:%s[sec]", train_time)
            train_evaluator.stop_epoch()
            if has_valid:
                valid_evaluator.stop_epoch()
            if early_stop:
                break
        if self.flags.model is not None:
            logger.info("... saving last model to %s", self.flags.model + ".last.model")
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
            logger.warning("[SKIP] skip loading: %s is not found", filename)

    def pred(self, input_data: Optional[List[InputData]] = None, verbose: bool = False) -> Tuple[Any, Any]:
        """Run prediction (inference).

        Args:
            input_data: Dataset or None for no-data inference.
            verbose: Deprecated and ignored; verbosity is controlled by the
                logging level (see `tprism.util.setup_logging`).

        Returns:
            Tuple[Any, Any]: (labels, outputs), usually converted to NumPy.
        """
        if input_data is not None:
            return self._pred(input_data)
        else:
            return self._pred_no_data()

    def _pred_no_data(self) -> Tuple[Optional[Any], Optional[Any]]:
        """Infer without input data.

        Returns:
            Tuple[Optional[Any], Optional[Any]]: (labels, outputs) detached and converted.
        """
        logger.info("... prediction")
        logger.info("... loaded variables")
        for key, param in self.comp_expl_graph.state_dict().items():
            param_logger.debug("%s %s", key, param.shape)
        logger.info("... initialization")
        evaluator = TprismEvaluator()
        ###
        feed_dict: Dict[PlaceholderData, Tensor] = {}
        for embedding_generator in self.embedding_generators:
            if embedding_generator is not None:
                feed_dict = embedding_generator.build_feed(feed_dict, None)
        self.tensor_provider.set_input(feed_dict)
        logger.info("... predicting")
        start_t = time.time()
        goal_inside, loss_list = self.comp_expl_graph.forward()
        loss, output, label = self.loss_obj.call(
            self.graph, goal_inside, self.tensor_provider
        )
        metrics=self.loss_obj.metrics(output, label)
        pred_time = time.time() - start_t
        logger.info("prediction time:%s[sec]", pred_time)
        # train_acc=sklearn.metrics.accuracy_score(all_label,all_output)
        if loss is not None:
            logger.info("loss: %s", torch.sum(loss))
        logger.info("metrics: %s", metrics)
        # 
        label_  = _check_and_detach(label)
        output_ = _check_and_detach(output)
        return label_, output_

    def export_computational_graph(self, input_data: Optional[List[InputData]] = None, verbose: bool = False) -> None:
        """Debug helper: print computational graph paths to stdout.

        Args:
            input_data: Dataset to build batches from, or None.
            verbose: Deprecated and ignored; verbosity is controlled by the
                logging level (see `tprism.util.setup_logging`).
        """
        logger.info("... prediction")
        goal_dataset=None
        if input_data:
            goal_dataset = build_goal_dataset(input_data, self.tensor_provider)
        logger.info("... loaded variables")
        for key, param in self.comp_expl_graph.state_dict().items():
            param_logger.debug("%s %s", key, param.shape)
        logger.info("... initialization")
        if input_data:
            batch_size = self.flags.sgd_minibatch_size
            test_idx = get_goal_dataset(goal_dataset)
        logger.info("... exporting")
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

    def _pred(self, input_data: List[InputData]) -> Tuple[List[Any], List[Any]]:
        """Infer with input data.

        Args:
            input_data: Dataset for prediction.

        Returns:
            Tuple[List[Any], List[Any]]: (labels, outputs) per goal.
        """
        logger.info("... prediction")
        goal_dataset = build_goal_dataset(input_data, self.tensor_provider)
        logger.info("... loaded variables")
        for key, param in self.comp_expl_graph.state_dict().items():
            param_logger.debug("%s %s", key, param.shape)
        logger.info("... initialization")
        batch_size = self.flags.sgd_minibatch_size
        test_idx = get_goal_dataset(goal_dataset)
        evaluator = TprismEvaluator(goal_dataset)
        logger.info("... predicting")
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
                goal_inside, loss_list = self.comp_expl_graph.forward()
                loss, output, label = self.loss_obj.call(
                    self.graph, goal_inside, self.tensor_provider
                )
                if loss is not None:
                    evaluator.update(loss[j], loss_list, j)
                if output is not None:
                    evaluator.update_data(output, label, j)
                else:
                    raise RuntimeError("output is None in prediction")
            ##
            logger.info("%s", evaluator.get_msg("test"))
            metrics = self.loss_obj.metrics(
                np.array(evaluator.output[j]), np.array(evaluator.label[j])
            )
            outputs.append(np.array(evaluator.output[j]))
            labels.append(evaluator.label[j])
        pred_time = time.time() - start_t
        logger.info("test time:%s[sec]", pred_time)
        return labels, outputs
