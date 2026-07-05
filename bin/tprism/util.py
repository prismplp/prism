#!/usr/bin/env python

import logging

import numpy as np
from numpy import ndarray
from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union
from typing import get_origin, get_args, get_type_hints
import h5py

import tprism.expl_pb2 as expl_pb2

import dataclasses

logger = logging.getLogger(__name__)

# functional debug categories (independent of the module/file structure):
#   feed:      feeding data into placeholders/minibatches
#   module:    loading/registering plugins (loss/op) and operator modules
#   minibatch: per-minibatch processing during training/prediction
#   embedding: assigning/retrieving tensors and embeddings for switches
#   graph:     tracing explanation-graph -> computational-graph construction
#   param:     model parameters (e.g. shapes of training variables)
DEBUG_CATEGORIES: Tuple[str, ...] = (
    "feed",
    "module",
    "minibatch",
    "embedding",
    "graph",
    "param",
)


def debug_logger(category: str) -> logging.Logger:
    """Return the logger for debug messages of a functional category.

    Args:
        category: One of DEBUG_CATEGORIES.

    Returns:
        logging.Logger: The logger named "tprism.debug.<category>"; which
        categories are actually shown is controlled by `setup_logging`.
    """
    return logging.getLogger("tprism.debug." + category)


def setup_logging(
    verbose: bool = False,
    quiet: bool = False,
    debug: Optional[List[str]] = None,
) -> None:
    """Configure logging for the tprism package.

    Args:
        verbose: If True, show all debug messages (log level DEBUG).
        quiet: If True, show only warnings and errors (log level WARNING).
            Ignored when verbose is True or debug is given.
        debug: Functional debug categories (see DEBUG_CATEGORIES) whose
            debug messages are shown; debug messages of the other
            categories are suppressed. Enables debug output even without
            verbose.
    """
    if verbose or debug:
        level = logging.DEBUG
        fmt = "[%(levelname)s] %(name)s: %(message)s"
    elif quiet:
        level = logging.WARNING
        fmt = "[%(levelname)s] %(message)s"
    else:
        level = logging.INFO
        fmt = "%(message)s"
    package_logger = logging.getLogger("tprism")
    package_logger.setLevel(level)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(fmt))
    package_logger.handlers.clear()
    package_logger.addHandler(handler)
    package_logger.propagate = False
    # per-category levels: NOTSET inherits the package level (all shown when
    # verbose); restricting to `debug` raises the other categories to INFO
    for category in DEBUG_CATEGORIES:
        category_level = logging.NOTSET
        if debug and category not in debug:
            category_level = logging.INFO
        debug_logger(category).setLevel(category_level)
    if debug:
        unknown = [c for c in debug if c not in DEBUG_CATEGORIES]
        if unknown:
            package_logger.warning(
                "unknown debug categories: %s (available: %s)",
                unknown,
                ", ".join(DEBUG_CATEGORIES),
            )

#[ {"goal_id": <int>, "placeholders": <List[str]>, "records": ndarray} ]
@dataclasses.dataclass
class InputData:
    goal_id: int
    placeholders: List[str] = dataclasses.field(default_factory=list)
    records: ndarray = dataclasses.field(default_factory=lambda: np.array([], dtype=np.int32))


def save_embedding_as_h5(filename: str, train_data: Dict[str,ndarray]={} , test_data:Dict[str,ndarray]={}):
    """ save embedding dataset as h5fs format

    Args:
        filename (str): h5 file name
        train_data (Dict[str,ndarray]): the key is the name of a tensor atom and the value is the numpy array associated with the tensor atom for training phase
        test_data (Dict[str,ndarray]): the key is the name of a tensor atom and the value is the numpy array associated with the tensor atom for test phase
    """
    with h5py.File(filename, "w") as fp:
        if train_data is not None:
            train_group = fp.create_group("train")
            for key, val in train_data.items():
                train_group.create_dataset(key, data=val)
        if test_data is not None:
            test_group = fp.create_group("test")
            for key, val in test_data.items():
                test_group.create_dataset(key, data=val)


def to_string_goal(goal):
    """
    s=goal.name
    s+="("
    s+=",".join([str(arg) for arg in goal.args])
    s+=")"
    """
    s = goal.name
    s += ","
    s += ",".join([str(arg) for arg in goal.args])
    return s



def cast_value(value: Any, typ: Any) -> Any:
    origin = get_origin(typ)
    args = get_args(typ)

    # Optional[T] is regarded as Union[T, None]
    if origin is Union and type(None) in args:
        if value is None:
            return None

        inner_types = [t for t in args if t is not type(None)]
        if len(inner_types) != 1:
            raise TypeError(f"Unsupported Optional type: {typ}")

        return cast_value(value, inner_types[0])

    # List[T] / list[T]
    if origin in (list, List):
        if not isinstance(value, list):
            raise TypeError(f"Expected list, got {type(value).__name__}")

        (elem_type,) = args
        return [cast_value(v, elem_type) for v in value]

    # bool: prolog flags use on/off, JSON configs may use true/false
    if typ is bool and isinstance(value, str):
        v = value.strip().lower()
        if v in ("on", "true", "yes", "1"):
            return True
        if v in ("off", "false", "no", "0"):
            return False
        raise TypeError(f"Cannot cast {value!r} to bool")

    # int: prolog flags such as max_iterate accept `inf`
    if typ is int and isinstance(value, str) and value.strip() in ("inf", "+inf"):
        return 2 ** 31 - 1

    # primitive type
    try:
        return typ(value)
    except Exception as e:
        raise TypeError(f"Cannot cast {value!r} to {typ}") from e


# Values used by the Prolog side (src/prolog/up/flags.pl) to mean "unset":
# `default` is the sentinel of max_iterate / sgd_minibatch_size / sgd_loss /
# sgd_patience / sgd_valid_ratio, and `$disabled` marks a flag disabled by an
# exclusive competitor (e.g. default_sw_a vs default_sw_d).
PROLOG_UNSET_VALUES = ("default", "$disabled", "'$disabled'")

# Renamed prism flags (old name -> new name), mirroring
# $pp_prism_flag_renamed/2 in src/prolog/up/flags.pl; lets flags.json files
# exported by an older PRISM build still be understood.
RENAMED_FLAGS = {
    "sgd_penalty": "sgd_weight_decay",
}

@dataclasses.dataclass(slots=True)
class Flags:
    """Runtime configuration merged from defaults, flags.json, and CLI args.

    The fields commented as "prolog flag" correspond one-to-one to the prism
    flags defined in src/prolog/up/flags.pl: they can be set in a .psm program
    with set_prism_flag/2 and reach this class through the exported flags.json.

    Precedence (weakest first): dataclass defaults < prism flags (flags.json)
    < explicitly given CLI arguments. Prolog-side sentinel values (`default`,
    `$disabled`) are ignored.
    """

    # data / dataset
    dataset: Optional[List[str]] = None
    # intermediate data
    input: Optional[str] = None
    # prolog related
    expl_graph: Optional[str] = None
    # model / vocab
    model: Optional[str] = None
    vocab: Optional[str] = None
    # embeddings
    embedding: Optional[List[str]] = None
    const_embedding: Optional[List[str]]  = None

    # graph / output
    output: Optional[str] = None

    # SGD / training (prolog flags)
    sgd_minibatch_size: int = 1
    max_iterate: int = 10
    sgd_learning_rate: float = 0.01
    sgd_loss: str = "base_loss"
    sgd_patience: int = 3
    sgd_valid_ratio: float = 0.1

    # optimizer (prolog flags)
    sgd_optimizer: str = "adam"
    # weight decay; when a flags.json is given, its value (prolog default: 0.01)
    # takes precedence over this fallback
    sgd_weight_decay: float = 1.0e-10
    sgd_adam_beta: float = 0.9
    sgd_adam_gamma: float = 0.999
    sgd_adam_epsilon: float = 1.0e-8
    sgd_adadelta_gamma: float = 0.95
    sgd_adadelta_epsilon: float = 1.0e-8

    # others
    cycle: bool = False
    verbose: bool = False
    def build(self, args: Any = None, options: Any = None):
        if args is None:
            args_dict = {}
        elif type(args) != dict:
            args_dict = vars(args)
        else:
            args_dict = args
        if options is not None:
            flags = {f.key: f.value for f in options.flags}
        else:
            flags = {}
        self._build(args_dict, flags)

    def __contains__(self, k: str) -> bool:
        return hasattr(self, k) and getattr(self, k) is not None

    def add(self, k: str, v: Any) -> None:
        k = RENAMED_FLAGS.get(k, k)
        if hasattr(self, k):
            hints = get_type_hints(type(self))
            if k not in hints:
                raise AttributeError(f"{k} is not defined")
            if type(v) is str:
                # prolog atoms may be exported with quotes, e.g. '0.005'
                if len(v) >= 2 and v[0] == "'" and v[-1] == "'":
                    v = v[1:-1]
                if v in PROLOG_UNSET_VALUES:
                    return
            setattr(self, k, cast_value(v, hints[k]))

    def _build(self, args_dict, flags) -> None:
        # flags.json first, CLI arguments second: an explicitly given CLI
        # argument overrides a prism flag
        for k, v in flags.items():
            if v is not None:
                self.add(k, v)
        for k, v in args_dict.items():
            if v is not None:
                self.add(k, v)

class TensorInfoMapper():
    def __init__(self, options=None,init_dict={}):
        self.shape={}
        self.type={}
        if options is not None:
            self.shape.update({
                el.tensor_name: [d for d in el.shape] for el in options.tensor_shape
            })
            self.type.update({
                el.tensor_name: el.type for el in options.tensor_shape
            })
    def __repr__(self):
      return "TensorInfoMapper("+self.shape.__repr__()+"  "+self.type.__repr__()+")"


def get_goal_dataset(goal_dataset):
    out_idx = []
    for j, goal in enumerate(goal_dataset):
        all_num = goal["dataset"].shape[1]
        all_idx = np.array(list(range(all_num)))
        out_idx.append(all_idx)
    return out_idx


def split_goal_dataset(goal_dataset, valid_ratio=0.1):
    train_idx = []
    valid_idx = []
    for j, goal in enumerate(goal_dataset):
        ph_vars = goal["placeholders"]
        all_num = goal["dataset"].shape[1]
        all_idx = np.array(list(range(all_num)))
        np.random.shuffle(all_idx)
        train_num = int(all_num - valid_ratio * all_num)
        train_idx.append(all_idx[:train_num])
        valid_idx.append(all_idx[train_num:])
    return train_idx, valid_idx


#
# goal_dataset["placeholders"] => ph_vars
# goal_dataset["dataset"]: dataset
# dataset contains indeces: values in the given dataset is coverted into index
def build_goal_dataset(input_data: List[InputData], tensor_provider):
    goal_dataset: List[Dict[str, Any]] = []

    def to_index(value, ph_name):
        return tensor_provider.convert_value_to_index(value, ph_name)

    to_index_func = np.vectorize(to_index)
    for d in input_data:
        ph_names = d.placeholders
        # TODO: multiple with different placeholders
        ph_vars = [tensor_provider.ph_var[ph_name] for ph_name in ph_names]
        dataset: List[Any] = [None for _ in ph_names]
        goal_data = {"placeholders": ph_vars, "dataset": dataset}
        goal_dataset.append(goal_data)
        for i, ph_name in enumerate(ph_names):
            rec = d.records
            if tensor_provider.is_convertable_value(ph_name):
                debug_logger("feed").debug("%s: values converted to indices", ph_name)
                dataset[i] = to_index_func(rec[:, i], ph_name)
            else:  # goal placeholder
                dataset[i] = rec[:, i]
                logger.warning(
                    "no conversion from values to indices: %s (goal placeholder? records shape=%s)",
                    ph_name,
                    rec.shape,
                )
    for obj in goal_dataset:
        obj["dataset"] = np.array(obj["dataset"])
    return goal_dataset
