# Developer memo: logging and debug output in tprism

This memo describes how console output of the T-PRISM Python package
(`bin/tprism/`, the `tprism` command) is produced and controlled, and what to
do when adding new log messages or a new debug category.

## Overview

All console output of the package goes through Python's standard `logging`
module (no bare `print` for status/diagnostic messages). The configuration is
done once, in `main()`, by `tprism.util.setup_logging`, based on three CLI
options:

| CLI option | Log level | Output format |
|---|---|---|
| (none) | INFO | `%(message)s` (plain, like the historical output) |
| `--verbose` / `-v` | DEBUG (all categories) | `[%(levelname)s] %(name)s: %(message)s` |
| `--debug_verb CAT [CAT ...]` | DEBUG (selected categories only) | same as `--verbose` |
| `--no_verb` | WARNING | `[%(levelname)s] %(message)s` |

Precedence: `--verbose`/`--debug_verb` win over `--no_verb`; when both
`--verbose` and `--debug_verb` are given, `--debug_verb` restricts the debug
output to the listed categories. All three options can also be set through a
`--config` JSON file (logging is configured *after* the config merge).

Everything is attached to the `"tprism"` package logger: `setup_logging` sets
its level, installs a single `StreamHandler` (stderr), and disables
propagation to the root logger, so the package never interferes with a host
application's logging configuration.

## Library use (importing tprism from Python)

INFO-level console logging is the default as soon as the package is imported:
the bottom of `bin/tprism/util.py` calls `setup_logging()` at import time,
guarded by `if not logging.getLogger("tprism").handlers` so that a
configuration made *before* the first tprism import is kept. Since `util` is
imported by every functional module, `from tprism.model import TprismModel`
is enough to get the default.

The level can be changed in two equivalent ways:

```python
from tprism.model import TprismModel

# 1. per-model constructor argument (package-wide effect)
model = TprismModel(flags, tensor_shapes, graph, loss_obj=loss_obj,
                    log_level="debug")   # "debug"/"info"/"warning"/"error"
                                         # or a logging constant (logging.DEBUG)

# 2. explicitly, without constructing a model
from tprism.util import set_log_level, setup_logging
set_log_level("warning")
setup_logging(debug=["feed", "graph"])   # category-restricted debug output
```

`TprismModel(..., log_level=...)` simply calls `tprism.util.set_log_level` at
the beginning of `__init__` (so plugin-loading debug messages of that model
are already affected); `log_level=None` (the default) keeps the current
configuration. `set_log_level` accepts a level name or `logging` constant,
raises `ValueError` for unknown names, and internally reconfigures via
`setup_logging` so that the output format follows the level (verbose format
for DEBUG, quiet format for WARNING+). For category-restricted debug output,
call `setup_logging(debug=[...])` directly.

## Logger structure

Two kinds of loggers are used:

1. **Module loggers** — `logger = logging.getLogger(__name__)` at the top of
   each module. Used for INFO (progress, `[LOAD]`/`[SAVE]`), WARNING, and
   ERROR messages.
2. **Functional debug loggers** — `tprism.util.debug_logger(category)`
   returns `logging.getLogger("tprism.debug." + category)`. **All DEBUG
   messages must go through one of these**, never through a module logger.

Debug categories are *functional*: a category groups messages by what part of
the pipeline they describe, independently of which file emits them (e.g.
`feed` messages come from `model.py`, `embedding_generator.py`, and
`util.py`). The categories are declared in `DEBUG_CATEGORIES` in
`bin/tprism/util.py`:

| Category | What it traces | Emitting modules |
|---|---|---|
| `feed` | feeding data into placeholders/minibatches (feed dicts, value-to-index conversion) | `model.py`, `embedding_generator.py`, `util.py` |
| `module` | loading/registering plugins (`loss/`, `op/`) and operator/torch modules (incl. geotorch availability) | `loader.py`, `expl_graph.py`, `constraint.py` |
| `minibatch` | per-minibatch processing during training (goal dataset dumps per epoch) | `model.py` |
| `embedding` | assigning/retrieving tensors and embeddings for switches (switch⇔tensor association, vocab variables, placeholder graph) | `expl_tensor.py`, `embedding_generator.py` |
| `graph` | explanation-graph → computational-graph construction (tensor equations, operators, distributions) | `expl_graph.py` |
| `param` | model parameters (shapes of training variables) | `model.py` |

Category selection is implemented purely with logger levels — no custom
`Filter` classes. `setup_logging` sets each `tprism.debug.<category>` logger
to `NOTSET` (inherit the package level, i.e. shown when the package level is
DEBUG) or, when `--debug_verb` restricts the output, raises the *unselected*
categories to INFO so their debug records are dropped. In verbose mode the
category is visible in the logger name of each line:

```
[DEBUG] tprism.debug.feed: [feed] tensor_in_ => tensor_in__ph
[INFO] tprism.loader: [LOAD] ./mnist_tmp/mnist.expl.json
```

Unknown category names are rejected by argparse (`choices=DEBUG_CATEGORIES`);
values injected through a `--config` file bypass argparse, so `setup_logging`
additionally warns about unknown categories.

## Level conventions

| Level | Use for | Examples |
|---|---|---|
| INFO | high-level progress a user wants by default | `... training phase`, epoch loss lines, `[LOAD]`/`[SAVE] <file>`, timing |
| DEBUG | anything voluminous or per-item; must use a category logger | parameter shapes, per-switch embedding lookups, per-batch feed shapes, plugin imports |
| WARNING | recoverable/ignorable anomalies | skipped model loading, unknown embedding, no value-to-index conversion |
| ERROR | failures (usually followed by an exception or abort) | unknown file format, einsum failure diagnostics before re-raise |

Guidelines:

- Use `%`-style lazy formatting (`logger.info("[LOAD] %s", filename)`), not
  f-strings, so the message is only rendered when the level is enabled.
- Guard debug blocks whose *argument evaluation* is expensive (array slicing,
  `.sum()`, large reprs) with `if <cat>_logger.isEnabledFor(logging.DEBUG):`.
  See `TprismModel._build_feed` and `SwitchTensorProvider.get_embedding`.
- Do not add level tags like `[WARN]`/`[ERROR]` inside the message text; the
  formatter adds `[%(levelname)s]` in verbose/quiet modes.
- Plain `print` remains only where stdout output *is* the feature:
  `expl_print.py`, `TprismModel.export_computational_graph` (graph dumps),
  `plot/`, `mba.py`, `data/mnist.py`, and `__main__` diagnostic blocks
  (`loader.py`, `tensor_index.py`).

## Backward compatibility notes

- The public methods `TprismModel.build/fit/pred`,
  `SwitchTensorProvider.build`, and `ComputationalExplGraph.forward` still
  accept a `verbose` (and `verbose_embedding`) keyword argument, but it is
  **deprecated and ignored** — verbosity is controlled only by the logging
  level. The parameters are kept so that existing driver scripts do not break;
  do not add new `verbose` parameters.
- `--no_verb` historically did nothing; it is now the quiet mode.
- Messages emitted at *import time* (e.g. the geotorch availability check in
  `constraint.py`) run under the default INFO configuration installed by
  `util.py`, before the CLI applies `--verbose`/`--debug_verb`; DEBUG
  messages are therefore never visible during import.
- `main.py` resolves its logger as
  `logging.getLogger(__name__ if __name__ != "__main__" else "tprism.main")`
  so that `python -m tprism.main` (where `__name__` is `"__main__"`) still
  routes through the `"tprism"` logger.

## How to add a debug message

1. Pick the functional category that matches *what the message is about*
   (not the file it lives in). At the top of the module:

   ```python
   from tprism.util import debug_logger

   feed_logger = debug_logger("feed")
   ```

2. Emit with lazy formatting:

   ```python
   feed_logger.debug("[feed] %s => %s", vocab_name, ph_name)
   ```

3. If building the arguments is costly, guard the block:

   ```python
   if feed_logger.isEnabledFor(logging.DEBUG):
       feed_logger.debug("[feed] shape=%s", dataset[i, idx].shape)
   ```

## How to add a new debug category

1. Add the name to `DEBUG_CATEGORIES` in `bin/tprism/util.py` and document it
   in the comment above the tuple (and in the table in this memo).
2. Nothing else is required: `main.py` derives the `--debug_verb` choices and
   help text from `DEBUG_CATEGORIES`, and `setup_logging` iterates over it to
   set per-category levels.
3. Verify: run an example with `--debug_verb <new-category>` and confirm only
   that category's DEBUG lines appear, e.g.

   ```sh
   cd exs/tensor/mlp1
   tprism train --input ./mnist_tmp/mnist --embedding ./mnist/mnist.h5 \
       --sgd_loss ce --max_iterate 1 --sgd_minibatch_size 100 \
       --debug_verb <new-category> 2>&1 | grep DEBUG
   ```

## File map

| Role | Location |
|---|---|
| `setup_logging`, `set_log_level`, `DEBUG_CATEGORIES`, `debug_logger`, import-time default (INFO) | `bin/tprism/util.py` |
| CLI options (`--verbose`, `--debug_verb`, `--no_verb`) | `bin/tprism/main.py` (`main()`) |
| `log_level` constructor argument | `bin/tprism/model.py` (`TprismModel.__init__`) |
| `feed`/`minibatch`/`param` emitters | `bin/tprism/model.py` |
| `module` emitters (plugin loading) | `bin/tprism/loader.py`, `bin/tprism/constraint.py` |
| `embedding` emitters | `bin/tprism/expl_tensor.py`, `bin/tprism/embedding_generator.py` |
| `graph` emitters | `bin/tprism/expl_graph.py` |
| Intentional `print` (stdout output is the feature) | `bin/tprism/expl_print.py`, `bin/tprism/plot/`, `bin/tprism/mba.py`, `export_computational_graph` in `model.py` |
