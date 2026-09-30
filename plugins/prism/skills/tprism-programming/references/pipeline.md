# T-PRISM pipeline reference: predicates, files, CLI, Python extension points

Contents: 1 compile-time predicates · 2 intermediate files · 3 `tprism` options · 4 flags · 5 custom operators and losses ·
6 embeddings · 7 typical `run.sh`

## 1. Compile-time predicates (run by `upprism`, defined in `src/prolog/up/tensor.pl`)

| Predicate | Purpose |
|---|---|
| `tensor_atom(Pattern,Shape)` (also `tensor_atom(Pattern,Shape,Type)`) | declare shape; `Type` selects a constrained parameterisation (`symmetric`, `orthogonal`, `low_rank(r)`, `positive_definite`, … — needs the optional `geotorch` package; see `bin/tprism/constraint.py`) |
| `index_atoms([i,j,..])` | pool of index symbols (see gotcha T4) |
| `save_expl_graph(Goals)` / `(ExplFile,FlagFile)` / `(ExplFile,FlagFile,Goals)` / `(ExplFile,FlagFile,ExplMode,FlagMode,Goals)` | build the explanation graph for `Goals` (ground goals, or placeholder patterns) and write graph + flags. Modes `json` (default), `pb`, `pbtxt`. Defaults `expl.json`, `flags.json` |
| `save_flags(File)`, `save_flags(File,Mode)` | flags only (e.g. for the test phase) |
| `save_placeholder_goals(Patterns,Goals)` / `(File,Patterns,Goals)` / `(File,Mode,Patterns,Goals)` | pattern-match `Goals` against `Patterns`, replace pattern variables by placeholder atoms `$placeholderN$`, write the integer table. Default file `data.h5`, default mode hdf5 (**not available in the standard build**: use `json` ⇒ file `<name>1_0`, or `npy` ⇒ directory with `placeholder.npy.json` + `placeholder_N.npy`) |
| `save_placeholder_data(Placeholders,Data)` (+ file/mode variants) | low-level version: lists of placeholders and integer rows |
| `save_embedding_from_pattern(Vars,Pattern,tensor(Name),FileBase[,Mode])` | 0/1 tensor from facts matching `Pattern` (axes = `Vars`); default hdf5 (**not in the standard build**); with mode `npy` writes `FileBase.npy`, `FileBase.npy.json` (pass this to `--embedding`) and `FileBase.txt` (axis-index-label table) |
| `save_embedding_from_pattern_with_value(Vars,Val,Pattern,Target,FileBase[,Mode])` | same but with real-valued entries `Val` |
| `load_clauses(File,Goals)` | read a file of facts as a goal list |
| `set_index_range/2`, `atoms_concat/2`, `transpose/2`, `unique/2` | small helpers |
| `set_prism_flag(Name,Value)` | flags (see §4); saved into `flags.json` |
| `prism_main([Mode])` | entry: `upprism model.psm train` → `prism_main([train])`. Arguments are atoms (`parse_atom/2`) |

`upprism` also accepts everything from classic PRISM (`values`, `msw`, …) because T-PRISM programs are PRISM programs; `probf/1` prints tensor equations.

## 2. Intermediate files

| File | Content |
|---|---|
| `expl.json` (or `.pb/.pbtxt`) | explanation graph (schema `src/c/external/expl.proto`) |
| `flags.json` | all PRISM flags + tensor shape list |
| `data.h5/.json/...` | placeholder substitutions: for each goal pattern group, an integer matrix (#samples × #placeholders) |
| embedding file(s) | fixed/initial tensors; H5 datasets named `tensor_<atom name>_` in groups `train` and `test`, or npy |
| vocab file | written by `tprism train` if absent (maps symbols to integer positions) |

Reading data H5 in Python: `h5py.File(f)` → groups per goal id → dataset `"data"` with attribute `placeholders`.

## 3. `tprism <train|test>` options (from `bin/tprism/main.py`)

- Locations: `--input P` (alias `-I`): if `P` is a directory the files are `P`+`expl.json` etc. — the code concatenates without adding `/`, so write `--input tmp/` (verified failure with `tmp`); otherwise `P.expl.json`, `P.flags.json`, `P.model`, `P.vocab.pkl`. Deprecated `--intermediate_data_prefix P` concatenates directly (`P`+`expl.json`). Or give `--expl_graph`, `--flags`, `--model`, `--vocab` separately. `--config file.json` supplies arguments. Modes: `train`, `test` (alias `pred`), `prepare`.
- Data: `--dataset FILE...` (or deprecated `--data`) placeholder data; `--embedding FILE...` and `--const_embedding FILE...` tensors (the latter not trained).
- Training: `--sgd_loss NAME[(params)]` (a Prolog-like term; current tprism accepts any arguments, e.g. `'ce_pl($placeholder2$)'` quoted for the shell, while up to a0ea215 only numbers), `--max_iterate N` (alias `--epoch`), `--sgd_minibatch_size N`, `--sgd_learning_rate x`, `--sgd_optimizer {sgd,adam,adadelta}`, `--sgd_weight_decay x`, `--sgd_patience N`, `--sgd_valid_ratio x`, `--sgd_goal_valid_ratio x`, Adam/Adadelta hyper-parameters (`--sgd_adam_beta/gamma/epsilon`, `--sgd_adadelta_gamma/epsilon`).
- Cyclic programs: `--cycle`.
- Device: `--cpu` or `--gpu 0,2`.
- Output: `--output FILE` (predictions in test; default `./output.pkl`), `--draw_graph FILE`.
- Logging: `--no_verb` (warnings only), `--verbose/-v` (debug), `--debug_verb feed module minibatch embedding graph param`.

Any option marked "[prolog flag]" can also be set in the `.psm` with `set_prism_flag/2` (`sgd_minibatch_size`, `max_iterate`/`epoch`, `sgd_learning_rate`, `sgd_loss`, ...); command-line values win.

## 4. Flags

`set_prism_flag(sgd_minibatch_size,N)`, `set_prism_flag(max_iterate,N)`, `set_prism_flag(sgd_learning_rate,X)`, `set_prism_flag(sgd_loss,ce)`, `set_prism_flag(error_on_cycle,off)`. Values propagate through `flags.json`; note floats in compound values (`ce(0.1)`) are exported with 6 decimals only (developer notes in `doc/devel_tprism/flag.md`).

PRISM-side defaults that end up in `flags.json`: `sgd_learning_rate` 0.0001, `sgd_optimizer` adam, **`sgd_weight_decay` 0.01** (L2; set 0 for
products of many small tensors, gotchas T19), and `default` for the others (then the Python defaults apply: `max_iterate` 10,
`sgd_minibatch_size` 1, `sgd_patience` 3, `sgd_valid_ratio` 0.1, `sgd_goal_valid_ratio` 0). `sgd_goal_valid_ratio` holds out whole goals
for early stopping when there is no placeholder data.

## 5. Custom operators and losses (Python)

Operators (`bin/tprism/op/*.py`; any file in that package is scanned). The Prolog name is the snake_case of the class name (`class Softplus` → `operator(softplus)`).

```python
import torch
from tprism.op.base import BaseOperator

class Softplus(BaseOperator):
    def __init__(self, parameters):        # list of argument strings from operator(name(args))
        pass
    def call(self, x):                      # x: tensor produced by the rest of the clause body
        return torch.nn.functional.softplus(x)
    def get_output_template(self, input_template):
        return input_template               # which index axes survive (unchanged for elementwise ops)
```
`get_output_template` returns the list of output index symbols; operators that change the axes (like `reindex`) must compute them. Note that tensors carry a leading batch axis `"b"` when placeholders are used; `Reindex` shows how to keep it.

Losses (`bin/tprism/loss/*.py`), subclass `BaseLoss`, implement `call(graph, goal_inside, tensor_provider)` returning `(loss, output, label)` and optionally `metrics(output,label)`. Built-in names: `nll`, `ce`, `ce_pl`, `mse`, `preference_pair`, `rmse_pair` (snake case of class names in `standard_loss.py`; numeric parameters via `name(0.5)`, e.g. the hinge margin gamma of `preference_pair`, default 1.0). `ce` takes the label from the goal's first argument; `ce_pl` from placeholder `$placeholder1$`; `preference_pair` expects goals in `[Pos,Neg]` pairs and minimises relu(score(Neg)−score(Pos)+γ). `goal_inside[sorted_id].inside` is the tensor value of a goal; `graph.root_list[k].roots` are the observed goals.

From Python there is no need to touch the package: register an operator with `model.operator_loader.set_cls("name", Cls)` before
`build`, and pass a loss class as `TprismModel(..., loss_cls=Cls)` (python-api.md §9–10; operators may also subclass `torch.nn.Module`
so that their own parameters are trained, and should implement `get_output_shape` when they change the shape).

Where these directories live after `pip install`: inside the installed `tprism` package (`python -c "import tprism,os;print(os.path.dirname(tprism.__file__))"`). For a project-local extension, work on an editable install of the repo (`pip install -e bin/`) so the files under `bin/tprism/op` and `bin/tprism/loss` are yours to edit.

## 6. Embeddings from Python

For large numeric inputs, generate the embedding file with h5py/numpy instead of `save_embedding_from_pattern` (see `exs/tensor/mlp*/mnist/build_mnist.py`, `transitive_closure02`, and tutorial §08): group `train` (and `test`), one dataset per tensor. The dataset name is `SwitchTensor.make_var_name("tensor(<atom>)")` — every run of `( ) [ ] , ' $` becomes one `_` — and depends on how the atom is written (verified 2026-09-30):

| Atom | Dataset |
|---|---|
| `relx` | `tensor_relx_`, the tensor's shape |
| `in(3)`, no placeholders | `tensor_in_3_`, one dataset per sample |
| `get(in,3)` | `tensor_in_`, shape `[num_samples, ...]`; row 3 is used |
| `in(X)` with `X` a placeholder | `tensor_in_`, shape `[num_samples, ...]`; row X is used |

A name that is not found silently becomes a random trainable parameter. For `.npy` data write the array plus a JSON file
`{"filename": "x.npy", "group": "train", "name": "tensor_relx_", "shape": [7,7]}` and pass the JSON file (python-api.md §12).
Use `--debug_verb embedding feed` to see how names are matched to tensors.

## 7. Typical `run.sh`

```sh
#!/bin/sh
cd `dirname $0`
mkdir -p model_tmp
upprism model.psm train                    # writes model_tmp/*.json + placeholder data (json: <name>1_0; hdf5 needs a USE_H5 build)
upprism model.psm test
tprism train --input model_tmp --dataset model_tmp/data.train.h5 --embedding emb.h5 \
             --sgd_loss ce --max_iterate 100 --sgd_minibatch_size 256 --sgd_learning_rate 0.01 --cpu
tprism test  --input model_tmp --dataset model_tmp/data.test.h5  --embedding emb.h5 \
             --sgd_loss ce --output pred.npy --cpu
```

## 8. Environment

See `known-issues.md` and `Dockerfile` for a verified way to build PRISM + T-PRISM (Linux amd64) and the patch for two bugs in `bin/tprism`.
For notebooks and Colab, and for everything the command line does from Python (`load_explanation_graph`, `TprismModel`), see `python-api.md`.
