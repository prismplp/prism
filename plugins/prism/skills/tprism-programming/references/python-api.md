# T-PRISM from Python (Jupyter, Colab)

The Python API does what `tprism train|test` does, but keeps the model in memory, so you can set flags, register
custom operators and losses, read goal values and parameters, and plot. This is how `T_PRISM_tutorial.ipynb` works.

Contents: 1 setup · 2 the Prolog part from Python · 3 workflow · 4 flags · 5 what fit/pred return · 6 test phase ·
7 placeholders · 8 goal values and parameters · 9 custom losses · 10 custom operators · 11 constrained tensors ·
12 embedding files · 13 drawing the graph · 14 logging

Verified on 2026-09-30 (Ubuntu 24.04, prebuilt `prism_tprism_pre_linux_ubuntu24`, Python 3.14, torch 2.13,
protobuf 7.36, geotorch; the tprism code is the same as at commit 169a258 used by the tutorial, apart from the
`sgd_loss` parser). "Verified" below means the snippet was run and the stated result observed.

## 1. Setup

Colab, as in the tutorial:
```sh
!wget "https://github.com/prismplp/prism/releases/download/v2.4.2a(T-PRISM)-prerelease/prism_linux_dev4colab.auto.zip"
!unzip -q -o prism_linux_dev4colab.auto.zip          # prism/bin/upprism
!pip install "git+https://github.com/prismplp/prism.git#egg=t-prism&subdirectory=bin"
!pip install "git+https://github.com/prismplp/pyprism.git"   # only for PrismEngine (section 2)
```
The tutorial also sets `%env PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`; it was not needed with protobuf 7.36.
Optional packages: `geotorch` (section 11), `networkx` + `matplotlib` or `pyvis` (section 13).
Other machines: install-linux and programming-pyprism skills.

## 2. Running the Prolog part from Python

Either write a `.psm` file and run `subprocess.run(["upprism", "model.psm"], check=True)` (`prism_main/1` is called),
or use PyPRISM:
```python
from pyprism import PrismEngine
engine = PrismEngine(bin_path="prism/bin")
engine.set_db("""
tensor_atom(rel1,[7,7]).
tensor_atom(rel2,[7,7]).
goal :- tensor(rel1,[i,j]), tensor(rel2,[j,k]).
""")
engine.query("save_expl_graph('data04/data.expl.json','data04/data.flags.json',[goal])")
# returns (output lines, 'yes'); look for the '[SAVE:json] ...' lines
```
`set_db` replaces the whole program, and every `query` runs a new PRISM process, so nothing survives between
queries except files. Create output directories first (`mkdir -p data04`). `probf(Goal)` works as a query and
prints the tensor equation (tutorial §12).

## 3. Workflow

```python
from tprism.loader import load_explanation_graph
from tprism.model import TprismModel
from tprism.loss import CE        # also CE_pl, NLL, MSE, PreferencePair, RMSEPair, BaseLoss

graph, tensor_shapes, flags = load_explanation_graph("d/train.expl.json", "d/train.flags.json")
flags.embedding = ["d/mnist.h5"]      # a list; .h5 files or .npy.json files (section 12)
flags.vocab = "d/vocab.pkl"           # written by build(load_vocab=False), read by build(load_vocab=True)
flags.model = "d/model"               # fit() writes d/model.best.model and d/model.last.model
flags.sgd_learning_rate = 0.01
flags.max_iterate = 100
model = TprismModel(flags, tensor_shapes, graph, loss_cls=CE)
model.build(input_data=None, load_vocab=False, embedding_key="train")   # "train"/"test" = group in the embedding file
evaluator = model.fit()
labels, outputs = model.pred()        # (labels, outputs): the tutorial's `loss, out = model.pred()` is misnamed
```
- Change `flags` after `load_explanation_graph` (which fills them from `flags.json`) and before `build`; `flags.model`
  only has to be set before `fit`. With `flags.model = None` nothing is saved.
- `loss_cls=None` means `BaseLoss`: no loss, prediction only (it logs `loss is not implemented`). A loss that needs
  parameters can be given as an object: `TprismModel(..., loss_obj=CE_pl(["$placeholder2$"]))`.

## 4. Flags that matter

Values come from `flags.json`, i.e. from the PRISM flags (`set_prism_flag/2` in the program), and then from what you set in Python.

| Flag | Default | Note |
|---|---|---|
| `sgd_learning_rate` | 0.0001 | small; the tutorial uses 0.001–0.05 |
| `max_iterate` | 10 | epochs |
| `sgd_optimizer` | `adam` | `sgd`, `adadelta` |
| `sgd_weight_decay` | **0.01** | L2 on every parameter. Models whose value is a product of several small tensors collapse to 0 (tutorial §07: loss stuck at 0.135, the RMSE of the zero tensor). Set `0` (verified: 0.135 → 0.057). |
| `sgd_minibatch_size` | 1 | only used with placeholder data |
| `sgd_valid_ratio` | 0.1 | rows of placeholder data held out for early stopping |
| `sgd_goal_valid_ratio` | 0 | goals held out for early stopping when there is no placeholder data (e.g. `0.1`: "goal-level split: 45 training goals / 5 held-out goals") |
| `sgd_patience` | 3 | epochs without validation improvement before stopping |
| `cycle` | False | cyclic programs (`--cycle`) |

## 5. What `fit` and `pred` return

- `fit()` without data returns one evaluator; `fit(input_data=...)` returns `(train_evaluator, valid_evaluator)`.
  `evaluator.loss_history[0]` holds one loss per epoch: the mean over goals in current tprism (2.30 = ln 10 for an
  untrained 10-class `CE`). The logs printed in the tutorial (e.g. `train-loss: 230.163` for 100 goals) are sums
  from an older version, so don't compare the numbers directly.
- `pred()` returns `(labels, outputs)`:

| Loss | labels | outputs |
|---|---|---|
| `CE` | array (N,) of the goals' first arguments | (N, C) goal values |
| `RMSEPair` | None | (#goal groups, 2, …) (verified `(1, 2, 7, 7)`) |
| none (`BaseLoss`) | None | all goal values stacked: **every goal must have the same shape**, else `RuntimeError: stack expects each tensor to be equal size` (use section 8 instead) |
| any, `pred(input_data=...)` | list with one entry per goal pattern | list per pattern, e.g. `[(100,)]` and `[(100, 10)]` |

`pred(input_data=...)` also prints a line `{}` per minibatch (a stray `print` in `model.py`).

## 6. Test phase: a new graph with trained parameters

```python
graph, tensor_shapes, flags = load_explanation_graph("d/test.expl.json", "d/test.flags.json")
flags.embedding = ["d/mnist.h5"]        # the flags are fresh: set the embedding again
flags.vocab = "d/vocab.pkl"             # the vocab written by the training build
model = TprismModel(flags, tensor_shapes, graph, loss_cls=CE)
model.build(input_data=None, load_vocab=True, embedding_key="test")
model.load("d/model.best.model")        # pred() never loads parameters by itself
labels, outputs = model.pred()
```
Tutorial §11 omits the embedding and `load`: every test image then gets the same output (verified on synthetic
data: 1 distinct output, accuracy 0.06; with the three lines above, accuracy 1.00).
`model.load` warns `[SKIP] skip loading` and continues if the file does not exist; check the log.

## 7. Placeholders from Python

```prolog
prism_main([]) :-
    P = [output(_,_)],                                     % label first: CE_pl reads $placeholder1$
    load_clauses('train.dat',G1), save_placeholder_goals('d/ph_train',npy,P,G1),
    load_clauses('test.dat',G2),  save_placeholder_goals('d/ph_test',npy,[output(_,_)],G2),   % a FRESH pattern
    save_expl_graph('d/ph.expl.json','d/ph.flags.json',P).
```
`save_placeholder_goals` binds the pattern's variables to `'$placeholder1$'`, … Reusing `P` for the test data writes
`{"placeholders":[]}` and a (0, 0) array without any error, and `build` later fails with `KeyError: '$placeholder2$'` (verified).

```python
from tprism.loader import load_input_data
from tprism.loss import CE_pl
train_data = load_input_data(["d/ph_train/placeholder.npy.json"])      # npy: pass the .json in the directory
flags.sgd_minibatch_size = 30; flags.sgd_valid_ratio = 0.2
model = TprismModel(flags, tensor_shapes, graph, loss_cls=CE_pl)
model.build(input_data=train_data, load_vocab=False, embedding_key="train")
train_ev, valid_ev = model.fit(input_data=train_data)                  # early stopping on the held-out rows

test_data = load_input_data(["d/ph_test/placeholder.npy.json"])
# new TprismModel from the same graph files, flags.embedding/vocab set again, then:
model2.build(input_data=test_data, load_vocab=True, embedding_key="test")
model2.load("d/model.best.model")
labels, outputs = model2.pred(input_data=test_data)                   # lists, one entry per goal pattern
```
Verified: early stop after 13 epochs, test accuracy 1.00, labels in the order of `test.dat`. The log line
`no conversion from values to indices: $placeholder1$ (goal placeholder? ...)` is normal for the label placeholder.

## 8. Goal values and learned parameters

```python
def goal_values(model):
    feed = {}
    for eg in model.embedding_generators:          # what pred() does before forward()
        feed = eg.build_feed(feed, None)
    model.tensor_provider.set_input(feed)
    goal_inside, _ = model.comp_expl_graph.forward()
    return {(g.node.goal.name, tuple(g.node.goal.args)): goal_inside[g.node.sorted_id].inside.detach().numpy()
            for g in model.graph.goals}
```
- Calling `comp_expl_graph.forward()` right after `build()` in a program with embeddings raises `KeyError:
  <tprism.placeholder.PlaceholderData ...>`; the tutorial only calls it after `fit()`, which sets the feed. Use the helper.
- Goal arguments are strings (`('row', ('3',))`).
- **Axis order**: the free indices of a goal's value are sorted by index name, not by their position in the atom
  (verified: `t :- tensor(m,[j,i]).` returns mᵀ; the tensor-train goal of §07 comes out as `[a1,a2,a3,a4]` although `a4` is written first).
- Parameters: `model.comp_expl_graph.state_dict()` or `torch.load("d/model.best.model")`. Keys are
  `SwitchTensor.make_var_name("tensor(<atom>)")` (`from tprism.expl_tensor import SwitchTensor`): every run of `( ) [ ] , ' $`
  becomes one `_`, e.g. `rel1` → `tensor_rel1_`, `v(a4)` → `tensor_v_a4_`, `in(3)` → `tensor_in_3_`.

## 9. A custom loss in Python (no package edit)

Pass any `BaseLoss` subclass as `loss_cls`. Example: the goal's value is a probability vector and the label is the
goal's first argument; minimise −log p[label]. This made the MNIST-addition model learn where the tutorial's
softmax + `CE` did not (tutorial-recipes.md §12).
```python
import numpy as np, torch
from tprism.loss.base import BaseLoss

class LabelNLL(BaseLoss):
    def call(self, graph, goal_inside, tensor_provider):
        loss, out, lab = [], [], []
        for rank_root in graph.root_list:              # one entry per goal group
            ls = []
            for el in rank_root.roots:
                sid = el.sorted_id
                y = int(graph.goals[sid].node.goal.args[0])
                p = goal_inside[sid].inside
                ls.append(-torch.log(p[y] + 1e-10)); out.append(p); lab.append(y)
            loss.append(torch.stack(ls).mean())
        return torch.stack(loss), torch.stack(out), torch.LongTensor(lab)   # (loss per group, output, label)
    def metrics(self, output, label):
        o = output.detach().numpy() if torch.is_tensor(output) else np.asarray(output)   # detach, or pred() fails
        return {"*accuracy": float((o.argmax(1) == np.asarray(label)).mean())}
```
For the `tprism` command, put the class in a module under `tprism/loss/` instead (pipeline.md §5).

## 10. Custom operators in Python

```python
from torch import nn
from tprism.op.base import BaseOperator

class MyMlp(BaseOperator, nn.Module):               # nn.Module: its parameters are trained with the model
    def __init__(self, parameters):                  # operator(my_mlp(a,b)) gives ['a','b'] (strings)
        nn.Module.__init__(self)
        self.net = nn.Sequential(nn.Linear(784, 120), nn.ReLU(), nn.Linear(120, 10))
    def call(self, x):                               # x = value of the rest of the clause body
        return self.net(x)
    def get_output_template(self, input_template):   # index symbols of the result
        return input_template
    def get_output_shape(self, input_shape):         # needed when the shape changes
        return tuple(list(input_shape[:-1]) + [10])

model = TprismModel(flags, tensor_shapes, graph, loss_cls=CE)
model.operator_loader.set_cls("my_mlp", MyMlp)      # before build(); the key is the name used in operator/1
model.build(input_data=None, load_vocab=False, embedding_key="train")
```
with `output(Y,X) :- operator(my_mlp), tensor(get(in,X),[k]).` Verified on synthetic data (train accuracy 1.00; the
four `nn.Linear` tensors appear in `state_dict`).
- Arguments are strings; a quoted atom keeps its quotes: `operator(text_embed('The capital of China is Beijing.'))`
  gives `["'The capital of China is Beijing.'"]`.
- The template entries are `TensorIndexRef` objects (`.symbol`), with a leading `b` axis when placeholders are used.
  The tutorial's `input_template[0] == "b"` compares an object with a string and is always False; use
  `getattr(t, "symbol", t) == "b"`. An operator that drops axes returns the surviving symbols (the tutorial's
  `ReadoutBlock` returns `input_template[2:]` after flattening a 3×4×4 feature map).
- For the `tprism` command, put the class in a module under `tprism/op/` (pipeline.md §5).

## 11. Constrained tensors (geotorch)

`tensor_atom(u,[7,7],orthogonal).` The types handled by `bin/tprism/constraint.py` are `symmetric`, `skew`, `sphere`,
`orthogonal`, `almost_orthogonal`, `grassmannian`, `low_rank(R)`, `fixed_low_rank(R)`, `invertible`, `sln`,
`positive_definite`, `positive_semidefinite`, `positive_semidefinite_low_rank(R)`, `positive_semidefinite_fixed_low_rank`
(only `orthogonal` verified: u·uᵀ = I from the first epoch). `tprism.constraint` imports geotorch once; if you
`pip install geotorch` in a running notebook, run `importlib.reload(tprism.constraint)` (tutorial §06) or restart the kernel.

## 12. Embedding files written from Python

An `.npy` file plus a JSON description (the same form `save_embedding_from_pattern(..., npy)` writes):
```json
{"filename": "data05/my_embedding.npy", "group": "train", "name": "tensor_relx_", "shape": [7, 7]}
```
HDF5: groups `train`/`test`, one dataset per tensor name. The name a tensor atom reads (verified):

| Atom in the program | Dataset |
|---|---|
| `relx` | `tensor_relx_` |
| `in(3)`, no placeholders | `tensor_in_3_` (one dataset per sample; the tutorial's `individual=True`) |
| `get(in,3)` | row 3 of `tensor_in_` (slice along the first axis; the tutorial text says "column", the output is a row) |
| `in(X)` with `X` a placeholder | row X of `tensor_in_` |

A dataset that is not found is not an error: the atom silently becomes a random trainable parameter (verified: `in(X)`
without placeholders and only `tensor_in_` in the file gave accuracy 0.10 and parameters `tensor_in_0_`, `tensor_in_100_`, …).
Look at `sorted(model.comp_expl_graph.state_dict())` for data names that should not be there.

## 13. Drawing the computational graph

```python
import matplotlib.pyplot as plt
from tprism.plot.graph import plot_and_or_graph             # networkx + matplotlib
goal_inside, _ = model.comp_expl_graph.forward(dryrun=True)  # symbolic, needs no feed
plt.figure(figsize=(12, 12)); plot_and_or_graph(goal_inside); plt.show()
```
`from tprism.plot.graph_pyvis import plot_and_or_graph` draws an interactive version with pyvis (writes `nx.html`).

## 14. Logging

Per-epoch lines go through `logging`. Quieter: `TprismModel(..., log_level="warning")` or
`logging.getLogger("tprism").setLevel(logging.WARNING)`. `verbose=` arguments are ignored.
