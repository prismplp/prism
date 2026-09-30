---
name: tprism-programming
description: How to write, build and train T-PRISM (Tensorized PRISM, prismplp/prism) programs, where Prolog atoms denote tensors combined by einsum and trained with PyTorch. Covers .psm programs (tensor/2, tensor_atom/2, operator/1, subgoal/2, get/2, placeholders, embeddings, save_expl_graph), `tprism train/test`, and the Python API (load_explanation_graph, TprismModel fit/pred, custom operators and losses) in Jupyter or Colab, also via PyPRISM's PrismEngine. Use it whenever the user mentions T-PRISM, TPRISM, tprism, TprismModel or T_PRISM_tutorial, or wants tensor decomposition, DistMult, an MLP or CNN, MNIST addition, a Markov chain or transitive closure written in Prolog with tensors, or to port a Prolog, ProbLog or PRISM model to T-PRISM. Read it before writing T-PRISM code, because its semantics (sum over Prolog solutions, einsum over index names, operator/1 scope, compile time vs run time) differ from Prolog, ProbLog and classic PRISM, and mistakes are silent. For classic PRISM (msw/values) use prism-programming.
---

# Writing T-PRISM programs

T-PRISM keeps the *syntax and search* of Prolog/PRISM but changes what atoms **mean**: an atom denotes a
**tensor** instead of a truth value or a probability. A program is compiled by Prolog (via `upprism`) into an
**explanation graph** — a set of tensor equations — and `tprism` (Python + PyTorch) trains/evaluates it, either as the
`tprism` command or from Python (Jupyter/Colab).
Repo: https://github.com/prismplp/prism (`exs/tensor/`, `T_PRISM_tutorial.ipynb`, `doc/manual/tprism_manual.tex`, `bin/tprism/`).
Assumes PRISM is installed (`upprism` on PATH) and `pip install "git+https://github.com/prismplp/prism.git#egg=t-prism&subdirectory=bin"` + PyTorch, numpy, h5py, scikit-learn, protobuf.
The release tarballs are Linux-only; on macOS use Docker (`references/Dockerfile`, linux/amd64). Colab setup: `references/python-api.md` §1.

**Read `references/known-issues.md` first.** In `bin/tprism` up to at least commit a0ea215 (unchanged since 169a258), `subgoal/2` crashes `tprism` and `--sgd_loss nll`
minimises the probability instead of maximising it; `references/tprism-fixes.patch` fixes both. The standard build also has no HDF5, so `.h5` defaults of the Prolog side fail — use json/npy.

## 1. The mental model (read this twice)

Two phases with a hard wall between them:

1. **Compile time (Prolog, `upprism model.psm [mode]`)**: the program is run exhaustively (tabled) for the goals you
   give to `save_expl_graph` (any of its /1,/2,/3,/5 forms). Everything symbolic happens here: recursion, arithmetic, `member/2`, comparisons,
   `findall`, file loading. Output: `expl.json` (graph), `flags.json`, optional placeholder data and embeddings.
2. **Run time (PyTorch, `tprism train|test` or `TprismModel`)**: evaluates the tensor equations, learns the tensors marked as parameters by SGD.

Semantics of the graph:

| Construct | Meaning |
|---|---|
| Several clauses (or several Prolog solutions) for a head | **sum** of their tensors (disjunction rule). Prolog nondeterminism such as `member(Y,[0,1,2])` becomes a *sum over all solutions*, not "first solution". |
| Conjunction of tensor atoms in one body | **Einstein summation** over *index symbols*: an index name appearing in ≥2 atoms is contracted (summed); an index appearing once stays *free* and becomes an axis of the result. **The result's axes are ordered by index name**, not by position (`tensor(m,[j,i])` alone is mᵀ). |
| `tensor(Name,[i,j])` (aliases `matrix/2`, `vector/2`) | a learnable (or loaded) tensor addressed by ground term `Name`, indexed by *atoms* `i`,`j` |
| An integer in an index list, `tensor(x,[i,1,0])` | selects that position; the axis disappears |
| `tensor(get(x,N),[i])` | row `N` of the data tensor `x` (slice along the first axis) |
| Non-tensor goal (arithmetic, `member`, user predicate with no tensors, comparison) | proved by Prolog at compile time, contributes scalar **1**; only its *bindings* matter |
| `operator(f)` in a body | apply non-linear op `f` (`sigmoid`, `relu`, `softmax`, `min1`, custom) to the tensor computed from the **remaining goals of that clause body**; several operators compose in written order |
| `subgoal(G,[k,l])` | equal to `G` logically; in the tensor equation it **renames** G's free indices to `k,l` (positionally) |
| PRISM's `msw/2` | not used directly; simulate it (`simulated_msw`, softmax + one-hot) or use `tensor/2` |

If you know ProbLog: there are no probabilities, no worlds, and no "query/evidence". If you know Prolog:
a predicate is a *tensor-valued function of its ground arguments*, and the search is only there to unfold the structure.
If you know PRISM: `msw(sw(X),V)` ⇒ `tensor(...)`, `sum over outcomes` ⇒ shared index name, exclusivity is not required
because sums are just sums (but normalisation is **your** job: use `softmax`).

## 2. Program skeleton (DistMult with placeholders and a ranking loss, adapted from `exs/tensor/distmult_sample02`, which I ran successfully)

```prolog
%% 1. declarations: shape of every tensor atom (name may contain `_` wildcards)
tensor_atom(v(_),[20]).            % v(anything) is a 20-vector
tensor_atom(r(_),[20]).
% index_atoms([i,j]).              % only needed when an index list is passed as a *variable* (gotcha 3)

%% 2. model: tensor equations
rel(S,R,O) :- tensor(v(S),[i]), tensor(v(O),[i]), tensor(r(R),[i]).   % sum_i v_S[i] * v_O[i] * r_R[i]

%% 3. data preparation (compile time, ordinary Prolog): goals come as [Positive,Negative] pairs
prism_main([]) :-
    load_clauses('sample.dat',Gs),                       % Gs = [rel(0,1,2), ...]
    generate_preference_pair(Gs,GoalPairList),           % your own utility, see references/examples.md §1
    GoalPlaceholder = [[rel(_,_,0),rel(_,_,0)]],         % pattern: only the two varying entities per goal are placeholders
    save_placeholder_goals('tmp/data.json',json,GoalPlaceholder,GoalPairList),
    save_expl_graph('tmp/expl.json','tmp/flags.json',GoalPlaceholder).
```
```sh
mkdir -p tmp                                             # nothing creates the directory for you
upprism model.psm                                        # or: upprism model.psm train  -> prism_main([train])
tprism train --input tmp/ --dataset tmp/data.json1_0 --sgd_loss preference_pair --max_iterate 10 --sgd_minibatch_size 5 --sgd_learning_rate 0.01 --cpu
```
Note `data.json1_0`: the JSON writer appends `<goal-group>_<chunk>` to the name you gave. **Use `json` (placeholders) or `npy` (embeddings) explicitly:**
the default format is hdf5, which the standard PRISM build does not contain (it prints `[ERROR] hdf5 format is not implemented` and still says `yes`).
The score is not a probability, so a ranking loss fits; for probability-valued goals (softmax models) use `nll` *after applying the fix in known-issues.md*, or a custom loss (python-api.md §9).

`--input P` (alias `-I`): if `P` is an existing directory the code *concatenates* `P`+`expl.json`, so **the trailing slash is mandatory** (`--input tmp/`; `--input tmp` fails with `FileNotFoundError: 'tmpexpl.json'` — hit by two eval agents); otherwise `P.expl.json`, `P.flags.json`
(files named `mnist.expl.json` are addressed as `--input ./mnist_tmp/mnist`). `train` writes `P.vocab.pkl`, `P.model.best.model`, `P.model.last.model`
next to them; `tprism test` with a *different* `--input` prefix must be given `--vocab` and `--model` of the trained one.
The older `--intermediate_data_prefix P` still works but is deprecated and is a plain string prefix (`P`+`expl.json`).
Details of files, options, placeholders, embeddings, custom operators/losses: `references/pipeline.md`.

## 3. Running it from Python (notebooks, `T_PRISM_tutorial.ipynb`)

```python
from tprism.loader import load_explanation_graph
from tprism.model import TprismModel
from tprism.loss import CE                      # CE_pl, NLL, MSE, PreferencePair, RMSEPair; None = forward only
graph, tensor_shapes, flags = load_explanation_graph("d/train.expl.json", "d/train.flags.json")
flags.embedding = ["d/data.h5"]; flags.vocab = "d/vocab.pkl"; flags.model = "d/model"
flags.sgd_learning_rate = 0.01; flags.max_iterate = 100
model = TprismModel(flags, tensor_shapes, graph, loss_cls=CE)
# model.operator_loader.set_cls("my_op", MyOp)  # custom PyTorch operator, before build
model.build(input_data=None, load_vocab=False, embedding_key="train")
evaluator = model.fit()                         # writes d/model.best.model, d/model.last.model
labels, outputs = model.pred()                  # (labels, outputs), not (loss, out)
```
The Prolog part can run through `upprism` or PyPRISM's `PrismEngine(bin_path=...).set_db(program)` + `.query("save_expl_graph(...)")`.
Rules that the tutorial gets wrong or leaves implicit (all verified; details and code in `references/python-api.md`):
- **Test phase**: a new `load_explanation_graph` gives fresh flags. Set `flags.embedding` and `flags.vocab` again, `build(..., load_vocab=True, embedding_key="test")`,
  then **`model.load("d/model.best.model")`**; `pred()` never loads parameters. Otherwise every test input gets the same output.
- `sgd_weight_decay` defaults to **0.01** (PRISM flag): products of several small tensors (tensor train) collapse to 0 and the loss never moves. Set 0.
- Custom losses and operators are plain Python classes passed as `loss_cls` / registered with `set_cls`; no package edit needed.
- Reading goal values: `pred()` with no loss stacks all goals (same shapes only); `comp_expl_graph.forward()` needs the feed set first (python-api.md §8).
The tutorial's sections as checked recipes (decomposition, SVD, tensor train, MLP, `get/2`, addition, CNN operators, slicing): `references/tutorial-recipes.md`.

## 4. Modeling recipes

- **Dot-product/bilinear scoring (DistMult):** same index in all atoms ⇒ contracted ⇒ scalar.
- **Matrix–vector layers (MLP):** put `operator(sigmoid)` *before* the matrix that produces the layer output; chain layers by calling the next layer predicate at the end of the body:
  ```prolog
  output(Y,X) :- layer0(X,Y).
  layer0(X,_) :- matrix(w(0),[i,j]), layer1(X).                       % W0 · hidden = logits (NO softmax with `ce`)
  layer1(X)   :- operator(sigmoid), matrix(w(1),[j,k]), layer2(X).    % sigmoid(W1 · x)
  layer2(X)   :- vector(in(X),[k]).
  ```
  (index `j` is free in `layer1`'s result and gets contracted in the caller *by name*: gotcha 4.)
  `Y` is the class label read by the `ce` loss (**the label is the goal's first argument**); it is deliberately unused in the body.
  **`ce`/`ce_pl` apply softmax themselves**: adding `operator(softmax)` on the output layer (as the shipped `mlp0` and tutorial §09 do) double-softmaxes: the loss cannot fall below −ln(e/(e+C−1)) per sample (1.46 for 10 classes) and training is slow or stalls (3 classes: loss 1.083, accuracy 0.38; 10 classes: loss 2.30 → 1.8, accuracy 0.92 against 1.00 without it). Use `softmax` only where you need probabilities.
- **Recursion with unrolled depth (Markov chain, RNN-style, tensor train):** Prolog counts steps at compile time, tensors carry the state; index atoms can be passed as arguments to alternate bond indices; see `references/examples.md` §3 and tutorial-recipes §07.
- **Probability-like outputs:** wrap a tensor in `operator(softmax)` (softmax is over the *last* axis) and use `--sgd_loss nll` (only after applying the NLL fix in `references/known-issues.md`).
- **Nondeterministic enumeration = marginalisation** (MNIST addition: enumerate digit pairs with `member`, add, look up a one-hot): the inner classifier needs `softmax`, the summed goal is then already a distribution — no outer softmax, and a −log p[label] loss (tutorial-recipes §12, verified; the tutorial's softmax + `ce` version does not learn).
- **Decomposition / fitting a given tensor:** put the model goal and the data goal in one goal group, `save_expl_graph(...,[[Model,Data]])`, and use `rmse_pair` (tutorial-recipes §05–07). Constrained factors: `tensor_atom(u,[7,7],orthogonal)` (geotorch).
- **Inputs that differ per example (features, entity ids):** make them *placeholders* (`save_placeholder_goals`), otherwise the graph is rebuilt per goal (huge and slow).
- **Fixed, data-derived tensors** (adjacency matrix, features): `--embedding file` / `flags.embedding`, written by `save_embedding_from_pattern/4-5` (a 0/1 tensor from facts *or rules*) or from Python (python-api.md §12).

## 5. Top gotchas (each expanded in `references/gotchas.md`)

1. **Compile time vs run time.** Prolog can't see tensor values. `X > 0` on a learned value, or reading a number out of a tensor, is impossible; branch on symbols/integers that are known at `upprism` time.
2. **Prolog nondeterminism is a sum.** A `member/2`, or two matching clauses, silently doubles/multiplies terms. Use `->`/guards when you need "the one matching case". Cuts are unreliable (tabled search).
3. **Indices are atoms, not variables.** `tensor(w,[i,j])` ✔, `tensor(w,[I,J])` ✘: Prolog enumerates the variables over the whole index-atom pool, so with pool `{i,j}` you silently get the *sum* of `[i,i]`, `[i,j]`, `[j,i]`, `[j,j]` (verified with `probf`). Keep the pool small. When a helper predicate receives an index list as an argument (`prob_tensor_msw(X,Index)`), declare the atoms with `index_atoms([i,j])`.
4. **Index names leak across predicate calls.** A callee's free index is contracted with a same-named index in the caller. Reusing `k` by accident contracts too much or too little; rename explicitly with `subgoal(G,[..])`. Index names are otherwise clause-local.
5. **`operator/1` covers the rest of *its own clause body*, in order.** To apply a non-linearity to a *sum of clauses*, put it in a wrapper clause and the alternatives in a helper predicate (transitive-closure example: `rel2 :- operator(min1), rel2_helper.`).
6. **Don't redefine reserved names**: `tensor/2`, `vector/2`, `matrix/2`, `operator/1`, `subgoal/2`, `prob_tensor/2`, `tensor_atom/2,3`, `index_atoms/1`, `index/2`. The old manual defines its own `prob_tensor/2` helper; the translator rewrites body atoms `prob_tensor(Dist,Goals)` (variational-distribution feature), and `upprism` then aborts with `existence_error(procedure,i/0)` (verified). Use `prob_tensor_msw/2`.
7. **Shapes are checked late, axis order not at all.** Missing/incorrect `tensor_atom` shapes, rank ≠ number of indices, or inconsistent dimensions for one index name only fail in `tprism` with einsum errors. Two goals compared by a loss (`rmse_pair`) must have their free indices in the same *sorted-name* order, or they are compared transposed without an error.
8. **Placeholders take integers only**, and the goals passed to `save_expl_graph` must be the *pattern* (`rel(_,_,_)`) when placeholder data is used, not the ground data goals. `save_placeholder_goals` binds the pattern's variables: use a fresh pattern for the second (test) call, or it silently writes an empty table.
9. **Losses and label position.** `ce` reads the class label from the goal's *first* argument (`output(Y,X)`) and fails with placeholders; with placeholders use `ce_pl` (label = `$placeholder1$`; another placeholder with `--sgd_loss 'ce_pl($placeholder2$)'` in current tprism, numeric arguments only up to a0ea215); `nll` expects the goal's value to be a probability **and is buggy** (known-issues.md); `preference_pair` needs `[Pos,Neg]` goal pairs; `rmse_pair` compares the first two goals of each group.
10. **Embedding names must match exactly.** Without placeholders `in(3)` reads dataset `tensor_in_3_` (one per sample); `get(in,3)` and a placeholder `in(X)` read row X of `tensor_in_`. A mismatch is *silent*: the tensor becomes a random trainable parameter and the model stalls at chance (verified: accuracy 0.10). Check with `--debug_verb embedding feed` or the `state_dict` keys.
11. **Default hyper-parameters are weak**: `sgd_learning_rate` 0.0001, `max_iterate` 10, `sgd_weight_decay` 0.01 (kills small-product models), Adam.
12. **Known repo bugs / broken examples** are listed in `references/known-issues.md` (with the fixes). Use `mlp0` (without its output softmax), `distmult_sample02`, `markov_chain` (after patch), `transitive_closure01` (with `train`, not `test`) as templates.
13. **Docs drift.** The PDF manual still says TensorFlow, `--flags flags.h5`, `choice/2`, `index_list/1`; the code is PyTorch, JSON flags, `index_atoms/1`. The tutorial notebook's saved logs predate the per-goal loss normalisation (sums like 230 instead of 2.3). Trust the `exs/tensor/*` programs, `bin/tprism/main.py --help`, and a run.

## 6. When something is unclear

- Read the closest example under `exs/tensor/` (README + `run.sh` show the exact command sequence) or the matching tutorial section (`references/tutorial-recipes.md`).
- Run `upprism` on a tiny goal and `probf(Goal)` (in the `prism` REPL, after `prism(model)`, or as a `PrismEngine` query) — it prints the tensor equation, e.g. `rel(1,3,0) <=> tensor(v(1),[i]) & tensor(v(3),[i]) & tensor(r(0),[i])`. If the equation isn't what you meant, the bug is in the Prolog part, before any PyTorch.
- Evaluate each goal on a tiny tensor and compare with numpy (`np.einsum`) — python-api.md §8 has the helper.
- `tprism ... --verbose` or `--debug_verb graph feed embedding param` for the PyTorch side; in Python, `forward(dryrun=True)` + `tprism.plot.graph.plot_and_or_graph` draws the graph.
- Reference files here: `references/known-issues.md` (verified environment, bugs, file names), `references/python-api.md` (TprismModel workflow, test phase, custom losses/operators, embedding files), `references/tutorial-recipes.md` (the tutorial, checked), `references/semantics.md` (index/einsum rules with worked derivations), `references/examples.md`, `references/gotchas.md`, `references/pipeline.md`.
