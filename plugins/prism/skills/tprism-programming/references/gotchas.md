# T-PRISM gotchas (wrong → right, with the Prolog/ProbLog intuition that causes each)

Contents: T1 compile/run wall · T2 nondeterminism is a sum · T3 cuts and negation · T4 index atoms vs variables ·
T5 index leakage across calls · T6 operator scope · T7 reserved names · T8 shapes/ranks · T9 placeholders ·
T10 goals passed to save_expl_graph · T11 labels · T12 normalisation · T13 cycles · T14 directories/modes ·
T15 stale docs · T16 debugging recipe · T17 broken repo files · T18 axis order · T19 weight decay and defaults ·
T20 test phase in Python · T21 reused placeholder pattern · T22 losses for marginals · T23 reading goal values in Python

---

## T1. Prolog cannot look at tensor values

Coming from ProbLog/PRISM you may want "if the score is high then …". Prolog runs only at `upprism` time, when tensors don't have values yet.

```prolog
% WRONG: X is a tensor; there is no number to compare
score(U,V) :- tensor(v(U),[i]), tensor(v(V),[i]), ...   , Score > 0.5.
```
Move numerics into operators (`relu`, `sigmoid`, `min1`, or a custom operator) and apply thresholds after `tprism test --output` in Python. Prolog conditions may only test *symbols / integers available at compile time* (ids, depths, list contents).

## T2. Every Prolog solution is a summand

```prolog
% Intended: "the label of X" (one value)
out(X,Y) :- member(Y,[0,1,2]), tensor(onehot(Y),[l]), ...   % enumerates 3 alternatives, sums the 3 tensors
```
This is exactly what you want for marginalisation (MNIST-addition enumerates digit pairs), but it is a bug if you meant a single case. When a unique branch is intended, select it with `->` or by making the test deterministic (ground arithmetic comparison) and avoid leaving choice points. Two clauses whose Prolog guards both succeed also add.
Also remember that unrolled enumeration multiplies graph size: `member` over 10 × 10 combinations = 100 terms per goal.

## T3. Cuts, negation, `findall` over tensor goals

The graph is built by tabled exhaustive search: `!` after a nondeterministic call doesn't commit as in Prolog, and `\+`/`not` over tensor-carrying goals has no tensor meaning. Use them only around *pure Prolog* tests (`\+ member(X,L)` on symbols) and in the data-preparation part (`prism_main`, helper preds like `candidate/4` in the DistMult example, which uses `findall` and `not member`).

## T4. Indices are atoms; variables are *enumerated*, not ignored

```prolog
tensor_atom(w,[3,4]). tensor_atom(v,[4]).
good     :- tensor(w,[i,j]), tensor(v,[j]).      % probf: tensor(w,[i,j]) & tensor(v,[j])
bad(_)   :- tensor(w,[I,J]), tensor(v,[J]).      % probf (verified), index pool = {i,j}:
%    tensor(w,[i,i]) & tensor(v,[i])  v  tensor(w,[i,j]) & tensor(v,[j])  v  tensor(w,[j,i]) & tensor(v,[i])  v  tensor(w,[j,j]) & tensor(v,[j])
```
A variable in an index list is bound to every index atom in the pool by the exhaustive search (`tensor/2` is compiled to a `msw` whose "outcomes" are all index combinations),
so the clause silently becomes a *sum of several einsums*. Always write literal atoms. Helper predicates that forward an index list need the pool declared:
```prolog
index_atoms([i,j]).
prob_tensor_msw(X,Index) :- operator(softmax), tensor(X,Index).   % Index is a variable here; callers pass [i,j]
```
The pool is collected from literal `tensor/2` index lists plus `index_atoms/1`; keep it small, since an accidental variable index multiplies terms by |pool|^rank.

## T5. Index names leak from callee to caller

```prolog
layer1(X) :- operator(sigmoid), matrix(w(1),[j,k]), layer2(X).       % free after contraction: j
output(X) :- matrix(w(0),[i,j]), layer1(X).                           % j contracted here — intended
%
out2(X)   :- matrix(w(2),[j,m]), layer1(X).     % ALSO contracts j with layer1's free j — maybe not intended
```
Use `subgoal(layer1(X),[q])` to rename the callee's result, then contract explicitly. Same predicate used twice with same free index: the second use must be renamed or the two results are contracted with each other.

## T6. `operator/1` scope and order

```prolog
% WRONG if you wanted min1 over the sum of both alternatives:
rel2 :- operator(min1), tensor(rel1,[i,k]).
rel2 :- operator(min1), tensor(rel1,[i,j]), subgoal(rel2,[j,k]).   % applies min1 to each term separately, then sums

% RIGHT: one wrapper clause, alternatives in a helper
rel2 :- operator(min1), rel2_helper.
rel2_helper :- tensor(rel1,[i,k]).
rel2_helper :- tensor(rel1,[i,j]), subgoal(rel2,[j,k]).
```
Composition order: `operator(f), operator(g), A` = f(g(A)). For a layer `sigmoid(W·h)` the operator goes in the same clause as `W`, ahead of the call that provides `h`.
`softmax` is over the last axis only; for a matrix with row-stochastic semantics the row index must be the *first* axis and the softmax axis last (`tr[i,j]` softmax over `j`).

## T7. Reserved names and the `prob_tensor` collision

Do not define predicates named `tensor/2`, `vector/2`, `matrix/2`, `operator/1`, `subgoal/2`, `prob_tensor/2`, `tensor_atom/2,3`, `index_atoms/1`, `index/2` (legacy), and PRISM's `msw/2`, `values/2` (the translator rewrites tensor and operator atoms into `msw` pseudo-switches internally). Defining `prob_tensor/2` yourself (as printed in the older PDF manual) makes `upprism` abort at `probf`/`save_expl_graph` with `error(existence_error(procedure,i/0),call/1)` — verified: the translator treats the index list of every `prob_tensor(_,List)` call as a list of goals. Name the helper `prob_tensor_msw/2` (as `exs/tensor/markov_chain/markov_chain.psm` does).

## T8. Shape errors show up late

- `tensor_atom(Name,Shape)` must cover every tensor name used (patterns with `_` allowed). Otherwise the tensor has unknown shape.
- rank(tensor) must equal `length(IndexList)`.
- For one index name, all atoms must agree in size at that axis.
- A predicate's clauses must leave the same free axes. `transitive_closure01` as shipped leaves `[i,k]` in one clause and `[j,k]` in the other: `tprism` prints `missmatch indices: [['i','k'],['j','k']]` at every step (it still converges, using the first clause's names). Use identical names — verified: same result, no warning.
- `tensor_atom(rel1,[N,N]) :- dim(N).` is allowed (body evaluated at compile time; `assert(dim(N))` in `prism_main`).
- The `onehot(_)` atom must be declared with its length: `tensor_atom(onehot(_),[10]).`

## T9. Placeholders (batched graphs)

- Values must be **integers**. Non-integer features go into an embedding file, indexed by an integer id (`in(X)` with `X` the sample id).
- Pattern variables must be exactly the parts that vary: `GoalPlaceholder=[output(_,_)]`. Anything constant must be spelled out (`rel(_,_,0)`).
- A pair/triple of goals per sample (preference learning) is a *list* pattern: `[[rel(_,_,0),rel(_,_,0)]]`.
- Different placeholder patterns ⇒ different explanation graphs; keep training/test on the same pattern.
- For train/test splits use two `prism_main([train])`/`prism_main([test])` clauses and run `upprism model.psm train` then `upprism model.psm test`; each writes its own data file. Only the train run needs to write the graph (`mlp/mnist.psm`), the test run needs the data file only.

## T10. What goes into `save_expl_graph/3`

It builds the graph for the goals you give. With placeholders, pass the **pattern list** (after `save_placeholder_goals` unified its variables with placeholder atoms, `$placeholder1$` …); without placeholders, pass the ground goals themselves. Passing ground data goals *and* placeholder data would rebuild one graph per goal.

## T11. Supervision: labels, losses and goal shape

- `ce`/`ce_pl` already apply softmax (`F.cross_entropy` / `log_softmax`): do not put `operator(softmax)` on the output layer. The loss then cannot go below −ln(e/(e+C−1)) per sample (1.46 for C = 10), and training stalls or crawls (3 classes: loss 1.083 / acc 0.38; 10 classes: 2.30 → 1.8, acc 0.92 vs 1.00 without the softmax — both reproduced). The shipped `mlp0` and tutorial §09/§11 have this softmax.
- `--sgd_loss ce`: the **first argument of the goal is the integer class label** (`output(Y,X)`, as in `exs/tensor/mlp0`, `mlp1`, `addition`); the head's free tensor axis is the vector of scores. The label must not be needed by the body. Putting the label second (`output(X,Y)`) makes `ce` read `X` as the label.
- With placeholders use `ce_pl` (class `CE_pl`): the label is taken from placeholder `$placeholder1$` by default, i.e. the *first* variable of the goal pattern, so `GoalPlaceholder=[output(_,_)]` with label first works. Another placeholder is chosen with a term: `--sgd_loss 'ce_pl($placeholder2$)'` (quote it in the shell), `set_prism_flag(sgd_loss, ce_pl('$placeholder2$'))`, or in Python `loss_obj=CE_pl(["$placeholder2$"])`. The term syntax needs the `sgd_loss` parser of the current repo; tprism up to commit a0ea215 only accepts numeric arguments, so there `ce_pl($placeholder2$)` (and the old `ce_pl2` of `exs/tensor/mlp/run.sh`) end in `output/loss is None in training`.
- Numeric loss parameters go in parentheses: `--sgd_loss preference_pair(0.5)` (margin gamma, default 1.0).
- `--sgd_loss nll`: goals are the observed events; their value should be a **probability** (softmax/prob-tensor construction), e.g. Markov chain, PCFG via `simulated_msw`. **Bug up to at least commit a0ea215:** `NLL` averages the probability itself, so training *lowers* the likelihood (observed 0.10 → 0.086 on `markov_chain`). Apply `references/tprism-fixes.patch` (verified: loss 2.28 → 0.61) before trusting `nll`; see `known-issues.md`. In Python a custom loss avoids the patch (T22).
- `preference_pair`: goals come as `[Pos,Neg]` pairs; `rmse_pair`, `mse`, `ce_pl` exist too (`bin/tprism/loss/standard_loss.py`). Custom loss = Python class in `tprism/loss/`.
- Add L2 etc. via flags: `sgd_weight_decay`, optimizer `--sgd_optimizer`.
- Training with `test` mode reports the loss/outputs of the same pipeline; `--output file.npy` stores predictions.

## T12. Normalisation is your responsibility

Nothing forces tensors to be distributions. To get a categorical distribution use `operator(softmax)`. Also remember the softmax axis (T6). Comparing with PRISM: where `msw(tr(S),Next)` ensured Σ_Next θ = 1, you now have to write it.

## T13. Cyclic programs

Transitive-closure-like programs produce cyclic tensor equations. Requirements: `:- set_prism_flag(error_on_cycle,off).` in the `.psm`, run **`tprism train ... --cycle`** (the shipped `run.sh` uses `test`, which fails without a vocab file that only `train` creates), and use a saturating operator (`min1`) so the fixed-point iteration converges (verified: loss 14→7→4→0 on the 7-node example). `train` is just the mode that runs the iteration; `test --cycle --output` crashes (`UnboundLocalError`). Needs the `subgoal/2` fix in `known-issues.md`.

## T14. Files, directories, formats

- `mkdir -p tmp` before `upprism`; the Prolog side does not create directories.
- Formats: explanation graph/flags `json` (default), `pb`, `pbtxt`; placeholder data `hdf5` (default), `json`, `pb`, `pbtxt`, `npy`; embedding `hdf5` (default) or `npy`. **The standard build (CI, `USE_NPY=1`) has neither HDF5 nor protobuf**: hdf5 calls print `[ERROR] hdf5 format is not implemented` yet the query still succeeds and writes nothing, so always pass `json`/`npy` explicitly. JSON placeholder data is written as `<name>1_0` (see known-issues.md for all file names).
- `save_expl_graph/2` (no goals) reads goals from the `data_source` flag file.
- Default output file names: `expl.json`, `flags.json`, `data.h5`.
- Path arguments in `tprism` are relative to the current directory; the `.psm` paths are relative to where `upprism` runs (`run.sh` does `cd $(dirname $0)`).

## T15. Documentation drift (don't copy from the PDF blindly)

| Older text | Current |
|---|---|
| TensorFlow, `tensorflow-gpu` | PyTorch |
| `--flags flags.h5`, `--model model.ckpt` | JSON flags; model file name via `--model` |
| `choice/2` | not in the translator; use one-hot tensors |
| `index_list/1` | `index_atoms/1` |
| `--data` | deprecated; use `--dataset` (both accepted) |
| `--intermediate_data_prefix P` | deprecated; use `--input P` (dir ⇒ `P/expl.json`, else `P.expl.json`) |
| `tprism test --input other_prefix` | needs `--vocab` and `--model` of the trained run (vocab is created by `train`) |
| `prob_tensor/2` helper | `prob_tensor_msw/2` |
| `matrix/2`, `vector/2` | still work as aliases for `tensor/2` (used in the shipped MLP examples) |

## T17. Broken things in the repo

See `known-issues.md` (`subgoal/2` crash, `nll`, `ce_pl2`, TC `run.sh`); patch in `tprism-fixes.patch`.

## T16. Debugging recipe

1. In the `prism` REPL: `prism(model).` then `probf(Goal)` for one ground goal: check each `<=>` line is the tensor equation you meant (index names, missing/extra terms from T2).
2. `upprism model.psm` and open `expl.json`/`flags.json`; flags contains each tensor's shape (`[Name,Shape,...]`).
3. `tprism train --cpu --verbose` (or `--debug_verb graph param feed embedding`) to see einsum shapes.
4. Shrink dimensions (e.g. 3×3) and hand-compute the expected value before scaling up (in Python: python-api.md §8, compare with `np.einsum`).
5. If `tprism` reports an einsum error, look at indices first (T4/T5/T8), then at rank/`tensor_atom`.
6. If the loss does not move at all, suspect the loss/softmax combination (T11, T22), weight decay (T19), or a missing embedding that became a random parameter (T8, `state_dict` keys).

## T18. The axes of a result are sorted by index name

The free indices of a clause or goal are the axes of its tensor **in the order of their names**, not in the order they are written
(verified: `t :- tensor(m,[j,i]).` evaluates to mᵀ; the tensor-train goal `tensor(v(a4),[i,a4]), t([a1,a2,a3],i)` comes out as
`[a1,a2,a3,a4]`). This matters whenever values are compared or read outside the einsum: `rmse_pair`/`mse` compare element by element,
so a model goal with free `[k,i]` and a data goal `[i,k]` are compared transposed without an error; `--output` arrays and
`pred()` outputs follow the sorted order too. Choose index names whose alphabetical order is the axis order you want.

## T19. Weight decay and other defaults

`sgd_weight_decay` is a PRISM flag with default **0.01**, exported to `flags.json` (the Python dataclass default 1e-10 never applies
to programs compiled by `upprism`). With Adam it acts as L2 on every parameter each step. A goal that is a product of several small
tensors has tiny gradients, the decay wins, and the model collapses to zero: tutorial §07's tensor train stays at loss 0.135 (the
RMSE of the zero tensor, output ≈1e-12). With `sgd_weight_decay` 0 it trains (0.135 → 0.057). Set it with
`set_prism_flag(sgd_weight_decay,0)`, `--sgd_weight_decay 0` or `flags.sgd_weight_decay = 0`. The other defaults are also
conservative: `sgd_learning_rate` 0.0001, `max_iterate` 10, `sgd_minibatch_size` 1.

## T20. The test phase in Python needs the embedding and the parameters again

`load_explanation_graph` on the test files returns fresh flags, and `TprismModel.pred()` does not read any model file. Without
`flags.embedding = [...]`, `flags.vocab = <train vocab>`, `build(..., load_vocab=True, embedding_key="test")` and
`model.load("<prefix>.best.model")`, the test model runs on random parameters and missing inputs: all outputs are identical
(tutorial §11; reproduced: one distinct output, accuracy 0.06; with them 1.00). The `tprism test` command does this itself when
`--model`/`--vocab` are given.

## T21. `save_placeholder_goals` binds its pattern

After `save_placeholder_goals(F,M,P,Gs)` the variables of `P` are the atoms `'$placeholder1$'`, … . That is what
`save_expl_graph(...,P)` needs, but reusing `P` for a second data set (train and test in one `prism_main`) writes
`{"placeholders":[]}` and an empty array without any message; `tprism` fails later with `KeyError: '$placeholder2$'`.
Write the pattern again for each call (`[output(_,_)]`), or use separate `prism_main([train])` / `prism_main([test])` runs.

## T22. Losses for marginal probabilities

When a goal's value is a probability vector built by summing Prolog solutions (MNIST addition, any marginalisation), `ce` is
wrong twice: it applies softmax to probabilities, and an extra `operator(softmax)` on the sum (as tutorial §12 has) makes it worse.
Verified: that version stays at loss ln 19 (sum accuracy 0.11). Keep the inner classifier's `softmax`, drop the outer one, and minimise
−log p[label] (python-api.md §9 `LabelNLL`; loss 2.84 → 1.22, sum accuracy 0.78, digit accuracy 0.80). With the `tprism` command,
put such a loss class under `tprism/loss/`. Another way, not tested: contract the vector with `tensor(onehot(Label,N),[l])` so the
goal is the scalar p[label], and use `nll` after the patch.

## T23. Reading goal values in Python

- `pred()` with no loss (`loss_cls=None`) stacks the values of all goals: goals of different shapes raise `RuntimeError: stack expects each tensor to be equal size`.
- `model.comp_expl_graph.forward()` directly after `build()` raises `KeyError: <tprism.placeholder.PlaceholderData ...>` when embeddings are used; set the feed first (python-api.md §8) or call it after `fit`/`pred`. `forward(dryrun=True)` (symbolic, for plotting) needs no feed.
- `pred()` returns `(labels, outputs)`; the tutorial's `loss, out = model.pred()` names the first element wrongly.
