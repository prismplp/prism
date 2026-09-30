# Verified environment and known issues (repo commit 169a258, checked 2026-09-29)

`bin/tprism` is unchanged from 169a258 to a0ea215, and the two bugs below are still in that code (checked 2026-09-30 by reading
`standard_loss.py` and `standard_op.py`). The Python-API checks of 2026-09-30 (python-api.md, tutorial-recipes.md) ran natively on
Ubuntu 24.04 with the prebuilt `prism_tprism_pre_linux_ubuntu24` package, Python 3.14, torch 2.13 and protobuf 7.36.

Everything below was **executed** in Docker (`references/Dockerfile`: Ubuntu 22.04, linux/amd64 because the bundled B-Prolog
libs are x86-64 only; PRISM built as the CI does with `USE_NPY=1`, i.e. **no HDF5, no protobuf**; PyTorch CPU 2.x, Python 3.10).
On Apple Silicon: `docker build --platform linux/amd64 ...` (slow under emulation but works). The release tarballs are Linux-only.

Build & use:
```sh
git clone --depth 1 https://github.com/prismplp/prism repo
docker build -t prism-skills -f Dockerfile .      # Dockerfile expects the checkout in ./repo
docker run --rm --platform linux/amd64 -v $PWD:/work -w /work prism-skills sh -c 'upprism model.psm && tprism train --input tmp/ ...'
```

## Bugs in the shipped code (apply `tprism-fixes.patch` to `bin/tprism`, then `pip install -e bin/`, or edit the installed package)

1. **`subgoal/2` crashes `tprism`.** `Reindex.get_output_template` (`bin/tprism/op/standard_op.py`) returns plain strings, so
   `expl_graph.py: _get_unique_symbol_list` fails with `AttributeError: 'str' object has no attribute 'symbol'`.
   Affects every program that uses `subgoal/2`: `markov_chain`, `transitive_closure01/02`. The patch converts the names with
   `parse_tensor_index`. Without the patch, avoid `subgoal/2` (rely on implicit indices) or patch first.
2. **`--sgd_loss nll` minimises the goal probability instead of maximising it.** `NLL.call` (`bin/tprism/loss/standard_loss.py`)
   computes `nll = -log(p)` but collects `p` itself and averages that as the loss. Observed on `markov_chain`: the "loss" starts at
   0.10 (= the uniform probability) and drops to 0.086, i.e. the model gets *worse*. With the patch (`o.append(nll)`) the loss goes
   2.28 → 0.61 and the reported value is a real negative log-likelihood. Note the patch also changes the `output` returned by `test --output`
   to the NLL values. If you cannot patch, use `mse`/custom loss or treat `nll` results as invalid.

## Shipped-example problems (don't copy blindly)

| Where | Problem | Use instead |
|---|---|---|
| `exs/tensor/transitive_closure01/run.sh` | runs `tprism test ...` first ⇒ `FileNotFoundError: ...vocab.pkl` (vocab is created by `train`) | `tprism train --input transitive_closure_tmp/ --embedding transitive_closure_tmp/embedding.npy.json --cycle --cpu` (works, loss 14→7→4→0; `train` is a dummy mode) |
| same, `tprism test --output f.npy --cycle` | `UnboundLocalError: local variable 'out'` in `main.py` (test with `--cycle` doesn't produce output) | read the result from the train logs, or patch `run_test` |
| same psm | clause 2 uses `[j,k]`, clause 1 `[i,k]` ⇒ `WARNING missmatch indices: [['i','k'],['j','k']]` at every step (still converges) | use `[i,k]` in both clauses: no warning, same result |
| `exs/tensor/mlp/run.sh` up to a0ea215 | `--sgd_loss ce_pl2` is not registered ⇒ `RuntimeError: output/loss is None in training`; goal is `output(X,Y)` (label second) | `--sgd_loss ce_pl` with goals `output(Y,X)` (label first) — trained to 100% train accuracy on synthetic data. The current repo's `run.sh` uses `--sgd_loss 'ce_pl($placeholder2$)'`, which needs the current `sgd_loss` parser |
| `exs/tensor/mlp0` (and tutorial §09/§11) | `operator(softmax)` on the output layer with `ce`: double softmax | remove the softmax (gotchas T11) |
| `save_placeholder_goals/3`, `save_embedding_from_pattern/4` defaults | default format is hdf5, which the standard build lacks: `[ERROR] hdf5 format is not implemented (please compile prism with the USE_H5 option)` — printed, but the query still says `yes` and no file is written | pass `json` (placeholders) / `npy` (embeddings) explicitly |

## File names actually produced (formats available in the standard build: json, npy)

| Call | Files |
|---|---|
| `save_placeholder_goals('d/data.json',json,P,Gs)` | `d/data.json1_0` (= name + goal-group counter + `_` + chunk), possibly more chunks. Pass that exact name to `--dataset`. |
| `save_placeholder_goals('d/ph',npy,P,Gs)` | directory `d/ph/` with `placeholder.npy.json` and `placeholder_0.npy`... |
| `save_embedding_from_pattern(Axes,Pat,tensor(t),'d/embedding',npy)` | `d/embedding.npy`, `d/embedding.npy.json`, `d/embedding.txt`; pass `d/embedding.npy.json` to `--embedding` |
| `save_expl_graph('d/x.expl.json','d/x.flags.json',Gs)` | as named; then `--input d/x` |
| `tprism train --input d/x` | writes `d/x.vocab.pkl`, `d/x.model.best.model`, `d/x.model.last.model` |

`tprism test --input d/y` with a *different* prefix `y` looks for `d/y.vocab.pkl` and fails; give `--vocab d/x.vocab.pkl --model d/x.model` (verified for `--model`/`--vocab`; `--output p.npy` then works, shape `(1, N, classes)` in my run).

## `T_PRISM_tutorial.ipynb` (repo root)

Re-run on 2026-09-30 (details and fixes in tutorial-recipes.md):

| Section | Problem | Fix |
|---|---|---|
| §07 tensor train | loss stays 0.135 for 2000 epochs: the default `sgd_weight_decay` 0.01 shrinks the four factors to 0 | `flags.sgd_weight_decay = 0` (0.135 → 0.057) |
| §07 | `relx(X,Y,Z,d) :- member(X,..), member(X,..), member(X,..)` repeats `X` | `member(Y,..)`, `member(Z,..)` |
| §09, §11 | `operator(softmax)` + `CE` (double softmax) | drop the softmax |
| §10 | text says `get(x,n)` is the n-th column | it is row n (`x[n,:]`) |
| §11 test cell | fresh flags without `embedding`, no `model.load` ⇒ identical predictions for all 50 test images | python-api.md §6 |
| §11 goal-level validation | trains on the `graph` left over from the test cell | reload the training graph |
| §12 MNIST addition | outer `operator(softmax)` + `CE` on a marginal distribution; does not learn (early stop at epoch 6) | no outer softmax, −log p[label] loss (python-api.md §9) |
| §12 | `index_list/1` | ignored; `index_atoms/1` |
| §14 text embedding | quoted parameter passed with its quotes; `tokenize` gets a string and embeds characters | strip the quotes, pass a list |
| cells 102/104 | show `model` of the previous section before it is rebuilt | harmless leftovers |
| logs | per-epoch losses are sums (e.g. 230 for 100 goals); current tprism prints per-goal means (2.30) | compare only within one version |
| cell 4 | `%env PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` | not needed with protobuf 7.36 |

## Python API behaviour worth knowing (verified 2026-09-30)

- `TprismModel.pred()` returns `(labels, outputs)` and does not load a model file; call `model.load(...)`.
- `loss_cls=None`: `pred()` stacks all goal values, so goals of different shapes raise `RuntimeError: stack expects each tensor to be equal size`.
- `comp_expl_graph.forward()` right after `build()` raises `KeyError: <...PlaceholderData ...>` if embeddings are used (the feed is set by `pred`/`fit`).
- `pred(input_data=...)` prints `{}` once per minibatch (a `print` left in `model.py`).
- A reused, already bound placeholder pattern makes `save_placeholder_goals` write an empty table silently (gotchas T21).
- The free axes of a goal's value are sorted by index name (gotchas T18).

## Also found by the eval agents

- `tprism --input DIR` (no trailing slash) looks for `DIRexpl.json`: `main.py` uses `sep=""` for directories. Always `--input DIR/`.
- `ce` applies softmax internally; `mlp0`'s extra `operator(softmax)` stalls training. Embedding-name mismatches silently create a random trainable tensor
  (without placeholders `in(3)` reads `tensor_in_3_`; `get(in,3)` and placeholder `in(X)` read rows of `tensor_in_`).

## Behaviour confirmed by running (so it is safe to rely on)

- `probf/1` prints the tensor equation; `member/2` enumeration becomes `v tensor(onehot(0),[l]) v tensor(onehot(1),[l]) v ...` (sum over solutions).
- `tensor(w,[I,J])` with Prolog variables is **not ignored**: the variables are enumerated over the index-atom pool, giving a sum over all combinations, e.g. pool `{i,j}` ⇒ `tensor(w,[i,i]) v tensor(w,[i,j]) v tensor(w,[j,i]) v tensor(w,[j,j])`.
- Defining your own `prob_tensor/2` and calling it aborts `upprism` with `error(existence_error(procedure,i/0),call/1)`: the translator turns the index list into goals. Use another name (`prob_tensor_msw/2`).
- `operator(min1)` repeated in every clause is applied per clause (`operator(min1) & tensor(...) v operator(min1) & subgoal(...)`); the wrapper form gives `w <=> operator(min1) & wh`.
- `subgoal(G,[j,k])` appears in the graph as `operator(reindex([j,k])) & G`.
- MLP (`mlp0` style, synthetic 3-class data): `--sgd_loss ce` without placeholders works (label = first goal argument); with placeholders `ce` fails (`int('$placeholder1$')`) and `ce_pl` works.
- DistMult `distmult_sample02` (json placeholders + `preference_pair`) trains: loss 5.0, test-loss 4.9.
- `simulating_prism/pcfg.psm` trains, but is subject to the `nll` bug above.
