# Recipes from `T_PRISM_tutorial.ipynb`

The tutorial (repo root, also on Colab) runs every example through the Python API (python-api.md) with the
Prolog part given to PyPRISM's `PrismEngine`. Section numbers are the tutorial's. "Re-run" means the recipe was
executed again on 2026-09-30 (python-api.md has the environment); the others rely on the outputs saved in the notebook.

| § | Topic | Status | Watch out |
|---|---|---|---|
| 02–03 | tensor from facts/rules (`save_embedding_from_pattern`) | re-run | axis order = variable list; `.txt` maps positions to symbols |
| 04 | matrix product | re-run | `prod == X@X` |
| 05 | matrix decomposition X≈AB, fixed tensor from Python | re-run | goal groups `[[goalAB,goalX]]` + `RMSEPair` |
| 06 | SVD with `orthogonal` tensors | notebook; `orthogonal` re-run | geotorch, `importlib.reload(tprism.constraint)` |
| 07 | tensor-train decomposition, graph drawing | re-run | **does not train with the default weight decay**; output axes sorted by name |
| 08 | read inputs from HDF5 | re-run (synthetic) | dataset names |
| 09 | MLP on MNIST, minibatch/early stopping with placeholders | re-run (synthetic) | `operator(softmax)` + `CE` is a double softmax |
| 10 | `get/2` | re-run | row, not column |
| 11 | MLP with `get/2`, train/test, goal-level validation | re-run (synthetic) | the test cell forgets embedding and `load` |
| 12 | MNIST addition | re-run (synthetic) | **does not train as written**; fix below |
| 13 | custom operators (MLP, CNN) | MLP re-run; CNN notebook | `set_cls`, `get_output_shape` |
| 14 | text-embedding operator (under construction) | notebook | quoted atom parameter keeps quotes |
| 15 | slicing and contraction | re-run | integer index selects |

---

## §02–03 Tensors from a logic program

```prolog
tensor_atom(rel1,[7,7]).
rel1(X,X) :- member(X,[a,b,c,d,e,f,g]).      % rules work as well as facts (§03)
rel1(A,B) :- r(A,B).
rel1(C,A) :- r(A,B), r(B,C).
r(a,b). r(b,c). r(d,e). r(e,f). r(c,d). r(d,g).
```
`save_embedding_from_pattern([X,Y],rel1(X,Y),tensor(rel1),'data03/tensor_rel1',npy)` writes a 0/1 matrix with entry 1
exactly where `rel1(X,Y)` is provable: `data03/tensor_rel1.npy`, `.npy.json` (give this to `flags.embedding`) and `.txt`
(axis, index, label). The symbols of each axis are also printed (`[[a,b,c,d,e,f,g],[a,b,c,d,e,f,g]]`).
Several relations of the same size go into separate files; a tensor with a third axis `[x,y,z]` works the same way (§15).

## §04 Matrix product

```prolog
tensor_atom(rel1,[7,7]). tensor_atom(rel2,[7,7]).
goal :- tensor(rel1,[i,j]), tensor(rel2,[j,k]).        % X_ik = Σ_j A_ij B_jk
```
Save both embeddings and `save_expl_graph(...,[goal])`, then `TprismModel(..., loss_cls=None)`, `build`, and `pred()`:
`outputs` has shape (1, 1, 7, 7) in the tutorial and equals `A @ B`. The tutorial passes `loss_cls=NLL` as a "dummy loss";
`None` is cleaner.

## §05 Matrix decomposition, goal groups, fixed tensors from Python

```prolog
tensor_atom(rel1,[7,7]). tensor_atom(rel2,[7,7]). tensor_atom(relx,[7,7]).
goalAB :- tensor(rel1,[i,j]), tensor(rel2,[j,k]).      % learnable A·B
goalX  :- tensor(relx,[i,j]).                           % data (embedding)
% save_expl_graph('data05/data.expl.json','data05/data.flags.json',[[goalAB,goalX]])
```
- `[[goalAB,goalX]]` is one **goal group** of two goals. `RMSEPair` compares the first two goals of each group:
  sqrt(mean((goalAB − goalX)²)). Re-run: loss 0.543 → 0.001 in 300 epochs (lr 0.01), max |A·B − X| = 0.004.
- `outputs` of `pred()` is (1, 2, 7, 7): `outputs[0][0]` is goalAB, `outputs[0][1]` goalX.
- The learned A and B: `torch.load("model5.best.model")["tensor_rel1_"]`.
- `relx` can come from Python instead of Prolog: write `my_embedding.npy` and the JSON of python-api.md §12 (`"name":"tensor_relx_"`).

## §06 SVD with orthogonal tensors

```prolog
tensor_atom(rel1,[7,7],orthogonal).      % U
tensor_atom(rel2,[7,7],orthogonal).      % V
tensor_atom(d,[7]).                      % singular values (a vector = a diagonal matrix)
goalAB :- tensor(rel1,[i,j]), tensor(d,[j]), tensor(rel2,[j,k]).     % U diag(d) V
goalU :- tensor(rel1,[i,j]).  goalD :- tensor(d,[j]).  goalV :- tensor(rel2,[j,k]).
goalX :- tensor(relx,[i,j]).
% save_expl_graph(..., [[goalAB,goalX,goalU,goalV,goalD]])   RMSEPair uses the first two; the others are for reading
```
The shared index `j` in three atoms makes `d` act as a diagonal. The notebook reaches loss 0.000 after 2000 epochs and reads U, D, V
through the goal values (python-api.md §8). The values match `np.linalg.svd` up to order and sign.

## §07 Tensor-train decomposition

```prolog
tensor_atom(v(_),[3,7]).  tensor_atom(w(_),[3,7,3]).  tensor_atom(relx,[7,7,7,7]).
index_atoms([i,j,k,a1,a2,a3,a4]).                 % index atoms are passed as arguments below
goal :- tensor(v(a4),[i,a4]), t([a1,a2,a3],i).
t([A],X)     :- tensor(v(A),[X,A]).
t([A,B|R],j) :- tensor(w(A),[j,A,i]), t([B|R],i).
t([A,B|R],i) :- tensor(w(A),[i,A,j]), t([B|R],j).
goalX :- tensor(relx,[a1,a2,a3,a4]).
% save_expl_graph(..., [[goal,goalX]])  with RMSEPair
```
- The recursion alternates the bond index between `i` and `j`, so each core is contracted only with its neighbours
  (a ring v(a4)·w(a1)·w(a2)·v(a3)). Index names are ordinary Prolog atoms, so they can be passed as arguments
  (hence `index_atoms/1`), and the atom `a1` is used both as a tensor name part (`w(a1)`) and as an index.
- **As written it does not learn** (notebook: loss 0.135 for 2000 epochs). Re-run: 0.135 is exactly the RMSE of the zero tensor,
  and the output shrinks to ~1e-12, because the default `sgd_weight_decay` 0.01 beats the tiny gradient of a product of four
  small tensors. With `flags.sgd_weight_decay = 0`: 0.135 → 0.085 (300 epochs, lr 0.001) and → 0.057 (2000 epochs, lr 0.01).
- The goal's value comes out with axes `[a1,a2,a3,a4]` (sorted names), which happens to match `goalX`. With other names
  the pair would be compared transposed without any error; name the indices so that the sorted orders agree.
- The notebook's `relx(X,Y,Z,d) :- member(X,..),member(X,..),member(X,..)` has a typo (`X` three times), so the target is
  not what the text intends (44 ones out of 2401).
- Drawing the graph: python-api.md §13.

## §08 Inputs from HDF5

`get_mnist(...)` in the notebook writes `mnist.h5` with group `train` (and `test`) and either one dataset per image
(`tensor_in_0_`, `tensor_in_1_`, … for atoms `in(0)`, `in(1)`, …; `individual=True`) or one matrix `tensor_in_`
(for `get(in,X)` or placeholders; `individual=False`), plus `mnist.train.dat` with facts `output(Label,Id).`
The naming rules and the silent-random-parameter trap are in python-api.md §12.

## §09 MLP on MNIST

```prolog
tensor_atom(w(0),[10,256]). tensor_atom(w(1),[256,784]). tensor_atom(in(_),[784]).
output(Y,X) :- layer0(X,Y).
layer0(X,Y) :- operator(softmax), tensor(w(0),[i,j]), layer1(X,Y).     % see below
layer1(X,Y) :- operator(sigmoid), tensor(w(1),[j,k]), layer2(X,Y).
layer2(X,Y) :- tensor(in(X),[k]).
```
- `CE` is `torch.nn.functional.cross_entropy`, which applies softmax itself, so `operator(softmax)` here is a double
  softmax: the loss cannot go below −ln(e/(e+9)) = 1.46 per sample for 10 classes and training is slow (notebook: accuracy
  0.68 after 50 epochs on 100 images). Re-run on synthetic 10-class data, 100 epochs: with softmax loss 2.30 → 1.8, accuracy 0.92;
  without it 2.30 → 0.0, accuracy 1.00. Drop the output softmax when using `CE`/`CE_pl`.
- The label `Y` is the goal's **first** argument (`CE` reads `args[0]`).
- **Minibatch / early stopping**: with placeholders (`save_placeholder_goals('./data09/mnist_data_ph',npy,[output(_,_)],Gs)`,
  `load_input_data([".../placeholder.npy.json"])`, `CE_pl`, `sgd_minibatch_size`, `sgd_valid_ratio`). Full code in python-api.md §7.

## §10 `get/2`: a slice of a data tensor

```prolog
tensor_atom(get(relx,_),[7]).
goal(X) :- tensor(get(relx,X),[i]).
```
`get(relx,3)` is row 3 of the `relx` tensor (`relx[3,:]`, a slice along the first axis): re-run confirmed, the column `relx[:,3]`
differs. The data tensor is read from the embedding under the name of the first argument (`tensor_relx_`).

## §11 MLP with `get/2`, separate train and test graphs

```prolog
tensor_atom(get(in,_),[784]).
layer2(X,Y) :- tensor(get(in,X),[k]).        % image X = row X of the dataset tensor_in_
```
Build `train.expl.json` from the training facts and `test.expl.json` from the test facts. For the test model, set
`flags.embedding` and `flags.vocab` again, build with `embedding_key="test"` and `load_vocab=True`, and call `model.load(...)`
(python-api.md §6). The notebook's test cell does neither, and its 50 predictions are all the same vector.
The notebook's "goal-level validation" cell (`sgd_goal_valid_ratio=0.1`) reuses `graph` from the test cell, so it trains on the
test goals; reload the training graph first.

## §12 MNIST addition (neuro-symbolic marginalisation)

As written, the model does not learn: the notebook stops early after 6 epochs at accuracy ≈0.14, and a re-run on synthetic
digits keeps the loss at ln 19 = 2.944 (sum accuracy 0.11, digit accuracy 0.08). The cause is the loss:
`output_add(Y,X1,X2) :- operator(softmax), output_add1(Y,X1,X2).` puts a softmax on a vector that is already a probability
distribution, and `CE` applies another one. Working version (re-run: loss 2.84 → 1.22 in 40 epochs, sum accuracy 0.78, and the digit
classifier learned only from sums reaches 0.80 on test images):

```prolog
tensor_atom(w(0),[10,32]). tensor_atom(w(1),[32,20]). tensor_atom(get(in,_),[20]).
tensor_atom(onehot(_,10),[10]). tensor_atom(onehot(_,19),[19]).
number10(Y) :- member(Y,[0,1,2,3,4,5,6,7,8,9]).
sum_prob(_,X1,X2) :- number10(Y1), number10(Y2), S is Y1+Y2,       % label = first argument
                     tensor(onehot(S,19),[l]), digit_p(Y1,X1), digit_p(Y2,X2).
digit_p(Y,X)  :- tensor(onehot(Y,10),[i]), classifier(X).            % p(digit of X = Y)
classifier(X) :- operator(softmax), tensor(w(0),[i,j]), hidden(X).   % this softmax is needed: probabilities
hidden(X)     :- operator(sigmoid), tensor(w(1),[j,k]), tensor(get(in,X),[k]).
```
Goals `sum_prob(Sum,Id1,Id2)`; train with the `LabelNLL` loss of python-api.md §9 (−log p[Sum]). The 100 digit pairs per goal are
summed (T-PRISM's disjunction rule), so the goal's value is the marginal distribution over the 19 sums (it sums to 1).
- `onehot(K,N)`: the one-hot index is the first number in the name, the size comes from `tensor_atom(onehot(_,N),[N])`, so
  one-hot vectors of different lengths can coexist.
- `index_list([i,j,k,l])` in the notebook is a leftover and is ignored; the translator reads `index_atoms/1` only.
- `probf(output_add(10,0,1))` prints the 100-term sum and is a good check of the Prolog part.
- Cost: 100 paths per goal; 150 goals took about 2 s per epoch on a CPU.

## §13 Custom operators: MLP and CNN in PyTorch

Register a `BaseOperator` + `nn.Module` subclass with `model.operator_loader.set_cls(name, cls)` before `build`
(python-api.md §10). The notebook's operators:
- `layer0(X,Y) :- operator(custom_nn), vector(get(in,X),[k]).`: a whole `nn.Sequential` classifier (784 → 120 → 84 → 10);
  training accuracy 1.00 after 50 epochs on 100 images, re-run 1.00 on synthetic data.
- CNN: `tensor_atom(get(in,_),[1,28,28])`, `layer2 :- operator(conv_block(1,3,l1)), tensor(get(in,X),[i,j,k])`,
  `layer1 :- operator(conv_block(3,3,l2)), layer2(...)`, `layer0 :- operator(readout_block), layer1(...)`.
  `conv_block` gets `['1','3','l1']` (in/out channels + an unused label) and implements `get_output_shape` for the
  conv+pool size; `readout_block` flattens (3·4·4 = 48 → 10) and drops two index symbols. Training accuracy 1.00 in the notebook.
  The tensor `bias` declared there is never used.

## §14 Text embeddings as an operator (under construction)

`layer :- operator(text_embed('The capital of China is Beijing.')).`: an operator with no tensor input wrapping a
SentenceTransformer. The parameter arrives with its quotes (`"'The capital ...'"`), and the notebook passes it on with the
quotes to `model.tokenize`. Given a string instead of a list, that embeds every character separately, so the result is 34 × 1024
(the 32-character sentence plus its two quotes) rather than one sentence vector. Treat it as a sketch of how
to call a pretrained model from an operator, not as a working recipe.

## §15 Slicing and contraction

```prolog
tensor_atom(relx,[7,7,3]).
goalX :- tensor(relx,[i,j,k]), tensor(relx,[l,j,k]).    % Σ_jk relx_ijk relx_ljk: a 7×7 matrix over i,l
goal  :- tensor(relx,[i,1,1]).                          % integers select: relx[:,1,1], a 7-vector
```
Re-run on a 7×7×2 tensor: the contraction matches `np.einsum('ijk,ljk->il')` and `[i,1,0]` matches `relx[:,1,0]`. An integer in an
index list removes that axis, so the
notebook's `plt.imshow` of the 7-vector fails (a plotting error, not a T-PRISM one).
