# T-PRISM examples

Adapted from `exs/tensor/*` in the repo (each directory has a README and `run.sh` with the exact commands). They were not
executed in this environment, so run the tiny-goal check from gotchas T16 first when adapting them.

Contents: 1 DistMult with placeholders · 2 MLP · 3 Markov chain · 4 transitive closure (cyclic) ·
5 simulated msw / PCFG · 6 nondeterministic marginalisation (MNIST addition) · 7 porting a ProbLog/PRISM model

---

## 1. DistMult link prediction (`distmult_sample02`)

```prolog
tensor_atom(v(_),[20]).
tensor_atom(r(_),[20]).

rel(U,V,E) :- tensor(v(U),[i]), tensor(v(V),[i]), tensor(r(E),[i]).

% utility part: plain Prolog, run at compile time
candidate(U,E,Goals,L) :- findall(V,(member(rel(_,V,E),Goals), not member(rel(U,V,E),Goals)),L).
random_select_negative(rel(U,V,E),Goals,rel(U,NegV,E)) :- candidate(U,E,Goals,L), random_select(L,NegV).
generate_preference_pair(Gs,Pairs) :-
    maplist(G,NegG,random_select_negative(G,Gs,NegG),Gs,NegGs),
    maplist(X,Y,Z,Z=[X,Y],Gs,NegGs,Pairs).

prism_main([]) :-
    random_set_seed(1234),
    load_clauses('sample.dat',Gs),                        % rel(S,R,O). facts
    generate_preference_pair(Gs,GoalPairList),            % [[Pos,Neg],...]
    GoalPlaceholder = [[rel(_,_,0),rel(_,_,0)]],
    save_placeholder_goals('distmult_sample_tmp/data.json',json,GoalPlaceholder,GoalPairList),
    save_expl_graph('distmult_sample_tmp/expl.json','distmult_sample_tmp/flags.json',GoalPlaceholder).
```
```sh
mkdir -p distmult_sample_tmp
upprism distmult_sample.psm
tprism train --input distmult_sample_tmp --sgd_loss preference_pair --dataset distmult_sample_tmp/data.json1_0 --max_iterate 10 --sgd_minibatch_size 5 --sgd_learning_rate 0.01 --cpu
```
Notes (verified: runs, loss ≈5.0, test-loss ≈4.9): the JSON writer appends `1_0` to the file name; `sample01` is the same model without placeholders (`save_expl_graph(..., GoalPairList)` builds one graph per pair — fine for
tiny data, wasteful otherwise). The pattern fixes the relation constant `0` for *this* dataset; use `rel(_,_,_)` if relations vary.
A negative sample is produced at compile time by ordinary Prolog: T-PRISM programs are usually "Prolog data pipeline + tensor model".

## 2. Multi-layer perceptron on MNIST (`mlp0` / `mlp1`; `mlp` is the placeholder variant)

```prolog
tensor_atom(w(0),[10,256]).     % output layer
tensor_atom(w(1),[256,784]).    % hidden layer
tensor_atom(in(_),[784]).       % pixel vector of image id X (from an embedding file)

output(Y,X)  :- layer0(X,Y).                                         % label Y FIRST: `ce` reads goal arg 1 as the label
layer0(X,_)  :- matrix(w(0),[i,j]), layer1(X).                     % logits: `ce` applies softmax itself (mlp0's extra softmax stalls training)
layer1(X)    :- operator(sigmoid), matrix(w(1),[j,k]), layer2(X).
layer2(X)    :- vector(in(X),[k]).

prism_main([train]) :-
    load_clauses('./mnist/mnist.train.dat',Gs),                      % output(Label,ImageId). facts
    save_expl_graph('./mnist_tmp/mnist.expl.json','./mnist_tmp/mnist.flags.json',Gs).
prism_main([test]) :-
    load_clauses('./mnist/mnist.test.dat',Gs),
    save_expl_graph('./mnist_tmp/mnist_test.expl.json','./mnist_tmp/mnist_test.flags.json',Gs).
```
```sh
mkdir -p mnist_tmp
upprism mnist.psm train ; upprism mnist.psm test
tprism train --input ./mnist_tmp/mnist      --embedding ./mnist/mnist.h5 --sgd_loss ce --max_iterate 10 --sgd_minibatch_size 1000 --sgd_learning_rate 0.001
tprism test  --input ./mnist_tmp/mnist_test --vocab ./mnist_tmp/mnist.vocab.pkl --model ./mnist_tmp/mnist.model \
      --embedding ./mnist/mnist.h5 --sgd_loss ce --output mnist_output.npy      # different prefix ⇒ --vocab/--model of the trained run are required
```
(`--input X` with a non-directory `X` reads `X.expl.json` / `X.flags.json`; check the shipped `run.sh` for the exact flags of the variant you copy.)
`mlp1` names the embedding tensor `get(in,X)`. The embedding file is HDF5 written by Python (`h5py`: group `train`/`test`, dataset `tensor_in_`), which
`tprism` reads even when PRISM itself has no HDF5 support. Verified on synthetic data (60 samples, 3 classes, 8 features): without placeholders `ce` trains (label = first goal argument);
with placeholders (`save_placeholder_goals('tmp/ph.json',json,[output(_,_)],Gs)`, goals `output(Y,X)`, `--dataset tmp/ph.json1_0 --sgd_loss ce_pl`) it trains to 100% accuracy.
With placeholders `ce` fails (`int('$placeholder1$')`), and up to commit a0ea215 the `ce_pl2` in `exs/tensor/mlp/run.sh` is not a registered loss (`output/loss is None`);
the current `run.sh` uses `--sgd_loss 'ce_pl($placeholder2$)'` for its label-second goals `output(X,Y)`.
The image id `X` is an *integer* selecting `in(X)`; the label `Y` only feeds the loss. Without placeholders the graph holds one network copy per goal
(fine for small data, slow at MNIST scale).

## 3. Markov chain, recursion unrolled at compile time (`markov_chain`)

```prolog
tensor_atom(onehot(_),[10]).                 % built-in one-hot encoding
tensor_atom(tr,[10,10]).                     % learnable transition logits
index_atoms([i,j]).                          % needed: Index is passed as a variable below

mc(S,T,N)       :- observe_state(S,[i]), subgoal(transition(T,N),[i]).
transition(T,0) :- observe_state(T,[i]).
transition(T,N) :- N>0, NextN is N-1,
                   prob_tensor_msw(tr,[i,j]),
                   subgoal(transition(T,NextN),[j]).

prob_tensor_msw(X,Index) :- operator(softmax), tensor(X,Index).
observe_state(S,Index)   :- tensor(onehot(S),Index).

prism_main([]) :-
    Gs = [mc(0,1,2), mc(2,5,3), mc(4,0,4), mc(1,3,2)],
    save_expl_graph('markov_chain_tmp/expl.json','markov_chain_tmp/flags.json',Gs).
```
```sh
mkdir -p markov_chain_tmp
upprism markov_chain.psm
tprism train --input ./markov_chain_tmp --sgd_loss nll --max_iterate 10 --sgd_learning_rate 0.01 --cpu   # needs tprism-fixes.patch (subgoal + nll), see known-issues.md
```
The PRISM twin `markov_chain_msw.psm` uses `soft_msw(tr(I),J)` with an integer state and `learn(Gs)`; in T-PRISM the state is an
*index* instead of a bound value, so all states are handled in one matrix product rather than by enumeration.
Here the same `mc(S,T,N)` goals are observed N-step transitions: `nll` maximises their probabilities.

## 4. Transitive closure with cyclic equations (`transitive_closure01`)

```prolog
tensor_atom(rel1,[7,7]).
:- set_prism_flag(error_on_cycle,off).

rel1(a,a). rel1(b,b). ... rel1(a,b). rel1(b,c). ...     % ordinary facts (compile-time data)

rel2 :- operator(min1), rel2_helper.
rel2_helper :- tensor(rel1,[i,k]).      % same free indices in both clauses (the shipped [j,k] works but warns)
rel2_helper :- tensor(rel1,[i,j]), subgoal(rel2,[j,k]).

prism_main([]) :-
    save_embedding_from_pattern([X,Y],rel1(X,Y),tensor(rel1),'transitive_closure_tmp/embedding',npy),
    Gs = [rel2],
    save_expl_graph('transitive_closure_tmp/expl.json','transitive_closure_tmp/flags.json',Gs).
```
```sh
upprism transitive_closure.psm
tprism train --input transitive_closure_tmp --embedding transitive_closure_tmp/embedding.npy.json --cycle --cpu   # `train`, not `test`; needs the subgoal patch
```
R₂ = min1(R₁ + R₁R₂) solved by fixed-point iteration. The 0/1 adjacency matrix is derived from the `rel1/2` facts by
`save_embedding_from_pattern(Axes,Pattern,tensor(rel1),FileBase,Mode)` (Mode `hdf5` default or `npy`); an auxiliary `.txt` maps
axis positions to symbols. `transitive_closure02` builds the matrix with a Python script and sets the size from the command line
(`tensor_atom(rel1,[N,N]) :- dim(N).` + `assert(dim(N))` in `prism_main([M])` using `parse_atom(M,N)`).

## 5. Simulating PRISM: PCFG with `simulated_msw` (`simulating_prism/pcfg.psm`)

```prolog
tensor_atom(onehot(_),[10]).
tensor_atom(sw(_),[10]).                        % logits for each switch

nonterminal(s). nonterminal(x).
values(s,[[x]]).                                % PRISM-style outcome spaces still work as *data*
values(x,[[a,x,a],[b]]).

pcfg(L) :- pcfg(s,L-[]).
pcfg(LHS,L0-L1) :-
    ( nonterminal(LHS) -> simulated_msw(LHS,RHS), proj(RHS,L0-L1)
    ; L0 = [LHS|L1] ).
proj([],L-L).
proj([X|Xs],L0-L1) :- pcfg(X,L0-L2), proj(Xs,L2-L1).

simulated_msw(Sw,Val) :-
    get_values(Sw,Values),            % candidate outcomes (compile-time Prolog)
    nth0(Index,Values,Val),           % nondeterministic choice => sum over outcomes
    p_tensor(Sw,[i]),                 % softmax(logits of Sw)
    tensor(onehot(Index),[i]).        % pick entry Index: dot(prob vector, one-hot)

p_tensor(X,Index) :- operator(softmax), tensor(sw(X),Index).
```
`nth0` enumerates outcomes (T2, on purpose); the dot product with a one-hot selects the outcome's probability. Trained with `nll` and SGD —
there is no EM/outside algorithm here; for a pure PCFG classic PRISM is better (`prism-programming` skill). Use this trick when the
probabilistic part is *combined* with neural/tensor parts.

## 6. Nondeterministic marginalisation (`addition`, MNIST digit addition)

```prolog
output(Y,X1,X2) :- number3(Y1), number3(Y2), number10(Ypred),
                   Ypred is Y1+Y2,
                   tensor(onehot(Ypred),[l]),
                   mnist(X1,Y1), mnist(X2,Y2).
mnist(X,Y)  :- tensor(onehot(Y),[i]), mnist0(X).                    % picks class Y's score of the classifier
mnist0(X)   :- operator(softmax), matrix(w(0),[i,j]), layer1(X).
layer1(X)   :- operator(sigmoid), matrix(w(1),[j,k]), layer2(X).
layer2(X)   :- vector(get(in,X),[k]).
number3(Y)  :- member(Y,[0,1,2]).
number10(Y) :- member(Y,[0,1,2,3,4,5,6,7,8,9]).
```
The Prolog conjunction enumerates all digit pairs, the arithmetic constrains the sum, and the tensor part multiplies the two classifiers'
probabilities for that pair; the sum over solutions is the marginal probability of each total. `l` is the free index (a length-10 vector
over totals); `Y` is the label consumed by the loss. This is the "probabilistic logic programming" use of T-PRISM (compare DeepProbLog).
The goal's value is already a distribution, so do not train it with `ce` (softmax of probabilities) or with an extra `operator(softmax)`
on top, as tutorial §12 does: that version does not learn. A −log p[label] loss does (verified on synthetic digits: sum accuracy 0.78,
and the digit classifier, trained only on sums, 0.80); the full program and loss are in tutorial-recipes.md §12 and python-api.md §9.

## 7. Porting a ProbLog / PRISM model (procedure)

1. Which parts are *structure* (rules over symbols, recursion depth, data joins)? Keep them as Prolog; they run at compile time.
2. Which parts are *numeric* (probabilities, scores, embeddings)? Replace each probabilistic fact/switch by a tensor atom;
   replace "outcome" enumeration by an index that is contracted; wrap in `operator(softmax)` where you need a distribution.
3. Decide the goal format for supervision (`output(X,Label)`, `rel(S,R,O)`, pairs for ranking) and pick the loss (`ce`, `nll`, `preference_pair`, `mse`).
4. Variable parts of goals become placeholders with integer ids; big constant data becomes embedding files.
5. Write `prism_main` (compile-time data prep + `save_expl_graph`), then the three commands: `upprism`, `tprism train`, `tprism test`.
6. Verify one ground goal with `probf/1` before scaling.

---
Verification status (2026-09-29, Docker, commit 169a258): §1 DistMult, §2 MLP (both with and without placeholders; synthetic data), §3 Markov chain (with patch), §4 transitive closure (with patch, `train`) and the `probf` equations of §6 (addition) were executed; §5 PCFG trains but shows the `nll` bug. See `known-issues.md`.
More examples, run from Python and checked on 2026-09-30 (matrix product and decomposition, SVD with orthogonal factors, tensor train,
`get/2`, slicing, MNIST addition that learns, PyTorch MLP/CNN operators): `tutorial-recipes.md`.
