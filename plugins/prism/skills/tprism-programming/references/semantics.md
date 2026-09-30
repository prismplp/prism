# T-PRISM semantics: how to read and write a program

Contents: 1 tensorized least-model semantics · 2 index rules · 3 operators · 4 subgoal / reindex · 5 where non-tensor
goals fit · 6 worked derivations · 7 comparison with PRISM/ProbLog constructs

Sources: `doc/manual/tprism_manual.tex` ch. 2, `src/prolog/up/tensor.pl` (the translator), `bin/tprism/expl_graph.py`.

## 1. Tensorized least-model semantics

Take the ground program's equations `H <=> W1 v W2 v ... v WL`, `Wl = B1 & B2 & ...`. T-PRISM assigns a tensor q(A) to every atom:

- **Tensor atom** `tensor(x,[i,j])` ↦ the tensor named `x`, with its axes labelled `i,j`.
- **Disjunction** (several bodies / several solutions): q(A ∨ B) = q(A) + q(B).
- **Conjunction** of tensor-valued atoms: Einstein summation. Repeated index names ("dummy" indices) are summed out; the
  remaining ("free") indices become the axes of the result. No free indices ⇒ scalar; a conjunction of scalars is their product.
- **Non-tensor atoms** (facts, arithmetic, any predicate with no tensor below it) are proved by Prolog and count as scalar 1.
- **Operator** atoms wrap the result: q(H) = op1 ∘ op2 ∘ … ∘ einsum(remaining atoms), per clause.

Total per predicate instance: `q(H) = Σ_clauses op…(einsum(body atoms))`.

## 2. Index rules

- Indices are **atoms** (`i`, `j`, `k`, `l`, ...), listed as a Prolog list whose length equals the tensor's rank (`tensor_atom` shape).
- Scope of a name = one clause body *plus* whatever predicates it calls (see 4). Two atoms in the same body with the same
  index name must have the same dimension at that axis.
- Free index ⇒ axis of the head's tensor. A predicate whose bodies leave free indices `[j]` is a vector-valued function.
  All clauses of one predicate should leave the same number of free axes with the same sizes, otherwise the sum is ill-defined.
- **Axis order of a result = the free index names in sorted order**, not their order in the body (verified: `t :- tensor(m,[j,i]).`
  gives mᵀ; `tensor(v(a4),[i,a4]), t([a1,a2,a3],i)` gives axes `[a1,a2,a3,a4]`). Pick names so that the sorted order is the one
  a loss or a reader expects.
- An **integer** in an index list selects that position and removes the axis: `tensor(x,[i,1,0])` is `x[:,1,0]`, a vector over `i` (verified).
- Index atoms are ordinary Prolog atoms, so they can be passed as arguments and chosen by clauses, e.g. alternating bond indices
  `t([A,B|R],i) :- tensor(w(A),[i,A,j]), t([B|R],j).` (tensor train, tutorial §07; declare them with `index_atoms/1`).
- `index_atoms([i,j,...])` declares the pool of index symbols. It is required only when an index list is passed
  through a *variable* (e.g. `observe_state(S,Index) :- tensor(onehot(S),Index).` called with `[i]`), because the
  translator collects index symbols by looking at literal lists in `tensor/2` bodies. Declaring extra ones is harmless.
- Legacy: `index_list/1` appears in `exs/tensor/addition/`, but the translator only reads `index_atoms/1`; it is ignored there (works because the indices are literal).
- Tensor **names** (first argument) are ground Prolog terms: `w(0)`, `v(S)` with `S` bound at compile time, `get(in,X)`,
  `onehot(K)`. The pattern in `tensor_atom(v(_),[20])` declares all `v(...)` at once.
- Special names: `get(x,N)` is row `N` of the data tensor `x` (first-axis slice of the embedding `tensor_x_`, verified);
  `onehot(K)` / `onehot(K,N)` is the one-hot vector with a 1 at position `K` (the first number in the name), its length given by
  `tensor_atom(onehot(_),[10])` / `tensor_atom(onehot(_,19),[19])`, so one-hot vectors of several lengths can coexist.

## 3. Operators (`operator(Name)`)

Built-ins (`bin/tprism/op/standard_op.py`): `sigmoid`, `relu`, `softmax` (torch softmax over the **last** axis), `min1` (clamp to [0,1]),
plus the internal `reindex`. Anything else needs a Python class (see `pipeline.md`).

- An operator applies to the einsum of the *other* atoms of the same clause body. Order matters:
  `operator(f), operator(g), A` is f(g(A)), whereas swapping them is g(f(A)).
- Operators are applied per clause, before clauses are summed. To transform a sum, wrap:
  ```prolog
  rel2 :- operator(min1), rel2_helper.       % min1( sum of the helper's clauses )
  rel2_helper :- tensor(rel1,[i,k]).
  rel2_helper :- tensor(rel1,[i,j]), subgoal(rel2,[j,k]).
  ```
- Put `operator/1` first in the body by convention; the manual's formal reading is that the operator atoms wrap the einsum of all non-operator atoms of the clause.

## 4. `subgoal/2` and implicit indices

For a call `p(X,Y)` to a user predicate, the free indices of `p`'s result keep the names they had inside `p`'s clauses. That is why
the MLP example needs no annotation: `layer1` leaves free `j`, and the caller's `matrix(w(0),[i,j])` contracts it.

`subgoal(G,[k,l])` is defined in `tensor.pl` as `subgoal(G,S) :- msw($operator(reindex(S)),$operator), G.`, i.e. `G` with a
`reindex` operator: G's free indices are relabelled (positionally) to `[k,l]`. Use it when

- the same predicate is used more than once in a body with different indices (`transition(T,N)` in a recursion),
- you need to avoid accidental contraction with a same-named index in the caller,
- the callee's own index names are not the ones the caller wants (recursion, mutual recursion).

The predicate arity/functor of every subgoal is collected automatically; nothing to declare.

## 5. Non-tensor goals

Arithmetic, comparisons, `member/2`, user facts (`rel1(a,b).`) and predicates built only from those are executed by Prolog while
building the graph; they contribute 1 but decide **which tensor atoms appear** and which ground arguments they carry. This is
where T-PRISM keeps Prolog's power (recursion, list processing, tests on symbols), and also its danger: backtracking over several
solutions ⇒ several summed terms.

## 6. Worked derivations

**DistMult** `rel(S,R,O) :- tensor(v(S),[i]), tensor(v(O),[i]), tensor(r(R),[i]).`
Free indices: none. `i` is in three atoms ⇒ summed. q(rel(s,r,o)) = Σ_i v_s[i]·v_o[i]·r_r[i] — a scalar. `probf(rel(1,3,0))` prints
`tensor(v(1),[i]) & tensor(v(3),[i]) & tensor(r(0),[i])`.

**MLP** `output(Y,X):-layer0(X,Y). layer0(X,_):-operator(softmax),matrix(w(0),[i,j]),layer1(X). layer1(X):-operator(sigmoid),matrix(w(1),[j,k]),layer2(X). layer2(X):-vector(in(X),[k]).`
layer2 = x (free k). layer1 = sigmoid(Σ_k W1[j,k]·x[k]) (free j). layer0 = softmax(Σ_j W0[i,j]·layer1[j]) (free i) ⇒ a 10-vector of class probabilities.
`Y` (label, first goal argument) occurs in no tensor: it only labels the goal for the loss (`ce`). This is the shipped `mlp0`;
because `ce` applies its own softmax, leave `operator(softmax)` out of `layer0` when training with `ce` (gotchas T11).

**Markov chain** (`exs/tensor/markov_chain`):
```prolog
mc(S,T,N)     :- observe_state(S,[i]), subgoal(transition(T,N),[i]).      % π_S · (P^N π_T)
transition(T,0) :- observe_state(T,[i]).
transition(T,N) :- N>0, NextN is N-1, prob_tensor_msw(tr,[i,j]), subgoal(transition(T,NextN),[j]).
prob_tensor_msw(X,Index) :- operator(softmax), tensor(X,Index).            % row-stochastic matrix
observe_state(S,Index)   :- tensor(onehot(S),Index).
```
`N` is a compile-time integer; recursion unrolls N matrix products into the graph; `subgoal(...,[j])` passes the next state's index down.
The result is the scalar π_Sᵀ Pᴺ π_T.

## 7. Correspondence table

| PRISM / Prolog / ProbLog | T-PRISM |
|---|---|
| `msw(sw(S),V)` with parameters θ | `prob_tensor_msw(sw(S),[i])` (softmax of a learnable vector) contracted with `tensor(onehot(V),[i])` |
| sum over outcomes of a hidden variable | an index name shared by several atoms |
| explicit outcome enumeration `member(V,Vals)` | also allowed, but expands the graph; prefer an index |
| exclusiveness / independence assumptions | not required; sums and products are whatever the equations say |
| EM learning | SGD on a loss (`nll`, `ce`, `mse`, ...); no outside algorithm |
| ProbLog `query`/`evidence` | none; supervision = loss on labelled goals |
| `learn(Gs)` | `save_expl_graph` at compile time + `tprism train` |
