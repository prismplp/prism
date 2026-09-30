# PRISM examples (adapted from the repo's `exs/base/` and the manual)

Contents: 1 coin · 2 blood type (partial observation, EM) · 3 HMM · 4 PCFG with list-valued outcomes ·
5 Bayesian network + conditional query · 6 failure program · 7 ProbLog → PRISM translation ·
8 deterministic rules → probabilistic model, with a batch CLI (runnable: `../examples/activity.psm`)

These follow the structure of the programs shipped in the repo; they were not re-executed in this
environment, so if you adapt one, `probf/1` a tiny goal first (see gotchas G17).

---

## 1. Coin / direction (smallest model)

```prolog
values(coin,[head,tail]).

direction(D) :- msw(coin,Face), ( Face == head -> D = left ; D = right ).
```
```
?- prism(direction).
?- set_sw(coin,[0.7,0.3]).
?- sample(direction(D)).
?- prob(direction(left),P).          % P = 0.7
```

## 2. Blood type: hidden genes, EM from phenotype counts

```prolog
values(gene,[a,b,o]).

bloodtype(P) :-
    genotype(X,Y),
    ( X=Y -> P=X
    ; X=o -> P=Y
    ; Y=o -> P=X
    ; P=ab ).

genotype(X,Y) :- msw(gene,X), msw(gene,Y).     % same switch, two independent draws (random mating)
```
```
?- learn([count(bloodtype(a),40),count(bloodtype(b),20),count(bloodtype(o),30),count(bloodtype(ab),10)]).
?- show_sw.
```
The genotype is never observed; EM estimates gene frequencies from the phenotype. The `->` chain is
exclusive (each genotype pair leads to exactly one phenotype), satisfying G2.

## 3. HMM (Moore type), learning, Viterbi, hindsight

```prolog
values(init,[s0,s1]).
values(out(_),[a,b]).
values(tr(_),[s0,s1]).

str_length(10).

hmm(L) :- str_length(N), msw(init,S), hmm(1,N,S,L).
hmm(T,N,_,[]) :- T>N, !.
hmm(T,N,S,[Ob|Y]) :-
    msw(out(S),Ob),
    msw(tr(S),Next),
    T1 is T+1,
    hmm(T1,N,Next,Y).

set_params :-
    set_sw(init,[0.9,0.1]),
    set_sw(tr(s0),[0.2,0.8]), set_sw(tr(s1),[0.8,0.2]),
    set_sw(out(s0),[0.5,0.5]), set_sw(out(s1),[0.6,0.4]).

hmm_learn(N) :- set_params, !, get_samples(N,hmm(_),Gs), !, learn(Gs).

viterbi_states(Obs,States) :-
    viterbif(hmm(Obs),_,E),
    viterbi_subgoals(E,E1),
    maplist(hmm(_,_,S,_),S,true,E1,States).

prism_main([Arg]) :- parse_atom(Arg,N), hmm_learn(N).
```
Run: `upprism hmm 100`. Hidden-state posterior at time 2:
`?- hindsight(hmm([a,b,a,b]),hmm(2,_,_,_),Ps).` (the string length must then be 4).
Note: the recursion has only `T,N,S,Remaining` as arguments — no accumulator (G5).

## 4. PCFG: outcomes can be any ground term (here: right-hand sides as lists)

```prolog
values(s,[[np,vp],[vp]]).
values(np,[[noun],[noun,pp],[noun,np]]).
values(vp,[[verb],[verb,np],[verb,pp],[verb,np,pp]]).
values(pp,[[prep,np]]).
values(verb,[[swat],[flies],[like]]).
values(noun,[[swat],[flies],[ants]]).
values(prep,[[like]]).

:- p_not_table proj/2.          % keep explanation graphs readable

pcfg(L) :- pcfg(s,L-[]).
pcfg(LHS,L0-L1) :-
    ( nonterminal(LHS) -> msw(LHS,RHS), proj(RHS,L0-L1)
    ; L0 = [LHS|L1] ).           % terminal: consume one symbol (difference list)
proj([],L-L).
proj([X|Xs],L0-L1) :- pcfg(X,L0-L2), proj(Xs,L2-L1).

nonterminal(s). nonterminal(np). nonterminal(vp). nonterminal(pp).
nonterminal(verb). nonterminal(noun). nonterminal(prep).
```
The choice of expansion is a switch named by the nonterminal; the parse structure is recovered from the
explanation (`viterbif/3`), not from an accumulator. Learn rule probabilities with `learn([pcfg([swat,flies,like,ants]), ...])`.
`:- set_sw(np,[0.4,0.4,0.2]).` as a directive fixes initial parameters at load time.

## 5. Bayesian network with evidence (conditional query)

```prolog
values(_,[yes,no]).                     % every switch is binary

world(Fi,Ta,Al,Sm,Le,Re) :-
    msw(fi,Fi),            msw(ta,Ta),
    msw(sm(Fi),Sm),
    msw(al(Fi,Ta),Al),     % CPT P(Alarm | Fire, Tampering): one switch per parent configuration
    msw(le(Al),Le),
    msw(re(Le),Re).

world(Sm,Re) :- world(_,_,_,Sm,_,Re).   % marginal over hidden variables
```
```
?- set_sw(fi,[0.1,0.9]), ... .            % CPTs: one set_sw per switch instance
?- chindsight_agg(world(_,_,_,yes,_,no), world(_,_,query,yes,_,no)).
% conditional hindsight probabilities:  P(Alarm = x | Smoke=yes, Report=no)
```
Evidence goes in the *top goal* (`world(_,_,_,yes,_,no)`), the query variable in the *pattern*
(`query` marks the argument to aggregate). This replaces ProbLog's `evidence/2` + `query/1`.
The naive joint is exponential in network size; for big networks the manual describes a junction-tree style
encoding (`exs/base/jtree`).
Learning CPTs from data: `get_samples(N,world(_,_),Gs), learn(Gs)`; protect parameters with `fix_sw(fi)`.

## 6. Failure program (constraint "both coins must agree")

```prolog
values(coin(_),[heads,tails]).

failure :- not(success).
success :- agree(_).

agree(A) :- msw(coin(a),A), msw(coin(b),B), A=B.      % runs with A\=B fail = lost probability mass

prism_main([Arg]) :-
    set_sw(coin(a),[0.8,0.2]), set_sw(coin(b),[0.1,0.9]),
    parse_atom(Arg,N),
    get_samples_c([inf,N],agree(_),true,Gs),          % conditional sampling: only successful runs
    learn([failure|Gs]),                              % note the atom `failure` in the goal list
    show_sw.
```
Run with `upprism prismn:agree 100` (from `exs/fail/agree.psm`). Without the failure machinery,
learning would treat rejected runs as if they never happened, which biases the parameters.

## 7. ProbLog → PRISM, side by side

ProbLog:
```prolog
0.1::burglary.
0.2::earthquake.
0.9::alarm :- burglary, earthquake.
0.8::alarm :- burglary, \+earthquake.
0.1::alarm :- \+burglary, earthquake.
calls(mary) :- alarm.
evidence(calls(mary),true).
query(burglary).
```
PRISM (generative, dependencies by switch name, evidence as observed goal argument, no `\+`):
```prolog
values(burglary,[t,f]).
values(earthquake,[t,f]).
values(alarm(_,_),[t,f]).                 % one switch per parent configuration
values(calls_given(_),[t,f]).             % Mary calls given the alarm state

world(B,E,A,C) :-
    msw(burglary,B), msw(earthquake,E),
    msw(alarm(B,E),A),
    msw(calls_given(A),C).
observe(C) :- world(_,_,_,C).             % wrapper: only the observable argument, hidden variables marginalised

prism_main([]) :-
    set_sw(burglary,[0.1,0.9]), set_sw(earthquake,[0.2,0.8]),
    set_sw(alarm(t,t),[0.9,0.1]), set_sw(alarm(t,f),[0.8,0.2]),
    set_sw(alarm(f,t),[0.1,0.9]), set_sw(alarm(f,f),[0.0,1.0]),
    set_sw(calls_given(t),[1.0,0.0]), set_sw(calls_given(f),[0.0,1.0]),
    chindsight_agg(observe(t), world(query,_,_,t)).   % P(Burglary | calls = t): t 0.82, f 0.18 (verified)
```
Differences to notice:
- `\+burglary` became an explicit outcome `f`; every "clause with a probability" became one row of a CPT switch
  (`alarm(B,E)`), and ProbLog's implicit "no clause fires means alarm is false" became the explicit row `alarm(f,f)=[0.0,1.0]`.
- ProbLog's deterministic `calls(mary) :- alarm` became a switch with parameters fixed to 1.0/0.0, or simply `C = A`.
- Evidence `calls(mary)=true` is the observed argument of the top goal `observe(t)`; the ProbLog query is the `query`
  marker in the pattern given to `chindsight_agg/2`.


## 8. Deterministic rules → probabilistic model, with a batch CLI

The files are `../examples/activity.psm` (model and CLI), `../examples/data.pl` (facts in SWI style, with directives) and `../examples/goals.pl` (training observations). The recipe is in `deterministic-to-probabilistic.md`.

The deterministic original:

```prolog
activity(sunny, picnic).  activity(rainy, reading).
plan(Day, A)         :- weather(Day, W), activity(W, A).
plan(Day, stay_home) :- \+ (weather(Day, W), activity(W, _)).
```

The probabilistic version (the core of `activity.psm`):

```prolog
:- dynamic weather/2.
activity_default(sunny, picnic).          % the old facts become the prior
activity_default(rainy, reading).
values(act(_), [picnic, reading, skiing]).
activity(W, A) :- msw(act(W), A).         % same name/arity, callers unchanged
plan(Day, A)         :- weather(Day, W), activity(W, A).
plan(Day, stay_home) :- \+ weather(Day, _).   % guard on the deterministic fact only (gotcha G7)
% + set_params (peaked 0.6 on the default), load_data (skips directives), filter_explainable,
%   learn_and_save (init=none), safe_prob, print_dist, prism_main/1 with three modes
```

Verified runs, in `examples/`:

```
$ upprism activity.psm data.pl d1        # defaults: sunny -> picnic 0.6, others 0.2
DIST:d1:picnic:0.6000
DIST:d1:reading:0.2000
DIST:d1:skiing:0.2000
DIST:d1:stay_home:0.0000
$ upprism activity.psm data.pl d3        # snowy has no default -> uniform 0.3333 each
$ upprism activity.psm data.pl d4        # no weather fact -> stay_home 1.0000
$ upprism activity.psm learn params.sw data.pl goals.pl
LEARN:goals:8:skipped:1                  # plan(d1,stay_home) has no explanation (gotcha G18)
$ upprism activity.psm infer params.sw data.pl d2
DIST:d2:reading:0.3333
DIST:d2:skiing:0.6667                    # learned from goals.pl (EM started from the prior)
```

Things to notice:
- The training goal `plan(d4,stay_home)` has an explanation without `msw` and is accepted. `plan(d1,stay_home)` has none and is filtered out, so it does not abort `learn`.
- With an empty data file, the `:- dynamic` declaration makes `weather/2` simply fail, and every day gets `stay_home` with probability 1.0 (verified).

---
Verification status (2026-09-29/30, PRISM 2.4.2a, commit 169a258, in Docker): §8 (all three CLI modes, output above) and §1–§3 (coin, blood type learning gene≈a 0.292 b 0.163 o 0.545, HMM learn + `viterbi_states`, `hindsight`), §4 PCFG (`prob`, `learn`, `viterbif`), §5 Bayesian network (`chindsight_agg` gives 0.6208/0.3792, matching the manual), §6 failure program (`prismn:`), §7 ProbLog translation (`chindsight_agg(observe(t),world(query,_,_,t))` ⇒ P(burglary)=0.82) were executed. `the `observe/1` wrapper of §7 was used because it is the pattern the manual recommends (`world(Sm,Re)` wrapper). Zero parameters such as `alarm(f,f)=[0.0,1.0]` worked here; not tested with `log_scale=on`.
