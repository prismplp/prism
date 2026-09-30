# PRISM gotchas (each with a wrong and a right version)

Contents: G1 independence of msw calls · G2 exclusiveness · G3 switch names must be ground · G4 declarations ·
G5 tabling and extra arguments · G6 cuts/if-then-else · G7 negation & failure · G8 observable goals must be a
proper distribution · G9 sampling vs explanation style · G10 values/2 · G11 learn overwrites set_sw ·
G12 EM local optima · G13 cyclic graphs · G14 B-Prolog dialect traps · G15 batch entry point ·
G16 underflow · G17 debugging recipe · G18 goals given to learn/prob · G19 data files written for
other Prolog systems

---

## G1. Every `msw` call is a new independent draw (opposite of ProbLog)

ProbLog: `0.5::coin.` then `two :- coin, coin.` uses the *same* fact twice, so P(two)=0.5.
PRISM: `msw(coin,X), msw(coin,Y)` are i.i.d., so P(X=Y=head)=0.25.

```prolog
values(coin,[head,tail]).
same(X,Y)  :- msw(coin,X), msw(coin,Y).      % two independent tosses
% One toss reused (shared value): bind once, pass around
twice(X,X) :- msw(coin,X).
```
To model "the same random variable appears in several places", draw it once and pass the *value* (mind G5).
To model "different individuals share the same distribution", reuse the same switch name (the blood-type genes do this).

## G2. Explanations must be mutually exclusive

```prolog
values(a,[x,y]). values(b,[x,y]).
% WRONG: both clauses can be true in the same run, PRISM adds the two probabilities (0.5+0.5 = 1.0)
p :- msw(a,x).
p :- msw(b,x).
% RIGHT: make the branches exclusive by construction (case split on the first draw)
p :- msw(a,A), ( A==x -> true ; msw(b,x) ).
```
Under the additive semantics the reported `prob(p,P)` is a *sum over explanations*, not P(a=x ∨ b=x)=0.75.
Symptoms: probabilities > 1, or observable goals not summing to 1. Rule: every `;`, every set of alternative
clauses, and every alternative branch of a subgoal must be exclusive for any parameter setting. Viterbi
computation and Viterbi training do not need exclusiveness; EM/`prob` do.

## G3. Switch names must be ground at call time; dependence is expressed by names

```prolog
% WRONG: switch name contains an unbound variable when msw is called
step(Next) :- msw(tr(S),Next), current(S).
% RIGHT: bind first
step(S,Next) :- msw(tr(S),Next).
```
To make choice C depend on earlier outcomes r1..rk, use a switch named `c(r1,...,rk)`. Then the modeling
assumption "different switches (and different positions) are independent" is exactly the conditional
independence you wanted. In a Mealy-type HMM `msw(out(S,Next),Ob)` must come *after* `msw(tr(S),Next)` binds `Next`.

## G4. Undeclared or shadowed switches

- Every switch name needs a matching `values/2` (use `_` for families: `values(tr(_),[s0,s1])`). No match -> error at `msw`.
- Multiple `values` heads: **first match wins**: put specific names before general ones (`values(f(a,a),...)` before `values(f(X,X),...)` before `values(f(_,_),...)`).
- `values(_, [yes,no])` declares every switch as yes/no — convenient (see `alarm.psm`) but you lose typo detection: a misspelled switch name silently becomes a new binary switch.
- Undeclared switch: `*** PRISM ERROR: outcome space not given -- nosuch` (verified). It is an error, **not** a uniform distribution. A switch that is declared but never set (`set_sw`, `values/3`, `restore_sw`, learning) is uniform, which is a sensible "no prior belief" start (G11).
- Ranges: `values(s,[1-10])`, with skip `[0-9@3]`; range bounds must be integers with Min ≤ Max. Expansion does not sort or dedupe.
- `values/2` bodies may compute outcomes (`values(class,[1-N]) :- num_class(N).`) but are evaluated at *every* `msw`: keep them cheap and side-effect free.
- Outcome variable bound to a value not in the outcome space: explanation search just finds no explanation (goal probability 0 / failure), it does not warn.

## G5. Tabling: keep subgoal argument patterns small

All probabilistic predicates are tabled. Extra arguments that carry history split otherwise-shared subgoals:

```prolog
% WRONG for explanation search (fine for sampling): Seq makes every path a different subgoal -> exponential
hmm(T,N,S,[S|Seq],[Ob|Y]) :- msw(out(S),Ob), msw(tr(S),Next), T1 is T+1, hmm(T1,N,Next,Seq,Y).
% RIGHT: keep hmm/4, extract the state path afterwards
viterbi_states(Obs,States) :-
    viterbif(hmm(Obs),_,E),
    viterbi_subgoals(E,Gs),
    maplist(hmm(_,_,S,_),S,true,Gs,States).
```
Also avoid passing large unrelated terms, counters that don't influence the choices, or output accumulators.
Tabling declarations: `:- p_table p/n.` (only listed predicates tabled) or `:- p_not_table p/n.` (all except these). They cannot co-exist. Use `p_not_table` for deterministic wrapper predicates you never need as subgoals. B-Prolog's own `:- table p/n.` is for *non-probabilistic* predicates.
`prism([consult],File)` loads without compiling and **disables tabling** of probabilistic predicates.

## G6. Cut and if-then-else

Linear tabling with the lazy strategy computes all solutions of a subgoal before continuing, so `p(X), !, q(X)` does not commit to the first solution the way Prolog does. Use arithmetic/guard-based termination clauses and `->` on ground tests:

```prolog
hmm(T,N,_,[]) :- T>N, !.        % OK: the cut only separates a ground test from the recursive clause
```
But do not use cuts to select among *probabilistic* alternatives. If a test is on a not-yet-bound outcome, restructure so the test comes after the switch is drawn (`msw(x,V), ( V==a -> ... ; ... )`).

## G7. Negation, `findall`, `forall` around probabilistic goals

These are not supported, because they break explanation search. Negation is the trap you meet most often when upgrading deterministic rules. After the upgrade, `\+ foo(K,_)` negates an `msw`-backed goal.

**The whole program silently does nothing.** A file with such a clause makes plain `upprism` load nothing (verified; interactive `prism(File)` was not checked). Verified:
- `g_neg :- \+ out(a).` with `out(V) :- msw(sw,V).` in the file;
- `upprism` prints only the banner;
- `prism_main` never runs (not even its first `format`);
- there is no error, and the exit status is 0.

Pick the right replacement:

```prolog
% WRONG: negation of a probabilistic goal
fallback(K) :- \+ out(K,a).
% RIGHT (generative "not a"): draw, then test the drawn value -- prob = 1 - P(a), verified 0.8 for P(a)=0.2
fallback(K) :- out(K,V), V \== a.
% RIGHT (guard): negate only a deterministic fact; fine inside a probabilistic predicate (verified)
plan(Day, stay_home) :- \+ weather(Day, _).
```

If the model really needs "failure", that is, constraints that make some generation runs fail (e.g. "two coins must agree"), write a **failure program**. Define `failure/0` and load the file with `prismn` or `upprism prismn:file`. The FOC compiles the negation away; `upprism prismn:file` also compiles files like the `g_neg` one above ("Compilation done by FOC"). The failure-adjusted EM (FAM) that learning then uses does not work with deterministic-annealing EM (DAEM) (manual). Example:

```prolog
failure :- not(success).
success :- agree(_).
agree(A) :- msw(coin(a),A), msw(coin(b),B), A=B.
% learn([failure|Gs]) — include the atom `failure` in the goal list
```
Without this, learning silently uses the wrong likelihood (probability mass lost to failed runs is ignored). Verified: `upprism prismn:agree 100` runs (FOC compilation, sampling `#success = 100 #failure = 365`, learned `coin(a): heads 0.29`, `coin(b): heads 0.48`); plain `upprism agree 100` on the same file prints only the banner and does nothing useful — no error, so always use `prismn:` for files with `failure/0`.

## G8. Observable goals must form a distribution

The system assumes that, for every parameter setting, the set of observable goal instances is mutually exclusive and sums to 1 (uniqueness condition), and that generating any of them never fails (no-failure condition; MAR relaxes the first for learning). Consequences:

- Don't observe a conjunction of unrelated predicates; wrap it in one predicate (`world(Sm,Re)`).
- Don't mix two different generative stories in the same observable predicate name unless they're exclusive.
- If a generation can fail (guards like `X\==Y`, arithmetic filters that can reject), that's G7.
- Repeated observations: `count(Goal,N)` instead of repeating N times.

## G9. Sampling execution vs explanation search

The same program runs in two ways: top-down sampling (`sample/1`, `get_samples/3`) and exhaustive explanation search (`prob`, `learn`, `viterbif`, `probf`, `hindsight`). A program that only works in one style (e.g. relies on unbound outcome enumeration, or on ordering that only makes sense when outcomes are pre-bound) should be split into a sampling version and an explanation version, or restructured generatively (choices first, deterministic derivations after). Test both: `sample(G)` and `probf(G)`.

## G10. `values/2` is a declaration, not a predicate

Since PRISM 2.0 you cannot call `values(Sw,Vs)` from clause bodies as a fact (verified: a *dynamic* `call(values(gene,V))` raises `existence_error(procedure,values/2)`); use `get_values(Sw,Vs)` (old-style calls in bodies are auto-rewritten at load). Same for `values/3` (declaration with parameter directive: `values(gene,[a,b,o],[0.5,0.2,0.3])`, `fix@[...]`, `set@[...]`, `d@0.5` pseudo counts, `uniform`). Directives run once at load time and only for ground switch names; if the outcome space changes dynamically they will not re-apply.

## G11. `learn` overwrites parameters (unless fixed)

```prolog
set_sw(init,[0.9,0.1]),
learn(Gs)          % init is re-estimated from Gs, the manual values are only the EM starting point
```
Verified on the blood-type model with `max_iterate=1` and `set_sw(gene,[0.9,0.05,0.05])` before `learn`:
- default flags (`init=random`): the result (a=0.354 b=0.201 o=0.446) shows EM restarted from a random point, so your `set_sw` values had no effect;
- `set_prism_flag(init,none)`: EM starts from your values (one iteration gives a=0.430 b=0.183 o=0.387);
- `fix_sw(gene)` after `set_sw`: the values stay untouched by `learn` (`fixed_p: a 0.9 b 0.05 o 0.05`); `unfix_sw(_)` releases them; `values(gene,[a,b,o],fix@[...])` does it in the declaration.
So: to *start* from chosen values use `init=none`; to *keep* values use `fix_sw`.

Other verified behaviours: `prob(Goal,P)` for an *impossible* goal (e.g. `bloodtype(zz)`) **fails** rather than returning 0; `prob` with a non-ground goal such as `bloodtype(_)` works and returns the total (1.0 here); `prob(same(head,head),P)` for `same(X,Y):-msw(coin,X),msw(coin,Y)` is 0.25 (G1); `prob` of the overlapping program in G2 prints `1.0` (bad) vs `0.75` (good).

## G12. EM finds local optima; small data gives degenerate parameters

Observed by an eval agent: learning an HMM from a *single* long string with `init=random` aborted once with "Parameter being zero" on `msw(init,s1)` (the initial-state switch sees only one observation, EM drove it to 0). Untested remedies: several training strings, `init=noisy_u`, pseudo-counts (`default_sw_a`/`d@`), `fix_sw(init)`.

Use `set_prism_flag(restart,N)` (random restarts), pseudo-counts as Dirichlet priors for MAP estimation (`default_sw_d` flag, or `values(...,d@0.5)`), and check `show_sw` for zeros. Learning progress is reported when the `verb`/`em_message` style options are on; see the flags list in the manual.

## G13. Cyclic explanation graphs are an error by default

Left-recursive or looping models (e.g. transitive closure style, random walks with cycles) produce cyclic graphs and raise an error unless `:- set_prism_flag(error_on_cycle,off).` — and probability computation then uses the special cyclic-graph algorithms (see manual chapter "Cyclic explanation graphs"). Prefer acyclic formulations (explicit step counters or ordered indices) unless you need the cyclic semantics. Also avoid creating infinite terms (`X=[a|X]`): B-Prolog has no occurs check and tabling hashes will crash.

## G14. B-Prolog dialect traps

- `maplist` is B-Prolog's lambda form: `maplist(X,Y,(Y is X*2),Xs,Ys)` gives `[2,4]` (verified); `maplist_func`, `maplist_math`, `reducelist` exist. A SWI-style `maplist(succ,[1,2],Ys)` does **not** raise an error: it silently binds `Ys=[]` (verified). B-Prolog also has `foreach` loops and list comprehensions (compiled at load time, faster than `maplist`); see the B-Prolog loops guide for the exact syntax.
- No SWI strings/dicts; atoms and lists only. `format("~w~n",[X])` style works for output.
- Command-line args reach `prism_main/1` as **atoms**: `parse_atom('50',N)`.
- `parse_atom`, `random_select/2-3`, `get_samples/3`, `nth0/3`, `findall/3` (in utility part) are available; verify anything else in the REPL.
- Variables in `msw` outcome lists cannot be non-ground in `values` facts (must be ground).
- Prefix operators `sample`, `prob`, `viterbif`, `hindsight`… have priority 1150, so `?- prob p(X), foo.` may parse unexpectedly; use parentheses.

## G15. Batch entry point

`upprism file args...` calls `prism_main([Arg1,...])`. If both `prism_main/0` and `prism_main/1` are defined only `/1` runs, and with no args it is called with `[]`. `random_set_seed/1` first for reproducibility. Model files with `failure/0` need `prismn:`.

## G16. Underflow on long sequences

Probabilities of long strings underflow doubles. `:- set_prism_flag(log_scale,on).` switches inside/outside computations to log space (learning, `prob`, Viterbi). Do it in the file header, before `learn`.

## G17. Debugging recipe

1. `probf(Goal)` on the smallest ground goal — read the `<=>` equations. Wrong sums? see G2. Exploding graph? see G5.
2. `sample(Goal)` a few times, confirm the story generates sensible data.
3. `prob(Goal,P)` for all observable goals of a tiny instance; they should sum to 1 (else G2/G7/G8).
4. `show_sw` after `learn`, look for switches with counts 0 (unused switch names -> typo in dependency naming).
5. Trace with `prism([consult],file)` + `trace` only for non-probabilistic helpers (tabling is off in consult mode).

## G18. Goals given to `learn/1` and `prob/2`

Every observed goal must meet two conditions. Otherwise **the whole call aborts**; the bad goal is not simply skipped. Both cases are verified.

**1. The goal must be an instance of a probabilistic predicate**, that is, one that can reach `msw`. A goal of a purely deterministic predicate aborts:

```
*** PRISM ERROR: invalid observed goal; tabled probabilistic atomic formula expected -- g_det(k0,a)
error(type_error(probabilistic_atom,g_det(k0,a)),$pp_learn_core/1)
```

`prob/2` aborts in the same way (`type_error(probabilistic_atom,...)`, `(prob)/2`). A goal of a probabilistic predicate whose particular explanation uses no `msw` is accepted. An example is a deterministic fallback clause like `plan(d4,stay_home)` in `examples/activity.psm`. It just carries no learning signal.

**2. The goal must have at least one explanation.** Otherwise:

```
*** PRISM ERROR: no explanations -- g_prob(k1,zz)
error(prism_runtime_error(explanation_not_found),$pp_find_explanations/1)
```

A typical cause is a value outside `values/2`, e.g. from data.

**Filter before learning, and report the count:**

```prolog
filter_explainable([], [], N, N).
filter_explainable([G|Gs], Out, Acc, N) :-
    ( catch(probf(G, _), _, fail) -> Out = [G|Rest], Acc1 = Acc     % probf fails when unexplainable,
    ; Out = Rest, Acc1 is Acc + 1 ),                                % raises for non-probabilistic goals
    filter_explainable(Gs, Rest, Acc1, N).
```

`prob(G,P)` *fails* for an impossible goal. When a caller wants 0 instead, use a wrapper:

```prolog
safe_prob(G,P) :- ( catch(prob(G,P0),_,fail) -> P = P0 ; P = 0.0 ).
```

## G19. Data files written for other Prolog systems

`consult/1` is not a reliable way to load facts written by or for SWI-Prolog. Verified with B-Prolog in PRISM 2.4.2a:
- `:- discontiguous foo/1.` only warns (`Directive ignored`).
- `:- use_module(library(lists)).` aborts the consult (`existence_error(procedure,use_module/1)`).

Read the terms yourself and skip the directives:

```prolog
:- dynamic weather/2.                 % in the .psm: a missing data file then means "no facts", not an error
load_data(File) :- open(File, read, S), load_data_loop(S), close(S).
load_data_loop(S) :-
    read(S, T),
    ( T == end_of_file -> true
    ; ( T = (:- _) -> true ; assert(T) ), load_data_loop(S) ).
```

This was verified with a file containing `:- module(...)`, `:- dynamic ...` and `:- discontiguous ...`.

`load_clauses(File,Cs)` also reads the terms, and is the usual way to get a list of training goals. It returns the directives as terms too (`(:-discontiguous foo/1)`), so filter them out when the file is not yours.
