# Recipe: turn a deterministic Prolog rule base into a PRISM model

This recipe is often the easiest way to *design* a PRISM program:
1. Get the deterministic rules right first. They are easier to write and to test.
2. Turn every hard-coded mapping into a random switch.
3. Seed the switches so that the default behaviour reproduces the rules.
4. Refine the parameters by EM from data.

A runnable, verified example is `../examples/activity.psm`. It is walked
through in §8 of `examples.md`.

Contents: 1 recipe · 2 fixing guards · 3 seeding · 4 training goals ·
5 learn and save · 6 structuring larger rule bases · 7 checklist

---

## 1. The recipe

Start from a deterministic module:

```prolog
category_outcome(k1, a).
category_outcome(k2, b).
result(X, V) :- item(X, K), category_outcome(K, V).
```

1. **Keep the mapping facts under a new name.** They document the prior, and they seed it:
   ```prolog
   category_outcome_default(k1, a).
   category_outcome_default(k2, b).
   ```
2. **Declare the outcome space.** It must contain every value that any default uses, plus the values you expect the data to show:
   ```prolog
   values(cat_switch(_), [a, b, c, d]).
   ```
   A value that is missing here but occurs in the data makes that goal unexplainable, and aborts `learn` (gotcha G18).
3. **Redefine the old predicate on top of the switch**, with the same name and arity. Every caller stays unchanged:
   ```prolog
   category_outcome(K, V) :- msw(cat_switch(K), V).
   ```
4. **Fix the guards that negate the wrapped predicate** (§2). This is the only rule change that is usually needed.
5. **Seed a peaked distribution** from the defaults (§3), then learn (§4 and §5).

## 2. Fixing negated guards

A deterministic rule like

```prolog
result(X, fallback) :- \+ (item(X, K), category_outcome(K, _)).
```

means "no *covered* key". An uncovered key used to fail to unify with the fact
table. After step 3, `category_outcome/2` succeeds for **every** key, with some
distribution, and it is probabilistic. So the old guard would now negate a
probabilistic goal. That is not allowed: the file then silently loads nothing
(gotcha G7).

The faithful rewrite negates only deterministic data:

```prolog
result(X, fallback) :- \+ item(X, _).
```

Because every key now has an outcome distribution, "no key at all" is exactly
the case the old rule handled. It also keeps the clauses of `result/2`
mutually exclusive (gotcha G2).

Whenever you upgrade a module, grep it for `\+` and `not`. Check whether each
negated goal transitively reaches a predicate that is now probabilistic. If
the rule really needs "the probabilistic event did not happen", draw and test
(`category_outcome(K, V), V \== a`), or model the failure explicitly with a
failure program (`prismn`, G7).

## 3. Seeding: reproduce the deterministic behaviour

Put mass `P0` on the default value, and split the rest uniformly over the other outcomes:

```prolog
default_peak(0.6).
set_params :- ( default_map(Sw, D), set_peaked_sw(Sw, D), fail ; true ).
default_map(cat_switch(K), V) :- category_outcome_default(K, V).   % one clause per switch family
set_peaked_sw(Sw, Default) :-
    get_values(Sw, Vs), length(Vs, N), default_peak(P0),
    ( N =:= 1 -> Dist = [1.0]
    ; Rest is (1.0 - P0) / (N - 1), peaked_dist(Vs, Default, P0, Rest, Dist) ),
    set_sw(Sw, Dist).
peaked_dist([], _, _, _, []).
peaked_dist([V|Vs], D, P0, R, [P|Ps]) :- ( V == D -> P = P0 ; P = R ), peaked_dist(Vs, D, P0, R, Ps).
```

- Switch instances without a default fact stay uniform, meaning "no prior belief, let the data decide". This was verified in `activity.psm`: `act(snowy)` gives 1/3 each.
- Call `set_params`, or `restore_sw(File)` with learned parameters, at the start of **every** entry point. Parameters are per process, and nothing survives between `upprism` runs.

## 4. Training goals

Training goals are ground instances of a **probabilistic** top-level predicate. There are two usual sources.

**Labeled facts already in the data:**

```prolog
findall(result(X, V), labeled(X, V), Goals)
```

**An explicit goals file.** Use this for held-out splits and cross-validation:

```prolog
load_clauses('train_goals.pl', Goals)      % a file of goal facts; filter directives if not yours (G19)
```

Then filter out the goals without an explanation, and report how many were
dropped (gotcha G18). A goal that is explained only through a deterministic
clause of a probabilistic predicate is accepted, but it carries no learning
signal.

If you need a probabilistic training goal and the query predicate also has
deterministic shortcuts (e.g. "if the label is given explicitly, return it"),
keep two predicate families:
- one for **querying**, which includes the shortcuts;
- one for **training**, which contains only the clauses that go through a switch.

## 5. Learn and save

```prolog
learn_and_save(All, ParamFile) :-
    set_params, set_prism_flag(init, none),        % start EM from the prior, not from a random point
    filter_explainable(All, Goals, 0, Skipped), length(Goals, N),
    format("LEARN:goals:~w:skipped:~w~n", [N, Skipped]),
    ( N =:= 0 -> true ; learn(Goals) ),
    save_sw(ParamFile).
```

- Without `init=none`, EM starts from a random point, so the seeded prior has no effect (gotcha G11, verified).
- To keep some switches exactly at their hand-set values, use `fix_sw/1`.
- With little data, the ML estimates contain zeros. For MAP estimates, add pseudo counts: `set_prism_flag(default_sw_d, 1.0)` before `learn`.

## 6. Structuring larger rule bases

- **Query wrappers with fixed arity.** If the rules carry bookkeeping arguments, such as which rule fired, add a wrapper that drops them for querying: `answer(X, V) :- result(X, V, _Source).`. `prob/2` then sums over the sources. The sources must be exclusive (G2).
- **Avoid mutual recursion between probabilistic branches.** An example is two entities whose outcomes are derived from each other. This creates cyclic explanation graphs (G13), or loops. Give each rule that needs "the other's value" a helper that computes it *without* the mutual rules.
- **Keep the domain knowledge in the fact tables** (`*_default`), and keep the PRISM mechanics (switch declarations, seeding, the learning driver, the CLI) generic. Then adapting the model to a new task means replacing the tables.
- **Use `safe_prob/2`** (`catch` + a default of 0.0, G18) in the reporting code, so that an impossible combination prints 0 instead of failing the whole report.

## 7. Checklist

1. Every hard-coded mapping is a switch, with its old facts kept as `*_default`.
2. `values/2` covers all default values and all values in the data.
3. No `\+` or `not` reaches a probabilistic predicate. Grep for them, and check that `upprism` prints your `prism_main` output at all.
4. `probf/1` on a small goal shows exclusive alternatives, and `prob/2` of the alternatives sums to 1.
5. `set_params` (or `restore_sw`) runs at every entry point, and `init=none` is set before `learn`.
6. The training goals are filtered, and the number skipped is printed.
