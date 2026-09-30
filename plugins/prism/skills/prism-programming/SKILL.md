---
name: prism-programming
description: How to write, run, debug and learn parameters for PRISM programs (probabilistic Prolog on B-Prolog, .psm files, prismplp/prism), with the differences from ProbLog and plain Prolog spelled out, plus two recipes - turning a deterministic Prolog rule base into a PRISM model whose default reproduces the rules and is then refined by EM, and driving a .psm from shell scripts, other languages and pipelines (prism_main/1 modes, parseable output, exit codes). Use it whenever the user mentions PRISM, .psm, msw/2, values/2, upprism, learn/1, save_sw/restore_sw, viterbif, hindsight, probf, graphical EM, 確率的推論 or probabilistic prolog, asks for an HMM, PCFG, Bayesian network or other model in PRISM, translates ProbLog or Prolog into PRISM, or debugs an msw, negation or learn error. Read it before writing any PRISM code, because ProbLog and Prolog intuitions (shared random variables, negation) silently give wrong answers. For T-PRISM use tprism-programming, for Python programming-pyprism and programming-prism-ds.
---

# Writing PRISM programs

PRISM is a **generative** probabilistic logic language on top of B-Prolog (not
SWI-Prolog). You write a Prolog program that *simulates how data is
generated*, drawing every random choice from a named "random switch" with
`msw/2`. From that same program, the system derives probabilities, Viterbi
paths, hindsight probabilities and EM-learned parameters by tabled explanation
search.

- Repository: https://github.com/prismplp/prism
- Manual: `doc/manual/manual.tex`, also as HTML on the GitHub Pages site
- Examples: `exs/`
- Everything marked verified was run on PRISM 2.4.2a (commit 169a258) on Linux amd64. Use Docker on macOS (`../tprism-programming/references/Dockerfile`). `upprism file.psm` with a `prism_main` is the easiest way to script checks.

## 1. Unlearn these ProbLog / Prolog habits first

| If you think in ProbLog or Prolog | In PRISM |
|---|---|
| `0.3::rain.`: a probabilistic fact with a number in the program | No numbers in the model. Declare an outcome space, `values(rain,[yes,no]).`, and draw with `msw(rain,X)`. Numbers are *parameters* held by the system: `set_sw/2`, `values/3`, or learned by `learn/1`. The default is uniform. |
| A probabilistic fact used twice is **one** shared random variable | Every `msw` **call is a fresh independent draw**, even with the same switch name. To share a value, bind one variable and pass it around. |
| Overlapping proofs are fine | Explanations of a goal **must be mutually exclusive**. PRISM just adds their probabilities, so overlaps give silently wrong numbers (gotcha G2). |
| `query(g).` and `evidence(e,true).` | Use `prob/2`, `viterbif/3`, `hindsight/3` and `chindsight_agg/2`. Evidence goes into the arguments of the observed goal. |
| `0.9::a :- b.`: dependent choice | Dependence is expressed by **different switch names per context**: `msw(tr(State),Next)`, `msw(al(Fire,Tamper),Alarm)`. |
| `\+`, `findall` or `forall` around any goal | **Never around probabilistic goals.** A `\+` of a probabilistic goal makes plain `upprism` run nothing, with no error (verified; G7). Negate only deterministic facts, or draw and then test. |
| `assert/retract` freely | Never inside probabilistic predicates. Use them only in utility code. `values/2` bodies must be free of side effects. |
| SWI-Prolog libraries and directives (`use_module`, strings, `forall/2`, `maplist/3` with a closure) | B-Prolog dialect (G14). Data files written for SWI need a loader (G19). |

## 2. Anatomy of a program

A `.psm` file has three parts:
- *declarations* (`values/2`);
- the *modeling part*: probabilistic predicates, which call `msw` directly or indirectly;
- the *utility part*: ordinary Prolog for data preparation, learning drivers, printing and `prism_main`.

```prolog
%% declarations (outcome spaces)
values(init,[s0,s1]).
values(out(_),[a,b]).          % family of switches out(s0), out(s1), ...
values(tr(_),[s0,s1]).

%% modeling part: generative, probabilistic predicates
hmm(L):- msw(init,S), hmm(1,S,L).
hmm(T,_,[]):- T>10, !.
hmm(T,S,[Ob|Y]):-
    msw(out(S),Ob),            % emission depends on S  -> switch named by S
    msw(tr(S),Next),           % transition depends on S
    T1 is T+1,
    hmm(T1,Next,Y).

%% utility part: ordinary Prolog, never called during explanation search
prism_main([]):-
    random_set_seed(1234),
    get_samples(50,hmm(_),Gs),
    learn(Gs),
    show_sw.
```

Rules of thumb that keep you inside the modeling assumptions:

1. **Write top-down, as the story of how an observation is produced.** Observed data are *instances of the head* of a probabilistic predicate (`hmm([a,b,b])`).
2. **Every switch name must match a `values/2` head.**
   - Use `_` for families.
   - The first matching head wins.
   - An undeclared switch is an error. A declared switch that was never set is uniform.
3. **Switch names must be ground when `msw` is called.** The outcome may be unbound, and explanation search then enumerates it.
4. **Keep the arguments of tabled goals minimal.** Do not thread a "path so far" accumulator through the recursion (G5). Recover paths from the Viterbi explanation afterwards.
5. **Probabilistic predicates are tabled.** Cuts do not behave as in Prolog, and cyclic terms crash. Use guards on ground tests, and check with `probf/1`.
6. **Keep deterministic helpers pure**, so they do not break exclusiveness.

## 3. Running

| Task | How |
|---|---|
| Interactive | Run `prism`, then `?- prism(model).`, which loads `model.psm`. `prism([consult],model)` is for tracing only, because nothing is tabled then. |
| Batch | Define `prism_main/1` and run `upprism model arg1 arg2`. The arguments arrive as atoms (`parse_atom/2`). If both `/0` and `/1` exist, only `/1` runs. Details: §6. |
| Failure programs (`failure/0`, negation) | `upprism prismn:model` or `prismn/1` (G7). |
| Multiple files | Probabilistic predicates in other files need `:- include('other.psm').` |
| From Python | Use pyprism; see the programming-pyprism skill. |

A typical session:

```prolog
?- prism(bloodtype).
?- set_sw(gene,[0.3,0.2,0.5]).           % optional; default uniform
?- sample(bloodtype(X)).                  % forward sampling
?- prob(bloodtype(a),P).                  % exact probability (fails for an impossible goal)
?- learn([count(bloodtype(a),40),count(bloodtype(b),20),count(bloodtype(o),30),count(bloodtype(ab),10)]).
?- show_sw.                               % learned parameters
?- save_sw('params.sw').                  % ... and restore_sw('params.sw') in a later run
?- viterbif(hmm([a,a,b])).                % most probable explanation, printed
?- probf(hmm([a,a,b])).                   % the explanation graph (best debugging tool)
?- hindsight(hmm([a,b,a,b]),hmm(2,_,_,_),Ps).                       % P(subgoal, goal)
?- chindsight_agg(world(_,_,_,yes,_,no),world(_,_,query,yes,_,no)).  % P(Alarm | evidence)
```

About learning:
- `learn/1` takes goals or `count(Goal,N)`. Every goal must be an instance of a probabilistic predicate and must have an explanation; otherwise the whole call aborts (G18).
- Useful flags (`set_prism_flag/2`):
  - `init`: `none` starts EM from the current parameters (G11).
  - `restart`, `max_iterate`, `epsilon`
  - `default_sw_d`: pseudo counts, for MAP estimation
  - `learn_mode` (`ml`, `vb`, ...) and `log_scale` (G16)
  - `error_on_cycle`
- `learn_statistics/2` gives `log_likelihood`, `bic`, `free_energy` and more after learning. See `references/builtins.md`.

## 4. Translating a ProbLog model (procedure)

1. **Give each probabilistic fact or annotated disjunction a switch and an outcome space.** `0.3::a.` becomes `values(a,[t,f])` with `msw(a,X)`.
2. **Turn each *use* of a fact into a fresh `msw` call.** If the model needs one shared variable, draw it once at the top and pass the value down (mind §2 rule 4).
3. **Make dependencies explicit as parameterized switch names**, e.g. `msw(alarm(B,E),A)`.
4. **Push the evidence into the arguments of the observed goal**, e.g. `world(Sm,Re)`.
5. **Check exclusiveness for every disjunction.** Restructure overlapping branches (G2).
6. **Check the result.**
   - Call `probf(G)` on a tiny ground goal and read the `<=>` lines.
   - Check that `prob/2` sums to 1 over the observable goals.

A side-by-side example is §7 of `references/examples.md`.

## 5. Recipe: make a deterministic rule base probabilistic

The easiest way to *design* a PRISM model is often to write, or reuse, the
deterministic Prolog rules first, and then turn each hard-coded mapping into a
switch. The default behaviour then reproduces the rules, and EM refines it
from data.

1. Rename the mapping facts to `foo_default/N`. They are the prior.
2. Declare the outcome space: every value a default uses, plus the values the data may show.
3. Redefine `foo/N` with the same name and arity as `msw`: `foo(K,V) :- msw(foo_sw(K),V).`. Callers stay unchanged.
4. **Fix the negated guards.** `\+ foo(...)` would now negate a probabilistic goal, which is not allowed (G7). Guard on a deterministic fact instead. This usually also keeps the rules exclusive.
5. Seed a peaked distribution from the defaults with `set_sw`. Switches without a default stay uniform.
6. Learn: `set_prism_flag(init,none)`, filter out goals without an explanation (G18), then `learn`, and finally `save_sw`.

The full recipe is in `references/deterministic-to-probabilistic.md`. A
verified, runnable example with three CLI modes is in `examples/activity.psm`
(§8 of `references/examples.md`).

## 6. Driving PRISM from scripts and pipelines

- **One entry point with modes.** `prism_main([learn, Params | Data])`, `prism_main([infer, Params, Data | Query])` and `prism_main([Data | Query])` (defaults only). Add a usage clause as the last one.
- **Print one self-describing line per result**, e.g. `DIST:Key:Value:Prob`. Never print raw terms that the caller has to parse as Prolog.
- **Learned state lives in files.** Use `save_sw` / `restore_sw`, one parameter file per data split. Mask held-out labels with `retractall/1` before querying.
- **Exit status.**
  - An uncaught exception gives exit status 1 and `Aborted by exception -- ...`.
  - A failing `prism_main` gives exit status **0** with the last line `no`.
  - A program that PRISM refused to load because of G7 also gives exit status 0, and prints only the banner.
  - So have the caller check for its expected tags, not only the exit status.

Details and a CLI skeleton: `references/batch-and-pipelines.md`.

## 7. Debugging checklist

1. Call `probf(Goal)` on the smallest ground goal and read the `<=>` equations.
   - Wrong sums point to G2.
   - A graph that explodes points to G5.
2. Call `sample(Goal)` a few times, and check that the story generates sensible data.
3. `prob/2` of all the observable goals of a tiny instance should sum to 1. If not, see G2, G7 and G8.
4. After `learn`, look at `show_sw` for zero or unused switches. They often come from typos in the switch names.
5. **The program prints nothing at all, not even the `prism_main` output?** Look for `\+` or `not` around a probabilistic goal (G7).

## 8. Where to look next

| File | Read it when |
|---|---|
| `references/gotchas.md` | A result looks off, or before writing anything non-trivial. G1–G19, each with wrong and right code. |
| `references/examples.md` | You want a complete, idiomatic program: coin, blood type, HMM, PCFG, Bayesian network with evidence, failure program, ProbLog translation, deterministic-to-probabilistic. |
| `references/builtins.md` | You need a built-in or a flag: sampling, learning, switches, B-Prolog list and loop constructs. |
| `references/deterministic-to-probabilistic.md` | You are upgrading existing Prolog rules (§5). |
| `references/batch-and-pipelines.md` | The program is called from a shell script, CI job or another language (§6). |
| `examples/activity.psm` (+ `data.pl`, `goals.pl`) | You need a runnable template for §5 and §6. |

Authoritative details are in the repository's `exs/base/*.psm`, in
`doc/manual/manual.tex` and in `testing/programs/`. For tensors, embeddings
or neural networks, use tprism-programming.
