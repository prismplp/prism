# PRISM built-ins cheat sheet

Authoritative source: `doc/manual/manual.tex` chapter "PRISM built-in utilities" (search by predicate name).
Listed here are the ones you will reach for first. `Goal` arguments must be probabilistic goals (heads of modeling-part predicates).

## Loading / running
| | |
|---|---|
| `prism(File)`, `prism(Opts,File)` | compile+load `File.psm`. Opts: `compile` (default), `consult` (no tabling!), `load` (`.psm.out`), `v`/`nv`, or `Flag=Value` pairs (e.g. `log_scale=on`) which override in-file `set_prism_flag`. |
| `prism_main/0,1` | batch entry for `upprism`; args are atoms (`parse_atom/2`). |
| `prism_help`, `show_sw`, `show_sw_a`, `show_sw_d` | usage; show switch parameters / hyperparameters. |

## Switches and parameters
| | |
|---|---|
| `values(Sw,Outcomes)` | declaration. `_` families, ranges `[1-10]`, `[0-9@3]`, body allowed. Not callable. |
| `values(Sw,Outcomes,Directive)` | + `[p1,..]`, `set@[..]`, `fix@[..]`, `uniform`, `d@0.5` (pseudo counts), `(uniform,d@0.5)` |
| `get_values(Sw,Vs)` | runtime access to an outcome space |
| `msw(Sw,V)` | draw (sampling) / enumerate (explanation search) |
| `set_sw(Sw,Ps)`, `set_sw(Sw)` | set parameters / reset to default (uniform); `Sw` may be a non-ground pattern to affect all matches: `set_sw(_)` |
| `get_sw(Sw,Info)`, `get_sw_a/d`, `set_sw_a/d` | inspect; hyperparameters (pseudo counts) |
| `fix_sw(Sw)`, `fix_sw(Sw,Ps)`, `unfix_sw(Sw)` | keep parameters constant during learning |
| `save_sw(File)`, `restore_sw(File)` | save the current parameters to a file / load them back (verified). Needed whenever parameters must survive the process, e.g. learn in one `upprism` run, infer in the next, or between pyprism queries |

## Inference
| | |
|---|---|
| `sample(G)`, `get_samples(N,G,Gs)`, `get_samples_c([inf,N],G,Cond,Gs)` | forward sampling; the `_c` version keeps only samples satisfying `Cond` |
| `prob(G,P)`, `probf(G)`, `probf(G,F)`, `print_graph(F)` | probability; explanation graph (print or as term) |
| `viterbi(G,P)`, `viterbif(G)`, `viterbif(G,P,E)`, `viterbi_subgoals(E,Gs)`, `n_viterbi(N,G,P)` (verified), `viterbit/1,3` (Viterbi tree) | most probable explanation, top-N, tree form |
| `viterbig(G)`, `viterbig(G,P)`, `viterbig(G,P,E)` | like `viterbif` but **binds the variables of a non-ground `G`** to the most probable explanation. E.g. `viterbig(obs(Y,L))` = the most probable class `Y` (verified) |
| `log_prob(G,LP)` | log probability, for goals whose probability would underflow (verified) |
| `hindsight(G,Pat)`, `hindsight(G,Pat,Ps)`, `chindsight(G,Pat)`, `chindsight_agg(G,PatWithQuery)` | posterior over subgoals, conditional, aggregated over the argument tagged `query`. The `/2` forms print; `hindsight/3` returns `[[SubGoal,P],...]` (verified) |
| `learn(Gs)`, `learn`, `count(G,N)` | MLE/MAP via graphical EM (also VB and Viterbi training via `learn_mode`, `viterbi_mode` flags). Goals must be probabilistic and explainable (gotcha G18) |
| `learn_b(Gs)` | variational Bayes learning (verified); objective via `learn_statistics(free_energy,F)` |
| (manual chapters "Variational Bayesian learning", "MCMC sampling") | Viterbi training (`viterbi_mode`/`learn_mode=ml_vt`), variational Bayes (`set_prism_flag(learn_mode,vb)`), MCMC |
| `learn_statistics(Name,Value)` | after learning (fails before any learning). Names: `log_likelihood`, `log_post`, `log_prior`, `lambda`, `num_switches`, `num_switch_values`, `num_parameters`, `num_iterations`, `num_iterations_vb`, `goals`, `goal_counts`, `bic` (= log-likelihood − params/2·ln N, larger is better; verified), `cs`, `free_energy`, `learn_time`, `learn_search_time`, `em_time` (from `src/prolog/up/util.pl`) |
| `random_set_seed(S)`, `random_select(L,X)`, `random_select(L,Ps,X)`, `random_int`, … | random routines used in the utility part |

## Flags (`set_prism_flag(Name,Value)` / `get_prism_flag`)
`data_source` (`data/1`, `file(F)`, `none`), `log_scale` (on/off, use on for long sequences), `epsilon` (EM convergence threshold), `max_iterate`, `restart`, `init` (`none`/`random`/`noisy_u`), `default_sw` and `default_sw_a`/`_d` (default distribution / pseudo counts), `learn_mode` (`ml`,`vb`,`both`,`ml_vt`,…), `viterbi_mode`, `error_on_cycle`, `clean_table`, `warn`. Defaults and semantics: manual, "Available execution flags". Flags passed to `prism/2` as options override file directives. Note: T-PRISM adds `sgd_*`, `epoch`, `max_iterate` flags too; see the tprism-programming skill.

## Tabling / declarations
`:- p_table p/n.` · `:- p_not_table p/n.` (not together) · `:- include('f.psm').` · `:- set_prism_flag(...).` · `data(File)` (superseded by the `data_source` flag)

## B-Prolog utility constructs you can use in the *utility* part (and carefully in deterministic helpers)
- `maplist(X,Body,Xs)`, `maplist(X,Y,Body,Xs,Ys)`, `maplist(X,Y,Z,Body,Xs,Ys,Zs)` (variables first, then body, then lists); `maplist_func(F,Xs,Ys)`, `maplist_math(Op,Xs,Ys)`, `reducelist(Y0,X,Y1,Body,List,Init,Out)`.
- Loops: `foreach(X in List, Goal)`, `foreach((A,I) in (L1,0..N-1), Goal)`, `Lo..Hi` ranges, list comprehension.
- Lists (verified in Docker: `nth1/3`, `unique/2`, `foreach`, `maplist/5`): `nth0/3`, `nth1/3`, `member/2`, `append/3`, `length/2`, `sublist`, `splitlist`, `countlist`, `filter/3`, `number_sort`, `custom_sort`, `unique`, `findall/3`, `parse_atom/2`, `term2atom/2`.
- Arithmetic: `is`, `=:=`, `<`, `>`; integer division `//`; `mod`.
- Files: `load_clauses(File,Gs)` reads the terms of a file into a list (used to feed `learn`; directives come back as terms too, gotcha G19), CSV utilities (manual "File IO"). `open/3` + `read/2` for custom loaders.

If a predicate you want is not listed, grep the manual and then try it in the REPL before building on it.

Verified to exist and work (PRISM 2.4.2a): `n_viterbi/3`, `get_samples_c/4` (`[inf,N]` form), `nth1/3`, `unique/2`, `save_sw/1`, `restore_sw/1`, `show_sw_a/0`, `print_graph/1`, `probf/2`, `get_values/2`, `viterbig/1`, `log_prob/2`, `learn_b/1`, `learn_statistics/2` (`log_likelihood`, `bic`, `free_energy`), `hindsight/3`, `retractall/1`, `maplist/5`, `foreach/2` with `X in 1..3`, `between/3`, `load_clauses/2` (existence_error if the file is missing), `msw/2` on an undeclared switch (error). Not found in the sources: `viterbi_learn/1`.
