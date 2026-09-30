# Driving a PRISM program from scripts, other languages and pipelines

Use this when a `.psm` is one stage of something bigger: a shell script, a CI
job, a Python or other subprocess call, or a config-driven benchmark. From
Python, pyprism (the programming-pyprism skill) is the ready-made wrapper,
and the same rules apply to it.

Contents: 1 entry point with modes · 2 output format · 3 exit status and error detection ·
4 state between runs · 5 pipeline integration · 6 interactive exploration

---

## 1. One `prism_main/1` with modes

`upprism file.psm a b c` calls `prism_main([a,b,c])` with the arguments as
**atoms**; convert numbers with `parse_atom/2`. Dispatch on the first element
and end with a usage clause. This is from `../examples/activity.psm`, verified:

```prolog
prism_main([learn, ParamFile, DataFile, GoalsFile]) :- !,        % build parameters
    load_data(DataFile), load_clauses(GoalsFile, Gs), learn_and_save(Gs, ParamFile).
prism_main([infer, ParamFile, DataFile, Key]) :- !,              % use learned parameters
    load_data(DataFile), restore_sw(ParamFile), print_dist(Key).
prism_main([DataFile, Key]) :- !,                                % defaults, no learning
    load_data(DataFile), set_params, print_dist(Key).
prism_main(_) :-
    format("usage: upprism activity.psm [learn <params> <data> <goals> | infer <params> <data> <key> | <data> <key>]~n").
```

```sh
upprism activity.psm data.pl d1
upprism activity.psm learn params.sw data.pl goals.pl
upprism activity.psm infer params.sw data.pl d2
```

Notes:
- More data files per mode are easy: take a list tail (`[learn, ParamFile | DataFiles]`, with `DataFiles = [_|_]`) and load each one.
- If both `prism_main/0` and `prism_main/1` exist, only `/1` runs; with no arguments it gets `[]`.
- Call `random_set_seed/1` first when sampling must be reproducible.

## 2. Output format

- **Print one self-describing line per result**, with a tag, e.g. `DIST:d1:picnic:0.6000` or `LEARN:goals:8:skipped:1`. The caller parses the lines by prefix, and a human can still read them.
- **Never print raw Prolog terms for a program to parse.** If you need structure, print one JSON object per line.
- **The tag vocabulary is an interface.** Keep it stable and documented; changing it breaks every caller.
- **Control the float formatting yourself** (`format("~4f",[P])`, or `~15e` for full precision). `~w` prints about 6 significant digits.
- **PRISM's own messages go to the same stdout.** This includes the banner, the learning statistics and `show_sw`, so filter by your tags.

## 3. Exit status and error detection

This is verified with `upprism` 2.4.2a:

| What happened | Exit status | Last line |
|---|---|---|
| `prism_main` succeeded | 0 | `yes` |
| `prism_main` **failed** | **0** | `no` |
| Uncaught exception (e.g. `instantiation_error`, `learn` on a bad goal) | 1 | `Aborted by exception -- error(...)` |
| The file contains `\+`/`not` of a probabilistic goal (gotcha G7) | **0** | only the banner; `prism_main` never ran |

**So the exit status is not enough.** Make the caller require its expected
tags. A simple way is to print a final `DONE:...` line and treat its absence
as a failure. Catch the errors you can handle inside the program, e.g. with
`safe_prob/2` (gotcha G18), and print them as tagged lines.

## 4. State between runs

Every `upprism` run is a fresh process:
- the parameters set by `set_sw` or learned by `learn`, the `assert`ed data and the flags are all gone;
- persist the parameters with `save_sw(File)`, and start every inference run with `restore_sw(File)`, or with the default seeding (`set_params`);
- when parameters belong to a data split (a cross-validation fold), put the split identifier in the file name (`params.fold3.sw`), and have the caller pass it through rather than infer it;
- when evaluating against held-out labels that are also in the data, mask them before querying: `retractall(known_label(Id, _))`. `retractall/1` is verified.

## 5. Pipeline integration

- **Make the executable, the `.psm`, the parameter files and the mode configuration**, not hard-coded paths, so that the same program can be run on other data or with other learned parameters without editing Prolog.
- **Relative paths** in `open/3`, `load_clauses/2` and `save_sw/1` are resolved against the working directory of the `upprism` process.
- **Data from other tools**, e.g. facts written for SWI-Prolog, need the directive-skipping loader of gotcha G19. Declare the data predicates `:- dynamic` in the `.psm`, so that a missing file means "no facts" instead of an existence error.
- **The `.psm` directory must be writable.** `upprism` writes the compiled program as `file.psm.out` next to the source, and loads that. In a read-only directory it aborts with `permission_error(open,source_sink,....psm.out)` (verified). Pass large data as files, not as command-line arguments.

## 6. Interactive exploration (for debugging, not for batch runs)

```prolog
?- prism(activity).
?- load_data('data.pl'), set_params.
?- prob(plan(d1,picnic), P).
?- probf(plan(d1,picnic)).                 % the explanation graph
?- viterbif(plan(d1,picnic)).              % print the most likely explanation (Viterbi_P = 0.6)
?- viterbig(plan(d1,A)).                   % bind A to the most likely value: A = picnic
                                           % (viterbif on a non-ground goal prints but leaves A unbound)
?- restore_sw('params.sw'), show_sw.       % learned instead of seeded parameters
```
