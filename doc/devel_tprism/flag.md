# Developer memo: prism flags — from Prolog to tprism

This memo describes how prism flag values flow from a `.psm` program through
the PRISM engine into the T-PRISM Python package (`tprism` command), and what
to do when adding a new flag.

## Overview of the flow

```
.psm program                     Prolog layer                  C layer                      Python layer
--------------                   -------------------------     -------------------------    --------------------------
set_prism_flag(Name, Value)  ->  $pp_prism_flag/4 check     -> (optional $pc_* callback)
                                 global_set($pg_flag_<Name>)

save_expl_graph / save_flags ->  get_prism_flag/2 (findall) -> $pc_set_export_flags     ->  flags.json
                                 $pp_format_flag_value/2       pc_save_options_3
                                 (float -> atom)               (tensor_preds.cpp)

tprism <mode> --flags ...                                                                   loader.load_explanation_graph
                                                                                            -> expl_pb2.Option (Flag key/value)
                                                                                            -> util.Flags.build(args, options)
                                                                                            -> model/main read typed fields
```

### 1. Flag definition and setting (Prolog)

- All prism flags are declared in `src/prolog/up/flags.pl` as facts:
  `$pp_prism_flag(Name, Type, Init, Pred)`.
  - `Type` is one of `bool`, `enum(Cands)`, `term(Patts)`, `integer(Min,Max)`,
    `float(Min,Max)`, or `special(CheckPred)` for custom validation.
  - `Pred` is an auxiliary predicate (usually a `$pc_*` C callback) called
    with the internal value after a successful `set_prism_flag/2`, or `$none`.
- `set_prism_flag(Name, Value)` validates the value against `Type`
  (via `$pp_check_prism_flag/4`) and stores it in a B-Prolog global variable
  named `$pg_flag_<Name>`. `get_prism_flag(Name, Value)` reads it back.
- Several flags use the atom `default` as an "unset" sentinel
  (e.g. `max_iterate`, `sgd_minibatch_size`, `sgd_loss`, `sgd_patience`,
  `sgd_valid_ratio`); a flag disabled by an exclusive competitor holds
  `$disabled` (e.g. `default_sw_a` vs `default_sw_d`).

### 2. Export to flags.json (Prolog -> C)

- `save_flags/0-3` in `src/prolog/up/tensor.pl` collects **all** flags with
  `findall([X,F1], (get_prism_flag(X,F), $pp_format_flag_value(F,F1)), G)` and
  hands them to the C predicate `$pc_set_export_flags/1`.
  `save_expl_graph/...` calls `save_flags` internally, so a T-PRISM `.psm`
  run (`upprism model.psm <mode>`) always produces the flag list alongside
  the explanation graph.
- `$pp_format_flag_value/2` converts float values to atoms via
  `number_codes/2` before export. This is required because the C-side
  term-to-string conversion (`bpx_term_2_string`, backed by B-Prolog's
  `bp_term_2_string`) prints floats in fixed `%f` notation with 6 digits,
  which truncates small values (`1.0e-8` would become `"0.000000"`).
  Caveat: floats nested inside compound terms (e.g. `sgd_loss = ce(0.1)`
  exported as `"ce(0.100000)"`) still go through the `%f` path, so very small
  loss parameters lose precision.
- `pc_set_export_flags_1` / `pc_save_options_3` in
  `src/c/up/tensor_preds.cpp` serialize the key/value pairs into the
  `Option.flags` field (`message Flag { string key; string value; }`) of
  `src/c/external/expl.proto`, written as `<prefix>.flags.json`
  (or protobuf binary/text). **Every value is a string** at this point;
  atoms that need quoting are exported with surrounding single quotes
  (e.g. `"'0.005'"`, `"'$disabled'"`).

### 3. Import and merge (Python)

- `tprism.loader.load_explanation_graph` parses the JSON into an
  `expl_pb2.Option` and calls `Flags.build(args, options)`
  (`bin/tprism/util.py`).
- `Flags` is a dataclass whose fields are the *typed* counterparts of the
  flags the Python side cares about. `Flags._build` merges values in this
  precedence order (weakest first):

  1. dataclass defaults,
  2. `flags.json` values (i.e. prism flags, including Prolog-side defaults),
  3. explicitly given CLI arguments (only non-`None` argparse values).

  Notes:
  - Keys with no matching dataclass field are silently ignored (flags.json
    contains *all* prism flags; most are irrelevant to the Python side).
  - `Flags.add` normalizes quoted atoms (`'0.005'` -> `0.005`), skips the
    sentinels `default` / `$disabled` (see `PROLOG_UNSET_VALUES`), and casts
    strings to the field type via `cast_value` (`on`/`off`/`true`/`false`
    -> bool, `inf` -> a large int for integer fields, etc.).
  - The Python side does **not** care whether a flags.json value equals the
    Prolog-side default: whatever is in flags.json is applied. In particular,
    with a flags.json present, `sgd_learning_rate` (Prolog default `0.0001`)
    and `sgd_weight_decay` (Prolog default `0.01`, used as weight decay) take
    effect unless overridden on the command line.
  - Renamed flags are handled on both sides: `$pp_prism_flag_renamed/2` in
    `flags.pl` keeps the old name working (with a warning) in Prolog, and
    `RENAMED_FLAGS` in `util.py` maps old keys found in a flags.json exported
    by an older PRISM build (e.g. `sgd_penalty` -> `sgd_weight_decay`). Keep
    the two tables in sync.
- For CLI arguments marked `[prolog flag]` in `main.py`, the argparse default
  must be `None`; a concrete argparse default would always override
  flags.json. Field defaults belong in the `Flags` dataclass instead.
- Consumers: `TprismModel` reads training flags (`max_iterate`,
  `sgd_minibatch_size`, `sgd_patience`, `sgd_valid_ratio`, ...) and
  `TprismModel._build_optimizer` honors `sgd_optimizer` (`adam` / `adadelta` /
  `sgd`), `sgd_learning_rate`, `sgd_weight_decay`, `sgd_adam_*`,
  and `sgd_adadelta_*`. `main._setup` passes `flags.sgd_loss` to
  `LossLoader.get_loss`, which accepts an optional parameter suffix
  (`ce(0.1)` -> loss `ce` with params `["0.1"]`).

## How to add a new flag

Example: adding a flag `sgd_foo` that should be settable both in a `.psm`
program and on the `tprism` command line.

1. **Declare it in `src/prolog/up/flags.pl`** (keep the list alphabetical):

   ```prolog
   $pp_prism_flag(sgd_foo, special($pp_check_sgd_foo), default, $none).
   ```

   - Use the `default` init value + a `special` check with a
     `(default,default,-1)` clause when the flag is consumed only by the
     Python side; this way the Python defaults apply until the user sets it:

     ```prolog
     $pp_check_sgd_foo(default,default,-1).
     $pp_check_sgd_foo(V,V,V) :- number(V), V >= 0.0.
     ```

   - Use a plain type (`bool`, `integer(...)`, `float(...)`, `enum([...])`)
     with a concrete init value when a real Prolog-side default is wanted —
     but remember that this default is then exported to flags.json and will
     override the Python dataclass default.
   - `Pred` is `$none` unless the built-in (C) learners also need the value;
     in that case a `$pc_set_*` C predicate must exist and be registered in
     `src/c/core/glue.c`.

2. **Rebuild the Prolog layer** (the flag table is compiled into the
   bytecode):

   ```sh
   cd src/prolog && make && make install
   ```

   No C/protobuf change is needed for export — `save_flags` exports every
   flag automatically.

3. **Add a typed field to `Flags`** in `bin/tprism/util.py`:

   ```python
   sgd_foo: float = 1.0   # python-side default
   ```

   The field name must equal the flag name. Values arrive as strings and are
   cast by `cast_value`; extend `cast_value` if the flag needs a new
   representation.

4. **(Optional) add a CLI argument** in `bin/tprism/main.py` with
   `default=None`:

   ```python
   parser.add_argument("--sgd_foo", type=float, default=None, help="[prolog flag]")
   ```

5. **Consume it** (e.g. in `tprism.model.TprismModel`) via
   `self.flags.sgd_foo`.

6. **Verify** end to end:
   - `set_prism_flag(sgd_foo, 2.0)` + `save_flags('f.json', json)` in a small
     `.psm` run under `upprism`, then check the value in `f.json`;
   - invalid values raise `domain_error` on `set_prism_flag/2`;
   - `Flags.build` picks the value up and an explicit CLI argument overrides
     it.

## File map

| Role | Location |
|---|---|
| Flag declarations & validation | `src/prolog/up/flags.pl` |
| Export entry (`save_flags`, float formatting) | `src/prolog/up/tensor.pl` |
| C serialization to Option/flags | `src/c/up/tensor_preds.cpp` (`pc_set_export_flags_1`, `pc_save_options_3`) |
| Schema (`Flag` message) | `src/c/external/expl.proto` |
| Python merge & typing (`Flags`) | `bin/tprism/util.py` |
| CLI arguments | `bin/tprism/main.py` |
| Optimizer/training consumers | `bin/tprism/model.py` |
