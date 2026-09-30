---
name: programming-pyprism
description: Install PyPRISM (the pyprism package, the Python interface of PRISM, prismplp/pyprism) so that it matches the Python environment (Google Colab, venv or conda on Linux, Jupyter), and use its PrismEngine to run Prolog and PRISM programs and queries from Python (set_db, query with out/findall/find_n, prob, sample, learn, save_sw/restore_sw, parsing the results). Use this skill whenever the user mentions pyprism, PyPRISM, PrismEngine, engine.query or engine.set_db, runs PRISM or Prolog from Python or a notebook, follows the Prolog or PRISM tutorials on Colab ("PythonからPRISM", "PythonからProlog", "pyprismのインストール"), or gets a pyprism result like (None, 'error'). For writing the probabilistic models themselves also use prism-programming, and for data-science workflows (pyprism.dataset, sw2df, prediction) use programming-prism-ds.
---

# Using PRISM from Python with PyPRISM

`pyprism` (https://github.com/prismplp/pyprism) runs PRISM as a subprocess. Every
`engine.query(...)` does the following:

1. Writes the program set by `set_db` plus `prism_main :- <query>.` to
   `./.prism_code/<YYYYmmdd-HHMMSS>.psm`.
2. Runs `<bin_path>/upprism` on that file.
3. Parses the standard output.

**Each query is therefore a fresh PRISM process.** Nothing survives between
queries: learned parameters, `assert`ed facts and the random state are all
lost. This one fact explains most surprises; see the gotchas in section 5.

## 1. Install to match the environment

The package's code is Python, but it bundles a Linux x86-64 PRISM binary in
`site-packages/pyprism/bin`. **The Python part installs anywhere. The question
is only which PRISM binary it runs**, and that is selected by
`PrismEngine(bin_path=...)`.

| PRISM binary | How to get it | Runs on | Notes |
|---|---|---|---|
| Bundled (`PrismEngine()` with no `bin_path`) | comes with `pip install` | Linux x86-64 with glibc >= 2.34 (Ubuntu 22.04, 24.04, 26.04) | an older PRISM 2.4.2a build (2023) |
| Colab package `prism_linux_dev4colab.auto.zip` | release `v2.4.2a(T-PRISM)-prerelease` | glibc >= 2.38 (Colab, Ubuntu 24.04 and later) | built by CI from master |
| `prism_tprism_pre_linux_ubuntu*.tar.gz` binary packages | made with `tools/` (install-linux skill, section 7) | the Ubuntu they were built for | need the HDF5 runtime packages |
| Your own build | install-linux skill | the machine it was built on | use this when you change PRISM itself |

### Google Colab

The tutorials use these cells:

```python
!wget -q "https://github.com/prismplp/prism/releases/download/v2.4.2a(T-PRISM)-prerelease/prism_linux_dev4colab.auto.zip"
!unzip -q -o prism_linux_dev4colab.auto.zip          # -> ./prism/bin
!pip install -I "git+https://github.com/prismplp/pyprism.git"

from pyprism import PrismEngine
engine = PrismEngine(bin_path="prism/bin")
```

### Linux: venv or conda

Ubuntu 24.04 and later refuse pip installs into the system Python, so use a
venv or conda:

```sh
python3 -m venv ~/venv/pyprism && . ~/venv/pyprism/bin/activate     # or: conda activate <env>
pip install "git+https://github.com/prismplp/pyprism.git"
pip install pandas "scikit-learn>=1.6" matplotlib   # needed by pyprism.dataset / pyprism.df (not declared by the package)
```

- `import pyprism` itself needs nothing else.
- `pyprism.dataset` needs scikit-learn >= 1.6. With older versions, `load_discrete_*` fail with `module 'sklearn' has no attribute 'datasets'`.
- To reproduce the numbers of the Colab tutorials exactly, use scikit-learn < 1.9. Version 1.9 changed the quantile discretization; see programming-prism-ds.

Then choose a binary:
- `PrismEngine()` uses the bundled one.
- `PrismEngine(bin_path="/path/to/prism/bin")` uses a release package or your own build.

`bin_path` must be the directory that contains `upprism`. A relative path is
resolved against the Python process's current directory.

### Check the installation

```python
from pyprism import PrismEngine
engine = PrismEngine(bin_path="prism/bin")          # or PrismEngine()
print(engine.query(r'format("hello world~n")'))     # (['hello world'], 'yes')
```

If this returns `(None, 'error')`, the binary did not start. The reason was
printed on stderr, and it is also kept in `engine.result_stderr`. A typical
case is a Colab or Ubuntu 24.04 package on Ubuntu 22.04:

```
... version `GLIBC_2.38' not found
```

In that case use the bundled binary, a package built for that Ubuntu, or your
own build.

### Jupyter

Use a normal Python kernel and `PrismEngine`. The "PRISM" Jupyter kernel in the
pyprism repository (`pyprism_kernel/kernel.py`) does not start with current
Jupyter: it imports `IPython.kernel`, which was removed from IPython long ago
(`ModuleNotFoundError: No module named 'IPython.kernel'`).

## 2. API

```python
engine = PrismEngine(bin_path=None, wd_path="./.prism_code/")
engine.set_db(program_text)          # facts, rules, values/2 ... (replaces the previous program)
lines, status = engine.query(q, out=None, findall=False, find_n=None,
                             verbose=False, err_verbose=True, args=[])
```

- `q` is the body of a query, without `?-`. A trailing `.` or `,` is stripped.
- `status` is `'yes'` or `'no'` when the program ran. It is the line `Aborted by exception -- ...` when the query raised an error, for example a syntax error. It is `'error'` (with `lines=None`) when PRISM printed nothing usable, for example because the binary did not start or the query called an undefined predicate.
- `lines` are the output lines of the query. That includes anything the program prints (`format/2`, `show_sw`, learning statistics), so the variable bindings are not always the only lines.
- `out="X"` or `out=["X", "Y"]` prints the bindings as `X=...,Y=...` lines:
  - by default, only the first solution;
  - `findall=True`, every solution;
  - `find_n=N`, the first N solutions.
- `pyprism.parse_output('X=pam,Y=bob')` returns `[('X', 'pam'), ('Y', 'bob')]`. The values are strings; convert numbers yourself.
- `verbose=True` prints the generated query and PRISM's output. `err_verbose=False` silences stderr.
- `engine.run(code)` runs a whole program (with your own `prism_main/0,1`) and returns the raw stdout lines. `args=[...]` are passed to `upprism`, which gives them to `prism_main/1`.

## 3. Patterns

Prolog (from the Prolog tutorial):

```python
engine.set_db(r"""
parent(pam,bob). parent(tom,bob). parent(tom,liz). parent(bob,ann). parent(bob,pat). parent(pat,jim).
female(liz). female(ann). female(pat).
grandparent(X,Z) :- parent(X,Y), parent(Y,Z).
predecessor(X,Z) :- parent(X,Z).
predecessor(X,Z) :- parent(X,Y), predecessor(Y,Z).
sister(X,Y) :- parent(Z,X), parent(Z,Y), female(X), X \= Y.
""")
engine.query("parent(tom,liz)")                                  # ([], 'yes')
engine.query("parent(X,liz)", out="X")                           # (['X=tom'], 'yes')
engine.query("predecessor(pam,X)", out="X", findall=True)        # (['X=bob','X=ann','X=pat','X=jim'], 'yes')
engine.query("sister(X,pat)", out="X", findall=True)             # (['X=ann'], 'yes')
engine.query("X is 1+2*3", out="X")                              # (['X=7'], 'yes')   ( = does not evaluate)
```

PRISM (from the PRISM tutorial):

```python
engine.set_db(r"""
values(dice,[s1,s2,s3,s4,s5,s6],[0.05,0.05,0.1,0.15,0.25,0.4]).
go(X) :- msw(dice,X).
""")
engine.query("prob(go(s6),P)", out="P")                          # (['P=0.4'], 'yes')
engine.query("get_samples(10,go(_),Gs)", out="Gs")               # 10 samples in ONE process

engine.set_db("values(dice,[s1,s2,s3,s4,s5,s6]).\nobs(X) :- msw(dice,X).\n")
engine.query("learn([obs(s1),obs(s4),obs(s5),obs(s6),obs(s6)]), save_sw('dice.sw')")
engine.query("restore_sw('dice.sw'), prob(obs(s6),P)", out="P")  # (['P=0.4'], 'yes')
```

Other built-ins used the same way:
- `learn_b/1` (variational Bayes)
- `show_sw` (prints the parameters into `lines`)
- `hindsight/1,2,3`
- `viterbif/1`, `viterbig/1,2`
- `log_prob/2`
- `load_clauses('data.dat', Gs)` (reads facts from a file)

For how to write PRISM models (`values`/`msw`, the generative reading), read the
prism-programming skill. For tabular data, discretization, prediction and plots,
read programming-prism-ds.

## 4. Verifying a change

Write the check as a short Python script and run it where the user's Python
and PRISM live. Compare `status`, not only `lines`: a failed query often
still returns lines. When the environment is in doubt, reproduce it in a clean
container (`docker run --rm -it ubuntu:24.04`, then the venv steps of
section 1).

## 5. Gotchas

| Symptom | Cause and fix |
|---|---|
| Learned parameters are gone in the next query (probabilities back to uniform) | Each query is a new process. Save with `save_sw('f.sw')` and start the next query with `restore_sw('f.sw')`, or learn and use the parameters in one query. The same holds for `assert` and `set_prism_flag`. |
| Repeated `engine.query("go(X)", out="X")` returns the same sample | The seed is the current time in seconds, so queries within the same second repeat. Draw all samples in one query (`get_samples(10,go(_),Gs)`, or `findall(X,(between(1,10,_),sample(go(X))),Xs)`), or pass a seed: `random_set_seed(123), ...`. |
| `engine.query(...)` never returns, and memory keeps growing | A non-terminating query, e.g. printing the cyclic term of `X=f(X)`, or left recursion. `query` has no timeout. Use the subclass below the table, and interrupt the kernel if it is already stuck. |
| `IndexError: string index out of range` | The query string is empty. |
| `(None, 'error')` | The binary did not start (`engine.result_stderr`, section 1), or the query used an undefined predicate. |
| `status` is `Aborted by exception -- error(...)` | The program or query raised an error, e.g. a syntax error. `lines` contains the message and the line number in the generated `.psm`. |
| `SyntaxWarning: invalid escape sequence '\='` (Python >= 3.12) | Use raw strings (`r"""..."""`) for Prolog code with `\=`, `\+` or `\n`. Inside Prolog, write the newline as `~n` in `format/2`. |
| Numbers have only about 6 significant digits (`P=4.94647e-09`) | `out=` prints with `~w`. Print more digits yourself, e.g. `format("~15e~n",[P])`, or compare `log_prob/2` values. |
| Files keep piling up in `./.prism_code/` | Every query writes a `.psm` (and its compiled `.psm.out`). Delete the directory when you like. |
| Parallel queries interfere | Files are named by the current second. Give each worker its own `wd_path`. |
| `load_clauses('x.dat',Gs)` cannot find the file | Paths are relative to the Python process's current directory, not to `bin_path`. |

A timeout for queries. `upprism` execs the PRISM binary, so the timeout kills
PRISM itself, and `subprocess.TimeoutExpired` is raised:

```python
import subprocess
from pyprism import PrismEngine

class TimeoutPrismEngine(PrismEngine):
    def __init__(self, *args, timeout=60, **kwargs):
        super().__init__(*args, **kwargs)
        self.timeout = timeout
    def run_file_(self, filename, args=[]):
        return subprocess.run([self.bin_path + "/upprism", filename] + args,
                              timeout=self.timeout, capture_output=True)

engine = TimeoutPrismEngine(bin_path="prism/bin", timeout=30)
```
