---
name: programming-prism-ds
description: Data science with PRISM from Python (pyprism). Turn tabular data (pandas DataFrames, CSV, scikit-learn datasets) into discretized PRISM facts with pyprism.dataset. Write generative models for it (independent features, naive Bayes classifiers, latent-class clustering). Learn parameters by EM, MAP or variational Bayes, read them as pandas DataFrames (pyprism.df.sw2df) and plot them. Predict and evaluate on test data (viterbig, prob, hindsight), handle missing values, and compare models by BIC. Use this skill whenever the user analyses data with PRISM or pyprism, follows the PRISM data-science tutorial (PRISM_DS_tutorial, "PRISMでデータ分析", "データサイエンス", "ナイーブベイズ", "離散化"), or mentions load_discrete_diabetes, load_discrete_california_housing, preprocess, sw2df, plot_dist or get_conditional_dist2. Builds on programming-pyprism (installation, PrismEngine, per-query processes) and prism-programming (how values/msw models work).
---

# Data science with PRISM (pyprism)

A PRISM model is a **generative program over discrete values**:

- `values/2` declares the random variables ("switches").
- `msw/2` draws them.
- `learn/1` estimates their probabilities from observed goals.

The workflow of the tutorial:

```
DataFrame --pyprism.dataset--> train.dat / test.dat      obs(Y,[X0,...,Xn]).   (discretized ints)
          --PRISM program + learn--> save_sw('nb.sw')
          --pyprism.df.sw2df--> parameters as a DataFrame, plots
          --restore_sw + viterbig / prob / hindsight--> predictions --> metrics in Python
```

**Setup.** Install pyprism and a PRISM binary as in programming-pyprism. Then:

```sh
pip install pandas "scikit-learn>=1.6,<1.9" matplotlib
```

- scikit-learn >= 1.6 is required by `pyprism.dataset`.
- The upper bound only matters for reproducing the tutorial numbers. 1.9 changed the quantile discretization, so 26 of the 442 diabetes rows get other bins.

**Every `engine.query` is a new PRISM process.** Learn and `save_sw` in one query. Start every later query with `restore_sw(...)`.

## 1. Prepare the data

```python
from pyprism.dataset import load_discrete_diabetes, load_discrete_california_housing, preprocess
ds = load_discrete_diabetes(out_filename="train.dat", out_test_filename="test.dat",
                            pred="obs", test_ratio=0.2)          # 353 / 89 rows
# your own data:
ds = preprocess(df_x, df_y, out_filename="train.dat", out_test_filename="test.dat",
                pred="obs", test_ratio=0.2)
```

Each row becomes a fact `obs(Y,[X0,...,Xn]).`, or `obs([X0,...,Xn]).` with `with_y=False`.

| Argument | Default | Meaning |
|---|---|---|
| `pred` | `"data"` | The predicate name written to the file. **Pass `pred="obs"`** if your program uses `obs`. The tutorial's final Pima exercise omits it and gets `data(...)` facts. |
| `disc_bins_x`, `disc_bins_y` | 5, 8 | The number of quantile bins (`KBinsDiscretizer`, ordinal). |
| `thresh_uniq_x`, `thresh_uniq_y` | 10 | Columns with fewer unique values are **kept as they are**. For example, diabetes `sex` stays 1/2, and an iris target stays 0/1/2. |
| `test_ratio` | 0.0 | Train/test split with a fixed `random_state=42`. The bins are fitted on the training part. |
| `missing_px`, `missing_py` | 0.0 | Random missing values, written as `_` (see section 3). |

The returned dict has these keys:
- `X`, `y`
- `X_discretized`, `y_discretized`
- `X_test_discretized`, `y_test_discretized`
- `X_discretizers`, `y_discretizer`: fitted bins, usable as plot labels
- `attr_list`: column names in the list order

`plot_discretization(ds["X"], disc_bins=5)` and
`plot_discretized_data(ds["X_discretized"])` show histograms before and after
discretization.

**Check the value ranges** (`ds["X_discretized"].max()`,
`ds["y_discretized"].unique()`). The ranges in `values/2` must contain every
value that occurs.

## 2. Write the model

Pass the model to `engine.set_db(r"""...""")`. The `@=` list comprehension is
built in, e.g. `N @= [X : X in 0..7]` gives `N = [0,1,...,7]`.

**Independent features.** One distribution per column:

```prolog
values(dice(_),[0,1,2,3,4]).
obs(L) :- obs(L,0).
obs([],_).
obs([X|R],N) :- msw(dice(N),X), N1 is N+1, obs(R,N1).
```

**Naive Bayes classifier.** Draw the target, then each feature given the target.
`out(T,N)` is the distribution of column N for class T:

```prolog
values(target,N)   :- N @= [X : X in 0..7].     % = disc_bins_y classes
values(out(_,_),N) :- N @= [X : X in 0..4].     % = disc_bins_x bins
obs(T,L) :- msw(target,T), nb(0,T,L).
nb(_,_,[]).
nb(N,T,[X|L]) :- msw(out(T,N),X), N1 is N+1, nb(N1,T,L).
```

**Latent-class clustering (unsupervised).** Use the same structure, but the
class is hidden. `clus(C,L)` exposes the cluster for the assignment afterwards:

```prolog
values(cluster,Cs) :- num_clusters(K), Cs @= [C : C in 1..K].
values(out(_,_),[0,1,2,3,4]).
num_clusters(3).
obs(L) :- clus(_,L).
clus(C,L) :- msw(cluster,C), nb(0,C,L).
nb(_,_,[]).
nb(N,C,[X|L]) :- msw(out(C,N),X), N1 is N+1, nb(N1,C,L).
```

The same pattern extends to HMMs, Bayesian networks and mixtures (see
prism-programming). The data only has to be goals that the program can
generate.

## 3. Learn

```python
lines, status = engine.query("""
  load_clauses('train.dat',Gs), learn(Gs), save_sw('nb.sw'),
  learn_statistics(log_likelihood,LL), learn_statistics(bic,BIC)""", out=["LL", "BIC"])
lines[-1]        # 'LL=-5550.71,BIC=-6509.88'  (learning statistics are in the earlier lines)
```

**Smoothing.** Maximum likelihood (the default) gives probability 0 to
combinations unseen in training: 39 of the 408 naive-Bayes parameters for
diabetes. Then `prob` of a test row can be 0. For MAP estimates with a pseudo
count, put this in front of `learn`:

```
set_prism_flag(default_sw_d,1.0), load_clauses(...), learn(Gs), ...
```

This gives no zero parameters.

**Variational Bayes.** Use `learn_b(Gs)`, and `learn_statistics(free_energy,F)` for the objective.

**Latent models.** Run several random restarts with `set_prism_flag(restart,5)`, and keep the best result.

**Missing values.** A `_` in a goal, e.g. `obs(4,[3,2,4,3,0,_,_,3,3,1])`, is an unobserved variable. EM sums over its values, so no imputation is needed.

**Model selection.** PRISM's `bic` is `log-likelihood − (#params/2)·ln N`, so **larger is better**. For example, latent-class models on diabetes (all 442 rows, 5 restarts, scikit-learn 1.8):

| clusters | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| BIC | −6688 | −6319 | −6303 | −6297 |

Other statistics: `num_parameters`, `num_iterations`, `learn_time`.

## 4. Look at the parameters in Python

```python
from pyprism.df import sw2df, plot_dist, get_conditional_dist2, disc2binlist
df = sw2df("nb.sw")   # columns: Name Arity Term Status Vals Param Arg1..Arg5 (one row per switch)
df[df["Name"] == "target"]          # P(target)
df[df["Name"] == "out"]             # P(X_n | target), Arg1 = class, Arg2 = column index

attr_y = disc2binlist(ds["y_discretizer"])                       # bin labels like "25.0-60.0"
attr_x = {k: disc2binlist(d) for k, d in ds["X_discretizers"].items()}
plot_dist(df[df["Name"] == "target"], attr_y, title="target")    # bar chart
get_conditional_dist2(df[df["Name"] == "out"], arg_var="Arg2", arg_cond="Arg1",
                      attr_var=ds["attr_list"], attr_cond=attr_y, attr_val=attr_x,
                      cond_var_name="y")                         # one heatmap per feature
```

- `Vals` and `Param` are Python lists.
- In `Arg*`, the switch arguments are strings.
- `disc2binlist` needs a fitted discretizer. Columns kept as they are have none: they are missing from `X_discretizers`, and `y_discretizer` is `None`. Check for `None`, as the tutorial does.

## 5. Predict and evaluate

**The whole test set in one query.** `viterbig(obs(Y,L))` binds `Y` to the most
probable class:

```python
lines, status = engine.query("""restore_sw('nb.sw'), load_clauses('test.dat',Gs),
  findall(T-Y, (member(obs(T,L),Gs), viterbig(obs(Y,L))), R)""", out="R")
pairs = [p.split("-") for p in lines[0][len("R=["):-1].split(",")]      # [['4','3'], ...]
acc = sum(t == y for t, y in pairs) / len(pairs)
```

**Always compare with a baseline.** For diabetes discretized into 8 classes,
naive Bayes gets 0.225, and predicting the majority class gets 0.213. The
model is weak there, and the pipeline itself is fine.

**The distribution over classes for one row.**
- `hindsight(obs(_,L),obs(_,_),HP)` returns `HP=[[obs(0,L),P0],[obs(1,L),P1],...]`. These are the joint probabilities; divide by their sum to get the posterior.
- `hindsight(G,G)`, as in the tutorial, only prints them.
- `member(Y,[0,...,7]), prob(obs(Y,L),P)` with `findall=True` gives the same numbers.

**Precision.**
- `out=` prints about 6 significant digits (`P=6.87932e-08`).
- Use `format("P=~15e~n",[P])` for full precision.
- Use `log_prob(G,LP)` when products of many small probabilities would underflow.

## 6. Gotchas

- **Parameters live in files.** Write `restore_sw('nb.sw')` at the start of every prediction query. Relative paths (`train.dat`, `nb.sw`) are resolved against the Python process's current directory.
- **The outputs are strings.** Parse them in Python, as with `pairs` above.
- **Values outside `values/2`.** The whole `learn` aborts with `error(prism_runtime_error(explanation_not_found),...)`, because that goal has no explanation. Check the ranges after discretization (section 1).
- **Big data.** Learning is fast; about 16k California-housing rows learn in under a second. However, one `findall` over all test rows prints one long line. Aggregate in Prolog when only counts are needed:
  ```
  findall(C, (member(obs(T,L),Gs), viterbig(obs(Y,L)), (T==Y -> C=1 ; C=0)), Cs), sumlist(Cs,Ok), length(Cs,N)
  ```
- **Plots in scripts.** `pyprism.df` draws with matplotlib and calls `plt.show()`. Use `matplotlib.use("Agg")` in scripts without a display.
