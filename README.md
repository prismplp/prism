![](https://github.com/prismplp/prism/actions/workflows/release.yml/badge.svg)
[![](https://dockerbuildbadges.quelltext.eu/status.svg?organization=prismplp&repository=prism)](https://hub.docker.com/r/prismplp/prism/builds/ 'DockerHub')
[![](https://img.shields.io/docker/stars/prismplp/prism.svg)](https://hub.docker.com/r/prismplp/prism 'DockerHub')
[![](https://img.shields.io/docker/pulls/prismplp/prism.svg)](https://hub.docker.com/r/prismplp/prism 'DockerHub')
# PRISM: PRogramming In Statistical Modeling
PRISM is a general programming language intended for symbolic-statistical modeling. It is a new and unprecedented programming language with learning ability for statistical parameters embedded in programs.
Its programming system is a powerful tool for building complex statistical models. 

For the papers or additional information
on PRISM, please visit http://rjida.meijo-u.ac.jp/prism/ .


## Tutorials

[PRISM manual](https://github.com/prismplp/prism/releases/download/v2.4.1(T-PRISM)-prerelease/manual.pdf)

Prolog tutorial (Japanese): [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1EhnP2ApqsuchEY-k9ZFUzBZg8Enjyytz?usp=sharing)

PRISM tutorial (Japanese):  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/182ujzp3Z1jfwDTnd61QnrPDdwoT5CSq7)


## Installation

#### 1. Download pre-build package from [release page](https://github.com/prismplp/prism/releases):

If you want to install the latest development version package pre-built with the latest version of ubuntu (the latest version from github is automatically built), 
you can install it with the following command.
```
wget "https://github.com/prismplp/prism/releases/download/v2.4.2a(T-PRISM)-prerelease/prism_linux_dev.auto.tar.gz"

```

#### 2. Extract binaries and sample programs.

If you downloaded a different release version, please change the file name and unzip it in the same way.
```
tar xvf prism_linux_dev.auto.tar.gz
```


#### 3. Setting the proper environmental variable: 
```
export PATH=<current directory>/prism/bin:${PATH}
```

#### 4. Try!
```
$ prism
```
Press Ctrl+D to quit the interactive mode.

## Contents of the Package
This is a software package of PRISM version, a logic-based
programming system for statistical modeling, which is built
on top of B-Prolog (http://www.probp.com/). 
Since version 2.0,
the source code of the PRISM part is included in the released
package.
Please use PRISM based on the agreement described in
LICENSE and LICENSE.src.

- LICENSE     ... license agreement of PRISM
- LICENSE.src ... additional license agreement on the source code of PRISM
- bin/        ... executables
- doc/        ... documents
- src/        ... source code
- exs/        ... example programs

For the files under each directory, please read the README file
in the directory.  


# PyPRISM: Python interface to PRISM
Please see: https://github.com/prismplp/pyprism

# T-PRISM: Tensorized-PRISM  (Pre-release)
[![arXiv](https://img.shields.io/badge/arXiv-1901.08548-b31b1b.svg)](https://arxiv.org/abs/1901.08548)

T-PRISM is a new logic programming language based on tensor embeddings.
Our embedding scheme, named tensorized semantics, is a modification of the distribution semantics in PRISM, one of the state-of-the-art probabilistic logic programming languages, by replacing distribution functions with multidimensional arrays.

T-PRISM tutorial:　[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/16yzyaglTq0nTvgzZS_nJHEPleddYgxfB?usp=sharing)

API Documents: https://prismplp.github.io/prism/tprism/tprism.html

```
@article{kojima2019tensorized,
  title={A tensorized logic programming language for large-scale data},
  author={Kojima, Ryosuke and Sato, Taisuke},
  journal={arXiv preprint arXiv:1901.08548},
  year={2019}
}
```
### T-PRISM Requirements

- PRISM (see [Installation](#installation) above): used to run T-PRISM programs (`.psm`) and export explanation graphs
- Python >= 3.10 (Recommendation: [Anaconda](https://www.anaconda.com/))
- [PyTorch](https://pytorch.org/)
- NumPy
- h5py
- scikit-learn
- protobuf >= 4.21.12 (>= 5.27.0 on Python 3.14 or later)

Optional:
- networkx, pyvis, and IPython: visualization of explanation graphs (`tprism.plot`)
- [geotorch](https://github.com/Lezcano/geotorch): constrained tensors (symmetric, orthogonal, low-rank, positive definite, etc.)

As a python, [conda](https://www.anaconda.com/docs/getting-started/miniconda/install/linux-install) is recomended.
The `pip install` command below does not install these packages.
Please install PyTorch following https://pytorch.org/, and then the others, e.g.:
```
pip install numpy h5py scikit-learn "protobuf>=4.21.12"
```

### T-PRISM Installation

Please Install T-PRISM by the following command:
```
pip install "git+https://github.com/prismplp/prism.git#egg=t-prism&subdirectory=bin"
```

Please see the details in [T-PRISM manual](https://github.com/prismplp/prism/releases/download/v2.4(T-PRISM)-prerelease/tprism_manual.pdf).

# AI Agent skills

[plugins/prism/](plugins/prism/) contains agent skills for building PRISM/T-PRISM and for writing PRISM, PyPRISM and T-PRISM programs.
These features are experimental and subject to change in future updates to the agent framework.

Claude Code:
```
claude plugin marketplace add prismplp/prism
claude plugin install prism@prismplp
```
(or `/plugin marketplace add prismplp/prism` and `/plugin install prism@prismplp` in a session).
To update: `claude plugin marketplace update prismplp` and `claude plugin update prism@prismplp`.

Codex:
```
codex plugin marketplace add prismplp/prism
codex plugin add prism@prismplp
```
To update: `codex plugin marketplace upgrade prismplp` and run `codex plugin add prism@prismplp` again.

To use a local clone instead, pass its path in place of `prismplp/prism`.

Antigravity has no marketplace for third-party plugins, so install the plugin from a clone:
```
git clone https://github.com/prismplp/prism.git
agy plugin install prism/plugins/prism
```
To update: `git -C prism pull` and run `agy plugin install prism/plugins/prism` again.
Alternatively, add `{"entries": [{"path": "<path to the clone>/plugins"}]}` to `~/.gemini/config/plugins.json`, so that Antigravity reads the plugin from the clone and `git pull` updates it.


