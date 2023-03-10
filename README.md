![](https://github.com/prismplp/prism/actions/workflows/release.yml/badge.svg)
[![](https://dockerbuildbadges.quelltext.eu/status.svg?organization=prismplp&repository=prism)](https://hub.docker.com/r/prismplp/prism/builds/ 'DockerHub')
[![](https://img.shields.io/docker/stars/prismplp/prism.svg)](https://hub.docker.com/r/prismplp/prism 'DockerHub')
[![](https://img.shields.io/docker/pulls/prismplp/prism.svg)](https://hub.docker.com/r/prismplp/prism 'DockerHub')
# PRISM 

This is a software package of PRISM version 2.3, a logic-based
programming system for statistical modeling, which is built
on top of B-Prolog (http://www.probp.com/).  Since version 2.0,
the source code of the PRISM part is included in the released
package.  Please use PRISM based on the agreement described in
LICENSE and LICENSE.src.

- LICENSE     ... license agreement of PRISM
- LICENSE.src ... additional license agreement on the source code of PRISM
- bin/        ... executables
- doc/        ... documents
- src/        ... source code
- exs/        ... example programs

For the files under each directory, please read the README file
in the directory.  For the papers or additional information
on PRISM, please visit http://rjida.meijo-u.ac.jp/prism/ .

## Installation

Pleaes download pre-build package from [release page](https://github.com/prismplp/prism/releases):
```
wget "https://github.com/prismplp/prism/releases/download/v2.4.2a(T-PRISM)-prerelease/prism_tprism_pre_linux_ubuntu20.tar.gz"

```

Extract binaries and sample programs
```
tar xvf prism_tprism_pre_linux_ubuntu20.tar.gz 
```


Setting the proper environmental variable: 
```
export PATH=<current directory>/prism/bin:${PATH}
```

Try!
```
$ prism
```
Press Ctrl+D to quit the interactive mode.


# PyPRISM: Python interface
PyPRISM is a Python interface to PRISM.

Please see: https://github.com/prismplp/pyprism

# T-PRISM: Tensorized-PRISM  (Pre-release)
T-PRISM is a new logic programming language based on tensor embeddings.
Our embedding scheme, named tensorized semantics, is a modification of the distribution semantics in PRISM, one of the state-of-the-art probabilistic logic programming languages, by replacing distribution functions with multidimensional arrays.

Tutorial: https://colab.research.google.com/drive/16yzyaglTq0nTvgzZS_nJHEPleddYgxfB?usp=sharing

API Documents: https://prismplp.github.io/prism/tprism/tprism.html

```
@article{kojima2019tensorized,
  title={A tensorized logic programming language for large-scale data},
  author={Kojima, Ryosuke and Sato, Taisuke},
  journal={arXiv preprint arXiv:1901.08548},
  year={2019}
}
```
## Installation

Requirements: PRISM, python(Recommendation: Anaconda) and Pytorch.
- Anaconda: https://www.anaconda.com/
- Pytorch: https://pytorch.org/

Please Install T-PRISM by the following command:
```
pip install "git+https://github.com/prismplp/prism.git#egg=t-prism&subdirectory=bin"
```

Please see the details in [T-PRISM manual](https://github.com/prismplp/prism/releases/download/v2.4(T-PRISM)-prerelease/tprism_manual.pdf).


