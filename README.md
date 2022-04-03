[![CircleCI](https://circleci.com/gh/prismplp/prism.svg?style=shield)](https://circleci.com/gh/prismplp/prism)
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
# Python interface
Please see: https://github.com/prismplp/pyprism

# Pre-release Tensorized-PRISM (T-PRISM)
## Installation

Preparation for Ubuntu18.04
```
sudo apt-get install libhdf5-100 libhdf5-cpp-100
```
Preparation for Ubuntu20.04

```
sudo apt-get install libhdf5-103 libhdf5-cpp-103
```



Installing python(Recommendation: Anaconda) and Pytorch.
- Anaconda: https://www.anaconda.com/
- Pytorch: https://pytorch.org/

For Ubuntu18.04, download `prism_tprism_pre_linux_ubuntu18.tar.gz` from [release page](https://github.com/prismplp/prism/releases):
```
wget "https://github.com/prismplp/prism/releases/download/v2.4.2a(T-PRISM)-prerelease/prism_tprism_pre_linux_ubuntu18.tar.gz"
```

For Ubuntu20.04, download `prism_tprism_pre_linux_ubuntu20.tar.gz` from [release page](https://github.com/prismplp/prism/releases) 
```
wget "https://github.com/prismplp/prism/releases/download/v2.4.2a(T-PRISM)-prerelease/prism_tprism_pre_linux_ubuntu20.tar.gz"

```


Extract binaries and sample programs
```
tar xvf prism_tprism_pre_linux_ubuntu18.tar.gz 
```
or
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

Please see the details in [T-PRISM manual](https://github.com/prismplp/prism/releases/download/v2.4(T-PRISM)-prerelease/tprism_manual.pdf).


