# What is T-PRISM?
T-PRISM is a Prolog-based modelling language using tensor, which allows modelling neural networks, matrix computation, and other tensor models.
T-PRISM's main feature is a focus on supporting many types of models with the simple rules:
 - Multi-layer perceptron
 - Knowledge graph embedding
 - PRISM-like models
 - Addition MNIST
 - other custom models

# Quickstart


### Installation of PRISM
```
wget https://github.com/prismplp/prism/releases/download/v2.4.2a(T-PRISM)-prerelease/prism_tprism_pre_linux_ubuntu20.tar.gz

tar xvf prism_tprism_pre_linux_ubuntu20.tar.gz

export PATH=<current directory>/prism/bin:${PATH}
```

### Installation of T-PRISM

Installing python(Recommendation: Anaconda) and Pytorch.
Anaconda: https://www.anaconda.com/
Pytorch: https://pytorch.org/

#### Requirements:

- scikit-learn
- h5py
- graphviz
- protobuf==3.20.0

#### Installation
```
pip install "git+https://github.com/prismplp/prism.git#egg=t-prism&subdirectory=bin"
```

Note: Efficient data exchange using protocol buffers and tensor data exchange using h5 are not built in by default, so if you want to use these, you need to change the PRISM compilation options.


