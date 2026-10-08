# T-PRISM Multi-layer perceptron Program

This program "mnist.psm" is a simple neural network program in T-PRISM.

This program carries out supervised training of a feed-forward neural network, multi-layer perceptron, using a hand-written digit benchmark dataset.
"run.sh" shows how to run these scripts.

## Requirement: HDF5

This program saves the placeholder data in the HDF5 format (`mnist_tmp/mnist_data.train.h5` and `mnist_tmp/mnist_data.test.h5`).
Writing HDF5 files from a T-PRISM program additionally requires the HDF5 C++ library (e.g., `libhdf5-dev` on Ubuntu) and PRISM built with the option `USE_H5=1`:

```
cd src/c
USE_NPY=1 USE_H5=1 make -f Makefile.gmake
USE_NPY=1 USE_H5=1 make -f Makefile.gmake install
cd ../prolog
make && make install
```

The pre-built packages are built with `USE_NPY=1` only, so they cannot run this program as it is.
With them, see `mlp0` and `mlp1`, which do not use placeholders, or save the placeholder data in the NumPy format (see the T-PRISM manual).
