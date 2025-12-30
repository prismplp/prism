import os

import h5py
import numpy as np
from sklearn.datasets import fetch_openml


def get_mnist(
    out_filename_base: str,
    N_train: int = 1000,
    N_test: int = 0,
    individual: bool = True,
    addition: bool = False,
    cnn_input: bool = False,
    verbose: bool = False,
) -> None:
    """
    Fetch MNIST from OpenML and export it to h5/dat files for tprism.

    Args:
        out_filename_base: Base filename (without extension) for exported files.
        N_train: Number of training samples to export.
        N_test: Number of test samples to export.
        individual: When True, store each sample as an individual dataset entry.
        addition: When True, generate labels for addition tasks; otherwise use class labels.
        cnn_input: When True, reshape images to NCHW format for CNN inputs.
        verbose: When True, print progress messages.
    """
    path = os.path.dirname(out_filename_base)
    if len(path) > 0:
        os.makedirs(path, exist_ok=True)

    if verbose:
        print(f"Fetching MNIST dataset from OpenML...")
    mnist_X, mnist_y = fetch_openml("mnist_784", version=1, data_home=".", return_X_y=True)

    x_all = mnist_X.astype(np.float32) / 255
    y_all = mnist_y.astype(np.int32)

    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray

    if N_train > 0:
        if cnn_input:
            X_train = x_all.values[:N_train, :].reshape(-1, 1, 28, 28)
        else:
            X_train = x_all.values[:N_train, :]
        y_train = y_all.values[:N_train]

    if N_test > 0:
        if cnn_input:
            X_test = x_all.values[N_train : N_train + N_test, :].reshape(-1, 1, 28, 28)
        else:
            X_test = x_all.values[N_train : N_train + N_test, :]
        y_test = y_all.values[N_train : N_train + N_test]

    # Save MNIST dataset as h5 format.
    if verbose:
        print(f"Saving MNIST dataset to {out_filename_base}.h5")
    with h5py.File(out_filename_base + ".h5", "w") as fp:
        if individual:
            if N_train > 0:
                fp.create_group("train")
                for i in range(len(X_train)):
                    fp["train"].create_dataset(f"tensor_in_{i}_", data=X_train[i, :])
            if N_test > 0:
                fp.create_group("test")
                for i in range(len(X_test)):
                    fp["test"].create_dataset(f"tensor_in_{i}_", data=X_test[i, :])
        else:
            if N_train > 0:
                fp.create_group("train")
                fp["train"].create_dataset("tensor_in_", data=X_train)
            if N_test > 0:
                fp.create_group("test")
                fp["test"].create_dataset("tensor_in_", data=X_test)
        

    # Save MNIST labels as dat format.
    if addition:
        if N_train > 0:
            if verbose:
                print(f"Saving addition trining dataset to {out_filename_base}.train.dat")
            with open(out_filename_base + ".train.dat", "w") as fp:
                for k in range(X_train.shape[0] // 2):
                    i = 2 * k
                    j = 2 * k + 1
                    line = "output_add(%d,%d,%d).\n" % (y_train[i] + y_train[j], i, j)
                    fp.write(line)
        if N_test > 0:
            if verbose:
                print(f"Saving addition test dataset to {out_filename_base}.test.dat")
            with open(out_filename_base + ".test.dat", "w") as fp:
                for k in range(X_test.shape[0] // 2):
                    i = 2 * k
                    j = 2 * k + 1
                    line = "output_add(%d,%d,%d).\n" % (y_test[i] + y_test[j], i, j)
                    fp.write(line)
    else:
        if N_train > 0:
            if verbose:
                print(f"Saving training dataset to {out_filename_base}.train.dat")
            with open(out_filename_base + ".train.dat", "w") as fp:
                for i in range(X_train.shape[0]):
                    line = "output(%d,%d).\n" % (y_train[i], i)
                    fp.write(line)
        if N_test > 0:
            if verbose:
                print(f"Saving test dataset to {out_filename_base}.test.dat")
            with open(out_filename_base + ".test.dat", "w") as fp:
                for i in range(X_test.shape[0]):
                    line = "output(%d,%d).\n" % (y_test[i], i)
                    fp.write(line)
