---
name: install-linux
description: Build PRISM and T-PRISM from source and set up their environment on Linux (Ubuntu 22.04, 24.04, 26.04). Covers the apt packages, the C and Prolog build with the USE_NPY, USE_H5, USE_PB and PROCTYPE=mp options, PATH, regression tests, the T-PRISM Python environment (venv, PyTorch, protobuf), the runtime libraries of the prebuilt packages, and fixes for common build and run errors. Use this skill whenever someone wants to install, compile, build, rebuild or set up PRISM or T-PRISM on Linux or Ubuntu (e.g. "コンパイル", "ビルド", "環境構築", "インストール"). Also use it when someone hits an error from make, upprism, prism or tprism on Linux, or prepares a Linux machine, container or CI job for PRISM, even if they don't say "install".
---

# Building PRISM / T-PRISM on Linux (Ubuntu)

PRISM has three layers, built in this order. The Prolog build runs the C binary,
so the C part must be built and **installed** first.

| Layer | Source | Output in `bin/` |
|---|---|---|
| C engine (links the B-Prolog static library) | `src/c/` | `prism_up_linux.bin` (`prism_mp_linux.bin` for MPI) |
| Prolog system | `src/prolog/` | `prism.out`, `batch.out`, `mpprism.out`, `foc.out` |
| T-PRISM Python package (`tprism` command) | `bin/tprism/` | installed with pip |

`bin/prism`, `bin/upprism` and `bin/mpprism` are shell scripts that start
`bin/prism_*_linux.bin` with `bin/bp.out` and the `.out` files, so everything
runs from `bin/` once it is built.

The commands below were checked on these Ubuntu versions:

| Ubuntu | gcc | Python | HDF5 | Notes |
|---|---|---|---|---|
| 22.04 | 11 | 3.10 | 1.10 | |
| 24.04 | 13 | 3.12 | 1.10 | |
| 26.04 | 15 | 3.14 | 1.14 | the MPI build needs one extra flag (section 3) |

## 1. Choose the build options

Options go on the `make` command line of the C part. The defaults are in
`src/c/makefiles/Makefile.opts.gmake`.

| Option | Enables | Needed for | Extra apt packages |
|---|---|---|---|
| `USE_NPY=1` | NumPy (`.npy`) output of tensors and data | T-PRISM embeddings saved as npy (e.g. `exs/tensor/transitive_closure01`) | none (bundled libnpy) |
| `USE_H5=1` | HDF5 output | T-PRISM placeholder data, which `save_placeholder_goals/2,3` writes as `.h5` by default (e.g. `exs/tensor/mlp`) | `pkg-config libhdf5-dev` |
| `USE_PB=1` | protocol buffer output (`pb`, `pbtxt`) | only when you need pb/pbtxt files (tprism itself reads JSON only) | `pkg-config libprotobuf-dev protobuf-compiler` |
| `PROCTYPE=mp` | MPI version (`prism_mp_linux.bin`, `mpprism`) | parallel EM learning | `libopenmpi-dev openmpi-bin` |

Recommendations:
- For PRISM alone, no option is needed.
- For T-PRISM, use `USE_NPY=1 USE_H5=1`. The `prism_tprism_pre_linux_ubuntu*` packages are built this way.
- The CI release package (`prism_linux_dev.auto.tar.gz`) uses only `USE_NPY=1`, so the HDF5-based examples do not work with it.
- `USE_PB=1` makes the binary depend on one exact libprotobuf version, so enable it only if you need pb or pbtxt output (see section 5).

## 2. Install the build tools

```sh
sudo apt-get update
sudo apt-get install -y g++ make git                          # always
sudo apt-get install -y pkg-config libhdf5-dev                # USE_H5=1
sudo apt-get install -y pkg-config libprotobuf-dev protobuf-compiler   # USE_PB=1
sudo apt-get install -y libopenmpi-dev openmpi-bin            # PROCTYPE=mp
```

In a Docker container you are root, so drop `sudo` and set
`DEBIAN_FRONTEND=noninteractive`.

## 3. Build and install

From the top of the repository:

```sh
git clone https://github.com/prismplp/prism.git && cd prism   # if not cloned yet

cd src/c
make -f Makefile.gmake -j USE_NPY=1 USE_H5=1
make -f Makefile.gmake USE_NPY=1 USE_H5=1 install      # copies prism_up_linux.bin into bin/
cd ../prolog
make
make install                                            # copies the .out files into bin/
cd ../..
export PATH=$PWD/bin:$PATH                              # add to ~/.bashrc to keep it
```

- Pass the same options to the build and the `install` step.
- **Run `make -f Makefile.gmake clean` in `src/c` before changing the options.** make does not track the flags, so objects compiled with other options are kept silently. For example, adding `USE_H5=1` without `clean` still gives a binary without HDF5 support.
- **The MPI build shares the object files, so clean before and after it:**
  ```sh
  cd src/c
  make -f Makefile.gmake clean
  make -f Makefile.gmake -j PROCTYPE=mp USE_NPY=1
  make -f Makefile.gmake PROCTYPE=mp USE_NPY=1 install    # bin/prism_mp_linux.bin
  make -f Makefile.gmake clean                            # before building the uniprocessor version again
  ```
  Notes on the MPI build:
  - **gcc >= 14 (Ubuntu 26.04)** rejects an implicit declaration of `pc_mp_abort_0` in `core/error.c`. Build with `CC="mpicc -Wno-error=implicit-function-declaration"` added to both `make` commands.
  - `mpprism` runs `mpirun`. The number of processes comes from `NPROCS` (default 4), and a machine file from `MACHINES`. Extra `mpirun` options go in `PRISM_MPIRUN_OPTS`.
  - Run the MPI tests with `cd testing && sh test_mp.sh`, which uses `NPROCS=2`.
  - Open MPI refuses to run as root, for example in a container. Set `OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1` there.
- Warnings are expected in both parts. For example, the Prolog compiler reports singleton variables.
- Only the Makefile build is maintained. The CMake build (`src/c/CMakeLists.txt`) fails at link time on Ubuntu 24.04 (`cannot find -lhdf5_cpp`).

## 4. Check the build

```sh
cd exs/base && upprism hmm 50 && cd ../..        # learns an HMM; prints "Final log likelihood: ..."
cd testing && sh test.sh && cd ..               # 23 programs; exits non-zero on the first failure
```

`test.sh` must run in `testing/`. It runs every `testing/programs/*.psm`
(expected to succeed) and every `testing/irregular_programs/*.psm` (expected to
fail). `prism` starts the interactive mode (Ctrl+D quits).

To check that the options are compiled in, look for the error messages of the
disabled formats in the binary:

```sh
strings bin/prism_up_linux.bin | grep -E "(hdf5|npy|Pb) format is not implemented"
# each line printed = a format that is NOT enabled
```

## 5. Protocol buffers (`USE_PB=1`) and the protobuf version

C++ code generated by protoc compiles only against the **same** libprotobuf
version. The bundled `src/c/external/expl.pb.cc/.h` are generated by protoc
3.21.12, the libprotobuf of Ubuntu 24.04 and 26.04.

| Ubuntu | libprotobuf (apt) | before `make ... USE_PB=1` |
|---|---|---|
| 22.04 | 3.12.4 | run `sh src/c/external/generate.sh` (regenerates with the installed protoc) |
| 24.04, 26.04 | 3.21.12 | nothing |

A version mismatch stops the build with one of these errors:

- `#error This file was generated by a newer version of protoc ...`
- `#error This file was generated by an older version of protoc ...`
- `#error "Protobuf C++ gencode is built with an incompatible version of"`

The Makefiles never regenerate the code, so that builds work without protoc.
Do not commit code regenerated by another protoc version.
`doc/devel/protoc.txt` has the full version policy.

## 6. T-PRISM Python environment

T-PRISM needs Python >= 3.10, PyTorch, NumPy, h5py, scikit-learn and protobuf.
`bin/requirements.txt` requires protobuf >= 4.21.12, or >= 5.27.0 on Python 3.14.
`pip install` of T-PRISM does not install these dependencies, so install them first.
On Ubuntu 24.04 and later, pip refuses to install into the system Python
(`externally-managed-environment`), so use a venv or conda.

```sh
sudo apt-get install -y python3-venv
python3 -m venv ~/venv/tprism && . ~/venv/tprism/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU; for CUDA see https://pytorch.org/
pip install numpy h5py scikit-learn -r bin/requirements.txt
pip install ./bin                    # this checkout; or "git+https://github.com/prismplp/prism.git#egg=t-prism&subdirectory=bin"
```

Optional packages:
- `pandas`: the MNIST examples need it, because `fetch_openml` of scikit-learn requires it.
- `networkx pyvis ipython`: for `tprism.plot`.
- `geotorch`: for constrained tensors. Install it with `pip install git+https://github.com/Lezcano/geotorch/`.

Check the environment with two examples:

```sh
cd exs/tensor/distmult_sample01 && sh run.sh && cd -   # JSON only, a few seconds
cd exs/tensor/mlp && sh run.sh && cd -                 # needs USE_H5=1 and pandas; downloads MNIST
```

About the MNIST example (`exs/tensor/mlp`):
- It ends with `accuracy: N / 10000`. With the default settings N is about 8900, while an untrained model gives about 1000.
- `run.sh` does not stop on errors, so look at the whole log, not only the exit status.
- OpenML downloads sometimes fail (`OpenMLError: Dataset with data_id 554 not found`). Rerun `python build_dataset.py` in the `mnist/` subdirectory until it succeeds.

## 7. Prebuilt packages instead of building

The binary packages contain the repository and a built `bin/`. Extract one and
add `prism/bin` to `PATH`. The binary needs a glibc at least as new as the
Ubuntu it was built on, plus the HDF5 runtime libraries if it was built with
`USE_H5=1`.

| Package | Built on | Runs on | apt runtime packages |
|---|---|---|---|
| `prism_tprism_pre_linux_ubuntu22.tar.gz` | 22.04 | 22.04, 24.04 | 22.04: `libhdf5-103-1 libhdf5-cpp-103-1`; 24.04: `libhdf5-103-1t64 libhdf5-cpp-103-1t64` |
| `prism_tprism_pre_linux_ubuntu24.tar.gz` | 24.04 | 24.04 | `libhdf5-103-1t64 libhdf5-cpp-103-1t64` |
| `prism_linux_dev.auto.tar.gz` (CI, `USE_NPY=1` only) | latest Ubuntu | that version and later | none |

Neither `ubuntu22` nor `ubuntu24` runs on Ubuntu 26.04. Its HDF5 is 1.14
(`libhdf5_serial.so.310`, packages `libhdf5-310 libhdf5-cpp-310`), and the
1.10 libraries (`.so.103`) are not available there. On 26.04, build from
source, or make a package on 26.04.

To make such a package, follow `tools/init_package.sh` in a clean container of
the target Ubuntu (e.g. `docker run --rm -it -v $PWD:/out ubuntu:22.04`):

1. Install the packages of section 2.
2. Clone the repository and build as in section 3 with `USE_NPY=1 USE_H5=1`.
3. Run `cd tools && sh init_package.sh`, then `tar czf /out/prism_tprism_pre_linux_ubuntu22.tar.gz prism/`.

Notes on packaging:
- `init_package.sh` clones GitHub master into `tools/prism` and copies the binaries from `bin/`. Push first, so that the packaged source matches the binaries.
- Its `cp: cannot stat` errors for the MPI, macOS and Windows binaries are expected.

## 8. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `Sorry, PRISM doesn't support this system.` (`prism`) or `Sorry, but PRISM doesn't support this system.` (`upprism`, `mpprism`) | `bin/prism_up_linux.bin` is missing (for `mpprism`, `bin/prism_mp_linux.bin`): build and `install` the C part (section 3). |
| `implicit declaration of function 'pc_mp_abort_0'` (MPI build, gcc >= 14) | Add `CC="mpicc -Wno-error=implicit-function-declaration"` (section 3). |
| `mpirun has detected an attempt to run as root` | Set `OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1`, or run as a normal user. |
| `Can't execute ../../bin/prism_up_linux.bin` while building `src/prolog` | The C part is not installed yet. Run `make -f Makefile.gmake install` in `src/c` first. |
| `[ERROR] hdf5 format is not implemented (please compile prism with the USE_H5 option)` | Rebuild the C part with `USE_H5=1` (after `make clean`). |
| `[ERROR] npy format is not implemented ...` | Rebuild with `USE_NPY=1` (after `make clean`). |
| `[ERROR] Pb format is not implemented ...`, or `Unknown format.` when saving pb/pbtxt | Rebuild with `USE_PB=1` (section 5). |
| `#error ... generated by a newer/older version of protoc`, `Protobuf C++ gencode is built with an incompatible version` | libprotobuf version differs from the generated code: `sh src/c/external/generate.sh` (section 5). |
| `cannot use keyword 'false' as enumeration constant` (gcc >= 15) | Old checkout. The current Makefile compiles C with `-std=gnu11`. Don't override `CFLAGS`/`CC` without it. |
| `C++ versions less than C++17 are not supported` (abseil, newer libprotobuf) | Old checkout. The current Makefile uses `-std=c++17`. |
| `libhdf5_serial.so.103 => not found` / `error while loading shared libraries` | Install the HDF5 runtime packages (section 7). |
| ``version `GLIBC_2.38' not found`` | The binary was built on a newer Ubuntu: use the package for your Ubuntu or build from source. |
| `error: externally-managed-environment` from pip | Use a venv or conda (section 6). |
| `TypeError: Descriptors cannot be created directly.` | Old `bin/tprism/expl_pb2.py` with protobuf >= 4.21. Update the checkout (current code works with protobuf >= 4.21.12). |
| `TypeError: Metaclasses with custom tp_new are not supported.` (Python 3.14) | protobuf < 5.27 on Python 3.14: `pip install -U protobuf`. |
| `ImportError: fetch_openml requires pandas.` | `pip install pandas`. |
| `unknown loss function: ...`, then `RuntimeError: output/loss is None in training` | Wrong `--sgd_loss`. Loss arguments are terms, e.g. `ce(0.1)` or `'ce_pl($placeholder2$)'`. Single-quote them in the shell so `$...` is not expanded. |

## Verifying in a clean environment

To reproduce a user's environment without touching the host, run the steps in
a container of the same Ubuntu version. Mount the repository read-only and
build in a copy:

```sh
docker run --rm -v "$PWD":/repo:ro ubuntu:24.04 bash -c '
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq && apt-get install -y -qq g++ make pkg-config libhdf5-dev >/dev/null
  cp -r /repo /tmp/prism && cd /tmp/prism
  { (cd src/c && make -f Makefile.gmake -j USE_NPY=1 USE_H5=1 && make -f Makefile.gmake USE_NPY=1 USE_H5=1 install) &&
    (cd src/prolog && make && make install); } >/tmp/build.log 2>&1 || { tail -20 /tmp/build.log; exit 1; }
  export PATH=$PWD/bin:$PATH
  cd testing && sh test.sh >/tmp/test.log 2>&1 && echo "build and test.sh: OK" || { tail -5 /tmp/test.log; exit 1; }'
```
