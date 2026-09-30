# Build PRISM + T-PRISM from the checked-out repo the same way CI does (USE_NPY=1, no HDF5/protobuf).
# The bundled B-Prolog static libs are x86-64 only, so build for linux/amd64 (emulated on Apple Silicon).
FROM --platform=linux/amd64 ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential make git ca-certificates python3 python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/*
COPY repo /prism
WORKDIR /prism/src/c
RUN USE_NPY=1 make -f Makefile.gmake && USE_NPY=1 make -f Makefile.gmake install
WORKDIR /prism/src/prolog
RUN make && make install
ENV PATH=/prism/bin:$PATH
RUN python3 -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && python3 -m pip install --no-cache-dir numpy h5py scikit-learn "protobuf>=4.21.12" \
 && python3 -m pip install --no-cache-dir /prism/bin
WORKDIR /work
