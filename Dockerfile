FROM ubuntu:18.04

MAINTAINER "" <>
ENV PATH $PATH:/prism/bin:/root/anaconda3/bin:~/cmake-3.17.3-Linux-x86_64/bin
SHELL ["/bin/bash", "-c"]

ADD . /prism
ADD ./tools/anaconda_exp.sh /root/

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y && \
    apt-get install -y curl && \
    apt-get install -y expect && \
    apt-get install -y make build-essential libhdf5-dev pkg-config libprotobuf-dev protobuf-compiler && \
    apt-get install -y wget && \
    apt-get install -y openmpi-doc openmpi-bin libopenmpi-dev && \
    apt-get install -y ssh && \
    apt-get clean && \
    cd ~/ && \
    wget https://github.com/Kitware/CMake/releases/download/v3.17.3/cmake-3.17.3-Linux-x86_64.tar.gz && \
    tar zxvf cmake-3.17.3-Linux-x86_64.tar.gz && \
    cd /prism  && \
    cd src/c/external/ && \
    sh ./generate.sh && \
    cd ../ && \
    # make -f Makefile.gmake && \
    # make -f Makefile.gmake install && \
    mkdir -p cmake && \
    cd cmake && \
    cmake .. && \
    cmake --build . && \
    cmake --build . --target install && \
    cd ../ && \
    cd ../prolog/ && \
    make && \
    make install

RUN cd ~/ && \
    # echo "export PATH=/prism/bin:$PATH" >> .bashrc && \
    # source ~/.bashrc && \
    curl -O https://repo.anaconda.com/archive/Anaconda3-2019.03-Linux-x86_64.sh && \
    sh anaconda_exp.sh

RUN python -V && \
    pip install graphviz tensorflow
