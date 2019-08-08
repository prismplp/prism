FROM ubuntu:18.04

MAINTAINER "" <>
ENV PATH $PATH:/prism/bin:/root/anaconda3/bin
SHELL ["/bin/bash", "-c"]

ADD . /prism
ADD ./tools/anaconda_exp.sh /root/

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y && \
    apt-get install -y curl && \
    apt-get install -y expect && \
    cd prism/ && \
    apt-get install -y make build-essential libhdf5-dev pkg-config libprotobuf-dev protobuf-compiler && \
    apt-get clean && \
    cd src/c/external/ && \
    sh ./generate.sh && \
    cd ../ && \
    make -f Makefile.gmake && \
    make -f Makefile.gmake install && \
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

