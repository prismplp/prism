FROM python:3.6

MAINTAINER "" <>

RUN apt-get update -y \
    && apt-get install -y libhdf5-dev \
    && apt-get install -y libprotobuf-dev 
ADD https://github.com/google/protobuf/releases/download/v3.5.1/protobuf-all-3.5.1.tar.gz .
RUN tar xvf protobuf-all-3.5.1.tar.gz
RUN cd protobuf-3.5.1
RUN ./autogen.sh
RUN ./configure
RUN make
RUN make check
