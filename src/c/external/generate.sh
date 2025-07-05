#!/bin/sh
cd `dirname $0`

#protoc --cpp_out=. --python_out=. expl.proto
# with mypy
# pip install mypy-protobuf
protoc --cpp_out=. --python_out=. --mypy_out=.  expl.proto
