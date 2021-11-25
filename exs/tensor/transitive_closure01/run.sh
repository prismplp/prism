#!/bin/sh
cd `dirname $0`

mkdir -p ./transitive_closure_tmp

##
## Construction of explanation graph
##
## This T-PRISM program makes internal data files ftom the Prolog (T-PRISM) program to the python program.
## Output: transitive_closure_tmp/
##    expl.json     : an explanation graph with cycles
##    flags.json    : flags
##    embedding.h5  : an embedded tensor for the adjacency matrix
##    embedding.txt : annotation for embedding.h5
##
upprism transitive_closure.psm

##
## Computing transitive closure
##
## Output: transitive_closure_tmp/
##    vocab.pkl
## --cycle option is required to solve a fixed point in the cyclic explanation graph.
## In this case, the loss function is not required.

tprism train \
    --input transitive_closure_tmp/ \
    --embedding transitive_closure_tmp/embedding.h5 \
    --cycle \
    --cpu

