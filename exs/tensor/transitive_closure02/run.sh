#!/bin/sh
cd `dirname $0`

##
## Construction of dataset
##
## This script generates random graph with N nodes where edges occurs independently with probability p.
## This script constructs an djacency matrix with the T-PRISM available format, i.e, .h5 format for 
## multi-dimensional arrays
## Output: tc_<N>_<p>.h5: this file contains output/2 facts representing train data. 
##         output(X,Y) means that the X-th sample (in mnist.h5) has a label Y for training.
##
N=100
p=0.001
cd ./random_graph
python build_dataset.py -n ${N} -p ${p}

cd ../
mkdir -p ./transitive_closure_tmp

##
## Construction of explanation graph
##
## This T-PRISM program makes internal data files ftom the Prolog (T-PRISM) program to the python program.
## Output: transitive_closure_tmp/
##    expl.json     : an explanation graph with cycles
##    flags.json    : flags
##
upprism transitive_closure.psm ${N}

##
## Computing transitive closure
##
## an embedding tensor is imported from ./random_graph/tc_${N}_${p}.h5

tprism train \
    --input transitive_closure_tmp/ \
    --const_embedding ./random_graph/tc_${N}_${p}.h5 \
    --cycle \
    --cpu

