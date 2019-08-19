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

##
## Construction of explanation graph
##
## This T-PRISM program makes internal data files ftom the Prolog (T-PRISM) program to the python program.
## (an explanation graph, flags)
## Output: mnist_tmp/mnist.expl.json: Exported explanation graph including placeholders for the 
##         given T-PRISM program, mnist.psm.
## Output: mnist_tmp/mnist.flags.json: The flags and options specified in the T-PRISM program are
##         stored.
##
cd ../
mkdir -p ./transitive_closure_tmp
upprism transitive_closure.psm ${N}


tprism train \
	--intermediate_data_prefix transitive_closure_tmp/ \
	--embedding ./random_graph/tc_${N}_${p}.h5 \
	--sgd_loss nll \
	--cycle \
	--cpu

