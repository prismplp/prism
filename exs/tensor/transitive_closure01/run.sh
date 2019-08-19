#!/bin/sh
cd `dirname $0`

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
mkdir -p ./transitive_closure_tmp
upprism transitive_closure.psm

tprism train \
	--intermediate_data_prefix transitive_closure_tmp/ \
	--embedding transitive_closure_tmp/embedding.h5 \
	--sgd_loss nll \
	--cycle \
	--cpu

