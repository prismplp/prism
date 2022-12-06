#!/bin/sh
cd `dirname $0`

mkdir -p distmult_sample_tmp

##
## Construction of explanation graph with placeholders and substitution values (data.h5)
##
upprism distmult_sample.psm 

##
## Training
## input: the generated intermediate files generated by the above command
## --data  specifies the substitution values for placeholders in the explanation graph
##
tprism train \
    --input ./distmult_sample_tmp/    \
    --sgd_loss preference_pair        \
    --dataset ./distmult_sample_tmp/data.h5  \
    --max_iterate 10          \
    --sgd_minibatch_size 5    \
    --sgd_learning_rate 0.01
