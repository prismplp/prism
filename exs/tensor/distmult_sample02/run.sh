#!/bin/sh
cd `dirname $0`

mkdir -p distmult_sample_tmp

upprism distmult_sample.psm 

tprism.py train \
    --intermediate_data_prefix ./distmult_sample_tmp/ \
    --sgd_loss preference_pair            \
    --data ./distmult_sample_tmp/data.h5     \
    --max_iterate 10            \
    --sgd_minibatch_size 5    \
    --sgd_learning_rate 0.01

