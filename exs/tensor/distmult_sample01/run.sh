#!/bin/sh
cd `dirname $0`

mkdir -p distmult_sample_tmp

upprism distmult_sample.psm 

tprism.py train \
    --internal_data_prefix ./distmult_sample_tmp/ \
    --sgd_loss preference_pair            \
    --max_iterate 10            \
    --sgd_learning_rate 0.01 --cpu

