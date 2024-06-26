#!/bin/sh
cd `dirname $0`

mkdir -p markov_chain_tmp

upprism markov_chain.psm 

tprism train \
    --input ./markov_chain_tmp/ \
    --sgd_loss nll              \
    --max_iterate 10            \
    --sgd_learning_rate 0.01 --cpu


