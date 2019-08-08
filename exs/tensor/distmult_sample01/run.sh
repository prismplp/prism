#!/bin/sh
cd `dirname $0`

mkdir -p distmult_sample_tmp

##
## Construction of explanation graph
##
upprism distmult_sample.psm 

##
## Training and test
## input: The input files with the prefix mnist_tmp/mnist.
## Input: A placeholder is replaced using ./mnist_tmp/mnist_data.train.h5
##        embedding tensor is explicitely specified ./mnist/mnist.h5
## Minibatch SGD trainging is done by minimizing the loss function (ce_pl2: cross-entropy loss function
## where the second augument is given as a placeholder)
## The optional arguments determine training parameters: the number of epoch (--max_iterate 300), batch
##
tprism.py train \
    --intermediate_data_prefix ./distmult_sample_tmp/ \
    --sgd_loss preference_pair            \
    --max_iterate 10            \
    --sgd_learning_rate 0.01 --cpu

