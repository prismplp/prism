#!/bin/sh
cd `dirname $0`


##
## Construction of explanation graph
##
## This program parse two sequences and constructs an explanation graph. This program also exports
## internal data files to read python program.
## Output: expl.json: Exported explanation graph including placeholders for the 
##         given T-PRISM program, pcfg.psm.
## Output: flags.json: The flags and options specified in the T-PRISM program are
##         stored.
##

mkdir -p pcfg_tmp
upprism pcfg.psm

##
## Training and test
## input: The input files with the prefix pcfg_tmp/
## Minibatch SGD trainging is done by minimizing the loss function (nll: negative loss likelihood loss function
## where a scalar value representing likelihood should be embedded into a goal)
## The optional arguments determine training parameters: the number of epoch (--max_iterate 10), batch
## size (--sgd_minibatch_size 10), and learning rate (--sgd_learning_rate 0.01).
##
tprism train \
    --intermediate_data_prefix ./pcfg_tmp/ \
    --sgd_loss nll            \
    --max_iterate 10          \
    --sgd_minibatch_size 10   \
    --sgd_learning_rate 0.01

