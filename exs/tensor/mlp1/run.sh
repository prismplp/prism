#!/bin/bash
cd `dirname $0`

##
## Construction of dataset
##
## This script downloads, if necessary, MNIST, a handwritten numeric character recognition data set,
## and converts it into the T-PRISM available format, i.e, .h5 format for multi-dimensional arrays
## and .dat format (text file) for a set of prolog facts.
## Input: mnist/MNIST_data/ (If this directory does not exist, it will be downloaded automatically)
## Output: mnist/mnist.h5: this file contains 4 arrays: train input matrix (55000 x 784) train label
##         vector (55000), train input matrix (55000 x 784), and  train label vector (55000).
##         In this  dataset, 784 means 28 x 28 gray-scale image, the number of training samples is 
##         55000, and the number of test samples is 10000
## Output: mnist/mnist.train.dat: this file contains output/2 facts representing train data. 
##         output(X,Y) means that the X-th sample (in mnist.h5) has a label Y for training.
## Output: mnist/mnist.test.dat: this file contains output/2 facts representing test data. 
##         output(X,Y) means that the X-th sample (in mnist.h5) has a label Y as the correct label.
##

cd ./mnist
python build_dataset.py

##
## Construction of explanation graph
##
## These two programs make intermediate data files between the T-PRISM (Prolog) and the python program.
## (explanation graph, flags, placeholders and values to replace the placeholders)
## Input: mnist/mnist.train.dat: A list of goals to construct arrays to replace placeholders in the
##        explanation graph. This replacement is done in the training phase data.
## Input: mnist/mnist.test.dat: A list of goals to construct arrays to replace placeholders in the
##        explanation graph. This replacement is done in the test phase data.
## Output: mnist_tmp/mnist_data.train.h5: Arrays containing values to replace the placeholders. This
##         file is constructed by combining two input files with a given program.
## Output: mnist_tmp/mnist.expl.json: Exported explanation graph including placeholders for the 
##         given T-PRISM program, mnist.psm.  This file is common in the training and test phase.
## Output: mnist_tmp/mnist.flags.json: The flags and options specified in the T-PRISM program are
##         stored. This file is common in the training and test phase.
##

cd ../
mkdir -p mnist_tmp
upprism mnist.psm train
upprism mnist.psm test

##
## Training and test
## input: The input files with the prefix mnist_tmp/mnist.
## Input: A placeholder is replaced using ./mnist_tmp/mnist_data.train.h5
##        embedding tensor is explicitely specified ./mnist/mnist.h5
## Minibatch SGD trainging is done by minimizing the loss function (ce_pl2: cross-entropy loss function
## where the second augument is given as a placeholder)
## The optional arguments determine training parameters: the number of epoch (--max_iterate 300), batch
## size (--sgd_minibatch_size 1000), and learning rate (--sgd_learning_rate 0.01).
## Output: a numpy array file (the number of samples x the number of classes), mnist_output.npy, that 
##         contains predcted class scores for each test sample.
##
tprism train \
    --input ./mnist_tmp/mnist    \
    --embedding ./mnist/mnist.h5 \
    --sgd_loss ce                \
    --max_iterate 50             \
    --sgd_minibatch_size 100     \
    --sgd_learning_rate 0.001

tprism test \
    --input ./mnist_tmp/mnist_test \
    --embedding ./mnist/mnist.h5   \
    --sgd_loss ce                  \
    --vocab ./mnist_tmp/mnist.vocab.pkl \
    --model ./mnist_tmp/mnist.model \
    --output mnist_output.npy

##
## Displaying accuracy
## Input: mnist_output.npy: prediction scores (the number of samples x the number of classes)
## Input: mnist_tmp/mnist_data.test.h5: correct labels
##
python mnist_eval.py 

