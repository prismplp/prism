#!/bin/sh
cd `dirname $0`

cd ./mnist
python build_dataset.py
cd ../

upprism mnist.psm train
upprism mnist.psm test

tprism.py train --internal_data_prefix ./mnist_tmp/mnist. --data ./mnist_tmp/mnist_data.train.h5 --sgd_loss ce --embedding ./mnist/mnist.h5 --max_iterate 300 --sgd_minibatch_size 1000 --sgd_learning_rate 0.01
tprism.py test --internal_data_prefix ./mnist_tmp/mnist. --data ./mnist_tmp/mnist_data.test.h5 --sgd_loss ce --embedding ./mnist/mnist.h5 --output mnist_output.npy

python mnist_eval.py 

