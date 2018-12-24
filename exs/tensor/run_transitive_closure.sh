#!/bin/sh
cd `dirname $0`

N=100
p=0.001

cd ./transitive_closure
python build_dataset.py -n ${N} -p ${p}
cd ../

upprism transitive_closure.psm ${N}
tprism train \
	--internal_data_prefix transitive_closure_tmp/ \
	--embedding ./transitive_closure/tc_${N}_${p}.h5 \
	--sgd_loss nll --cycle --cpu
