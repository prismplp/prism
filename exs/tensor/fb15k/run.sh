#python rel_mapping.py FB15k/freebase_mtr100_mte100-*.txt

python ./to_dat.py train ./FB15k/freebase_mtr100_mte100-train.txt --correct_data ./FB15k/freebase_mtr100_mte100-train.txt ./FB15k/freebase_mtr100_mte100-valid.txt ./FB15k/freebase_mtr100_mte100-test.txt
python ./to_dat.py valid ./FB15k/freebase_mtr100_mte100-valid.txt --correct_data ./FB15k/freebase_mtr100_mte100-train.txt ./FB15k/freebase_mtr100_mte100-valid.txt ./FB15k/freebase_mtr100_mte100-test.txt
python ./to_dat.py test ./FB15k/freebase_mtr100_mte100-test.txt --correct_data ./FB15k/freebase_mtr100_mte100-train.txt ./FB15k/freebase_mtr100_mte100-valid.txt ./FB15k/freebase_mtr100_mte100-test.txt

python mix.py train --positive ./fb15k.train_pos.dat --negative ./fb15k.train_neg1.dat ./fb15k.train_neg2.dat
python mix.py test --positive ./fb15k.test_pos.dat --negative ./fb15k.test_neg1.dat ./fb15k.test_neg2.dat
python mix.py valid --positive ./fb15k.valid_pos.dat --negative ./fb15k.valid_neg1.dat ./fb15k.valid_neg2.dat

python make_all.py ./FB15k/freebase_mtr100_mte100-test.txt --correct_data ./FB15k/freebase_mtr100_mte100-train.txt ./FB15k/freebase_mtr100_mte100-valid.txt ./FB15k/freebase_mtr100_mte100-test.txt


