#python rel_mapping.py FB15k/freebase_mtr100_mte100-*.txt
python make_all.py --test ./FB15k/freebase_mtr100_mte100-test.txt
python to_dat.py --train ./FB15k/freebase_mtr100_mte100-train.txt ./FB15k/freebase_mtr100_mte100-valid.txt --test ./FB15k/freebase_mtr100_mte100-test.txt 
python check_all.py

python build_dataset.py fb15k.train.dat --output ../fb15k_data.train.h5 --pair
python build_dataset.py fb15k.test_pos.dat --output ../fb15k_data.test.h5
python build_dataset.py fb15k.all1.dat --output ../fb15k_data.all1.h5
python build_dataset.py fb15k.all2.dat --output ../fb15k_data.all2.h5
