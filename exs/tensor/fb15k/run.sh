#python rel_mapping.py FB15k/freebase_mtr100_mte100-*.txt
#=> rel.json
#=> entity.json

#python make_all.py --test ./FB15k/freebase_mtr100_mte100-test.txt
#=> fb15k.all_o.dat
#=> fb15k.all_s.dat

#python to_dat.py --train ./FB15k/freebase_mtr100_mte100-train.txt ./FB15k/freebase_mtr100_mte100-valid.txt --test ./FB15k/freebase_mtr100_mte100-test.txt --base fb15k --train_pos --test_pos
#=> fb15k.train_pos.dat
#=> fb15k.test_pos.dat

python check_all.py
#=> fb15k.all_o.check.npy
#=> fb15k.all_s.check.npy


python build_dataset.py fb15k.train_pos.dat --output fb15k_data.train.h5
#python build_dataset.py fb15k.test_pos.dat --output fb15k_data.test.h5
python build_dataset.py fb15k.all_o.dat --output fb15k_data.all_o.h5
python build_dataset.py fb15k.all_s.dat --output fb15k_data.all_s.h5
echo "you can remove temporary files: '*.dat'"
