python rel_mapping.py ./Release/train.txt ./Release/valid.txt ./Release/test.txt 

python ./to_dat.py train Release/train.txt --correct_data Release/train.txt Release/valid.txt Release/test.txt 
python ./to_dat.py valid Release/valid.txt --correct_data Release/train.txt Release/valid.txt Release/test.txt 
python ./to_dat.py test Release/test.txt --correct_data Release/train.txt Release/valid.txt Release/test.txt 

python mix.py train --positive ./fb15k237.train_pos.dat --negative ./fb15k237.train_neg1.dat ./fb15k237.train_neg2.dat
python mix.py test --positive ./fb15k237.test_pos.dat --negative ./fb15k237.test_neg1.dat ./fb15k237.test_neg2.dat
python mix.py valid --positive ./fb15k237.valid_pos.dat --negative ./fb15k237.valid_neg1.dat ./fb15k237.valid_neg2.dat

python make_all.py  ./Release/test.txt --correct_data Release/train.txt Release/valid.txt Release/test.txt
