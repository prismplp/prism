import sys
import collections
import json
import argparse
import numpy as np
import copy
	
parser = argparse.ArgumentParser()
parser.add_argument('--train', type=str,
		nargs='+',
		help='train filename e.g. ./Release/train.txt')
parser.add_argument('--test', type=str,
		nargs='+',
		help='filenames for test data e.g. ./Release/test.txt')
parser.add_argument('--base', type=str,
		default="base",
		help='basename')
parser.add_argument('--rel',
		default="rel.json",
		help='relation json file')
parser.add_argument('--entity',
		default="entity.json",
		help='entity json file')
parser.add_argument('--train_pos',
		action='store_true',
		help='[OUTPUT]')
parser.add_argument('--test_pos',
		action='store_true',
		help='[OUTPUT]')
parser.add_argument('--train_pair',
		action='store_true',
		help='[OUTPUT]')
parser.add_argument('--sample_num',type=int,
		default=10,
		help='negative sample')
args=parser.parse_args()

basename=args.base
print('[LOAD]',args.rel)
with open(args.rel, 'r') as fp:
	rel_mapping=json.load(fp)


print('[LOAD]',args.entity)
with open(args.entity, 'r') as fp:
	ent_mapping=json.load(fp)

correct_data={}
train_data=[]
for filename in args.train:
	for line in open(filename):
		arr=line.strip().split("\t")
		rel=rel_mapping[arr[1]]
		e1=ent_mapping[arr[0]]
		e2=ent_mapping[arr[2]]
		if rel not in correct_data:
			correct_data[rel]=set()
		correct_data[rel].add((e1,e2))
		train_data.append((e1,rel,e2))

test_data=[]
for filename in args.test:
	for line in open(filename):
		arr=line.strip().split("\t")
		rel=rel_mapping[arr[1]]
		e1=ent_mapping[arr[0]]
		e2=ent_mapping[arr[2]]
		test_data.append((e1,rel,e2))

if args.test_pos:
	filename=basename+".test_pos.dat"
	print('[SAVE]',filename)
	with open(filename, 'w') as fp:
		for item in test_data:
			fp.write("rel(%d,%d,%d).\n"%item)
if args.train_pos:
	filename=basename+".train_pos.dat"
	print('[SAVE]',filename)
	with open(filename, 'w') as fp:
		for item in train_data:
			fp.write("rel(%d,%d,%d).\n"%item)



if args.train_pair: # or args.test_pair:
	sample_num=args.sample_num
	negative_data1=[]
	for item in train_data:
		e2=item[2]
		rel=item[1]
		pairs=[]
		if not args.only_o:
			e_set1={(pair[0],e2) for pair in correct_data[rel]}
			neg_set=e_set1.difference(correct_data[rel])
			neg_list=np.array(list(neg_set))
			if len(neg_list)>0:
				i=np.random.choice(len(neg_list),sample_num)
				pairs=neg_list[i]
		negative_data1.append(pairs)


	negative_data2=[]
	for item in train_data:
		e1=item[0]
		rel=item[1]
		pairs=[]
		if not args.only_s:
			e_set2={(e1,pair[1]) for pair in correct_data[rel]}
			neg_set=e_set2.difference(correct_data[rel])
			neg_list=np.array(list(neg_set))
			if len(neg_list)>0:
				i=np.random.choice(len(neg_list),sample_num)
				pairs=neg_list[i]
		negative_data2.append(pairs)

	def write_pair(fp,pos_item,neg_item):
		fp.write("pair([")
		fp.write("rel(%d,%d,%d)"%pos_item)
		fp.write(",")
		fp.write("rel(%d,%d,%d)"%(neg_item[0],pos_item[1],neg_item[1]))
		fp.write("]).\n")

	if args.train_pair:
		filename=basename+".train_pair.dat"
		print('[SAVE]',filename)
		with open(filename, 'w') as fp:
			for i,item in enumerate(train_data):
				for item_neg1 in negative_data1[i]:
					write_pair(fp,item,item_neg1)
				for item_neg2 in negative_data2[i]:
					write_pair(fp,item,item_neg2)
	
