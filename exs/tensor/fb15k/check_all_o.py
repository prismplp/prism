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
parser.add_argument('--valid', type=str,
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

args=parser.parse_args()

basename=args.base
print('[LOAD]',args.rel)
with open(args.rel, 'r') as fp:
	rel_mapping=json.load(fp)

print('[LOAD]',args.entity)
with open(args.entity, 'r') as fp:
	ent_mapping=json.load(fp)

train_data={}
for filename in args.train:
	for line in open(filename):
		arr=line.strip().split("\t")
		rel=rel_mapping[arr[1]]
		e1=ent_mapping[arr[0]]
		e2=ent_mapping[arr[2]]
		key=(e1,rel)
		if key not in train_data:
			train_data[key]=[e2]
		else:
			train_data[key].append(e2)

valid_data={}
for filename in args.test:
	for line in open(filename):
		arr=line.strip().split("\t")
		rel=rel_mapping[arr[1]]
		e1=ent_mapping[arr[0]]
		e2=ent_mapping[arr[2]]
		key=(e1,rel)
		if key not in valid_data:
			valid_data[key]=[e2]
		else:
			valid_data[key].append(e2)

test_data=[]
for filename in args.test:
	for line in open(filename):
		arr=line.strip().split("\t")
		rel=rel_mapping[arr[1]]
		e1=ent_mapping[arr[0]]
		e2=ent_mapping[arr[2]]
		test_data.append((e1,rel,e2))

z=np.zeros((len(test_data),len(ent_mapping)))
for i,el in enumerate(test_data):
	key=(el[0],el[1])
	if key in train_data:
		for e in train_data[key]:
			z[i,e]=1
	if key in train_data:
		for e in valid_data[key]:
			z[i,e]=2
	z[i,el]=3
filename="fb15ko.check.npy"
print("[SAVE]",filename)
np.save(filename,z)

