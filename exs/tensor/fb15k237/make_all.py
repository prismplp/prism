import sys
import collections
import json
import argparse
import numpy as np
import copy
	
parser = argparse.ArgumentParser()
parser.add_argument('input_filename', type=str,
		nargs='+',
		help='input filename e.g. ./Release/train.txt')
parser.add_argument('--correct_data', type=str,
		nargs='+',
		help='filenames for correct data e.g. ./Release/train.txt')
parser.add_argument('--rel',
		default="rel.json",
		help='relation json file')
parser.add_argument('--entity',
		default="entity.json",
		help='entity json file')
args=parser.parse_args()

print('[LOAD]',args.rel)
with open(args.rel, 'r') as fp:
	rel_mapping=json.load(fp)


print('[LOAD]',args.entity)
with open(args.entity, 'r') as fp:
	ent_mapping=json.load(fp)
ent_list=[v for k,v in ent_mapping.items()]

target_data=[]
for filename in args.input_filename:
	for line in open(filename):
		arr=line.strip().split("\t")
		rel=rel_mapping[arr[1]]
		e1=ent_mapping[arr[0]]
		e2=ent_mapping[arr[2]]
		target_data.append((e1,rel,e2))

correct_data={}
for filename in args.correct_data:
	for line in open(filename):
		arr=line.strip().split("\t")
		rel=rel_mapping[arr[1]]
		e1=ent_mapping[arr[0]]
		e2=ent_mapping[arr[2]]
		if rel not in correct_data:
			correct_data[rel]=set()
		correct_data[rel].add((e1,e2))

target1_set={(item[1],item[2]) for item in target_data}
target2_set={(item[0],item[1]) for item in target_data}

print(len(target_data))
print(len(target1_set))
print(len(target2_set))
all_data1=[]
for rel,e2 in target1_set:
	e_set1={(e,e2) for e in ent_list}
	#e_set1=e_set1.difference(correct_data[rel])
	for pair in e_set1:
		all_data1.append((pair[0],rel,pair[1]))


print('[SAVE] fb15k237.all1.dat')
with open('fb15k237.all1.dat', 'w') as fp:
	for item in all_data1:
		if item is not None:
			fp.write("rel(%d,%d,%d).\n"%item)

all_data2=[]
for e1,rel in target2_set:
	e_set2={(e1,e) for e in ent_list}
	#e_set2=e_set1.difference(correct_data[rel])
	for pair in e_set2:
		all_data2.append((pair[0],rel,pair[1]))

print('[SAVE] fb15k237.all2.dat')
with open('fb15k237.all2.dat', 'w') as fp:
	for item in all_data2:
		if item is not None:
			fp.write("rel(%d,%d,%d).\n"%item)

