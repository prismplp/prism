import sys
import collections
import json
import argparse
import numpy as np
import copy
	
parser = argparse.ArgumentParser()
parser.add_argument('file_type', type=str,
		help='train/test')
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

negative_data1=[]
for item in target_data:
	e2=item[2]
	rel=item[1]
	e_set1={(pair[0],e2) for pair in correct_data[rel]}
	neg_set=e_set1.difference(correct_data[rel])
	neg_list=list(neg_set)
	#print(len(neg_list),len(e_set1),len(correct_data[rel]))
	if len(neg_list)>0:
		i=np.random.choice(len(neg_list))
		neg_pair=neg_list[i]
		negative_data1.append((neg_pair[0],rel,neg_pair[1]))
	else:
		negative_data1.append(None)

negative_data2=[]
for item in target_data:
	e1=item[0]
	rel=item[1]
	e_set2={(e1,pair[1]) for pair in correct_data[rel]}
	neg_set=e_set2.difference(correct_data[rel])
	neg_list=list(neg_set)
	if len(neg_list)>0:
		i=np.random.choice(len(neg_list))
		neg_pair=neg_list[i]
		negative_data2.append((neg_pair[0],rel,neg_pair[1]))
	else:
		negative_data2.append(None)


print('[SAVE] fb15k.'+args.file_type+'_pos.dat')
with open('fb15k.'+args.file_type+'_pos.dat', 'w') as fp:
	for item in target_data:
		if item is not None:
			fp.write("rel(%d,%d,%d).\n"%item)
		else:
			fp.write("\n")

print('[SAVE] fb15k.'+args.file_type+'_neg1'+'.dat')
with open('fb15k.'+args.file_type+'_neg1'+'.dat', 'w') as fp:
	for item in negative_data1:
		if item is not None:
			fp.write("rel(%d,%d,%d).\n"%item)
		else:
			fp.write("\n")

print('[SAVE] fb15k.'+args.file_type+'_neg2'+'.dat')
with open('fb15k.'+args.file_type+'_neg2'+'.dat', 'w') as fp:
	for item in negative_data2:
		if item is not None:
			fp.write("rel(%d,%d,%d).\n"%item)
		else:
			fp.write("\n")
	


