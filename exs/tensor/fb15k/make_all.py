import sys
import collections
import json
import argparse
import numpy as np
import copy
	
parser = argparse.ArgumentParser()
parser.add_argument('--test', type=str,
		nargs='+',
		help='filenames for test data e.g. ./Release/test.txt')
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

test_data=[]
for filename in args.test:
	for line in open(filename):
		arr=line.strip().split("\t")
		rel=rel_mapping[arr[1]]
		e1=ent_mapping[arr[0]]
		e2=ent_mapping[arr[2]]
		test_data.append((e1,rel,e2))

ent_list=[v for k,v in ent_mapping.items()]
target1_set={(item[1],item[2]) for item in test_data}
target2_set={(item[0],item[1]) for item in test_data}
all_data1=[(e1,e2[0],e2[1]) for e2 in target1_set for e1 in ent_list]
all_data2=[(e1[0],e1[1],e2) for e1 in target2_set for e2 in ent_list]
print('[SAVE] fb15k.all_o.dat')
with open('fb15k.all_o.dat', 'w') as fp:
	for item in all_data1:
		fp.write("rel(%d,%d,%d).\n"%item)
print('[SAVE] fb15k.all_s.dat')
with open('fb15k.all_s.dat', 'w') as fp:
	for item in all_data2:
		fp.write("rel(%d,%d,%d).\n"%item)
	
