import sys
import collections
import json
import argparse
	
parser = argparse.ArgumentParser()
parser.add_argument('input_filename', type=str,
		help='input filename e.g. ./Release/train.txt')
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

rels=[]
entities=[]
print('[LOAD]',args.input_filename)
for line in open(args.input_filename):
	arr=line.strip().split("\t")
	r=rel_mapping[arr[1]]
	if arr[0] not in ent_mapping:
		entities.append(arr[0])
		
	if arr[2] not in ent_mapping:
		entities.append(arr[2])
		
	if arr[1] not in rel_mapping:
		rels.append(arr[1])
rel_counter = collections.Counter(rels)
rel_list=rel_counter.most_common()

ent_counter = collections.Counter(entities)
ent_list=ent_counter.most_common()

print("=== relation (%d/%d)==="%(len(rel_list),len(rel_mapping)))
rel_mapping={}
for y in rel_list:
	rel_mapping[y[0]]=len(rel_mapping)
	print(y[0],y[1])

print("=== entity (%d/%d)==="%(len(ent_list),len(ent_mapping)))
ent_mapping={}
for y in ent_list:
	ent_mapping[y[0]]=len(ent_mapping)
	print(y[0],y[1])



