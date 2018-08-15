import sys
import collections
import json
import argparse
	
parser = argparse.ArgumentParser()
parser.add_argument('input_filename', type=str,
		nargs='+',
		help='input filename e.g. ./Release/train.txt')
parser.add_argument('--rel',
		default="rel.json",
		help='relation json file')
parser.add_argument('--entity',
		default="entity.json",
		help='entity json file')
args=parser.parse_args()


rels=[]
entities=[]
for filename in args.input_filename:
	print('[LOAD]',filename)
	for line in open(filename):
		arr=line.strip().split("\t")
		rels.append(arr[1])
		entities.append(arr[0])
		entities.append(arr[2])

rel_counter = collections.Counter(rels)
rel_list=rel_counter.most_common()

ent_counter = collections.Counter(entities)
ent_list=ent_counter.most_common()

rel_mapping={}
for y in rel_list:
	rel_mapping[y[0]]=len(rel_mapping)
	print(y[0],y[1],len(rel_mapping))
ent_mapping={}
for y in ent_list:
	ent_mapping[y[0]]=len(ent_mapping)
	print(y[0],y[1],len(ent_mapping))

print('[SAVE]',args.rel)
with open(args.rel, 'w') as fp:
	json.dump(rel_mapping, fp)

print('[SAVE]',args.entity)
with open(args.entity, 'w') as fp:
	json.dump(ent_mapping, fp)


