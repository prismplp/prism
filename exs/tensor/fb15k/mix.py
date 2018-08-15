import sys
import collections
import json
import argparse
import numpy as np
import copy
	
parser = argparse.ArgumentParser()
parser.add_argument('file_type', type=str,
		help='train/test')
parser.add_argument('--positive', type=str,
		help='input filename e.g. ./Release/train.txt')
parser.add_argument('--negative', type=str,
		nargs='+',
		help='input filename e.g. ./Release/train.txt')
args=parser.parse_args()

pos_data=[]
for line in open(args.positive):
	pos=line.rindex('.')
	l=line[:pos]
	pos_data.append(l)

neg_data_list=[]
for filename in args.negative:
	neg_data=[]
	for line in open(filename):
		if line.strip()!="":
			pos=line.rindex('.')
			l=line[:pos]
			neg_data.append(l)
		else:
			neg_data.append(None)
	neg_data_list.append(neg_data)

print('[SAVE] fb15k.'+args.file_type+'.dat')
with open('fb15k.'+args.file_type+'.dat', 'w') as fp:
	for i,pos_item in enumerate(pos_data):
		for neg_data in neg_data_list:
			neg_item=neg_data[i]
			if neg_item is not None:
				fp.write("pair([")
				fp.write(pos_item)
				fp.write(",")
				fp.write(neg_item)
				fp.write("]).\n")


