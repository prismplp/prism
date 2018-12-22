import numpy as np
import h5py
import argparse
import numpy as np
import copy
import re

parser = argparse.ArgumentParser()
parser.add_argument('data', type=str,
		nargs='+',
		help='train filename e.g. ./Release/train.txt')
parser.add_argument('--output', type=str,
		default="a.h5",
		help='train filename e.g. ./Release/train.txt')
parser.add_argument('--pair',
		action='store_true',
		help='')
args=parser.parse_args()

data=[]
for filename in args.data:
	for line in open(filename):
		arr=re.split('[,\(\)]',line)
		if args.pair:
			print(arr)
			data.append([int(arr[2]),int(arr[3]),int(arr[4]),int(arr[7]),int(arr[8]),int(arr[9])])
		else:
			data.append([int(arr[1]),int(arr[2]),int(arr[3])])
data=np.array(data)

print("[SAVE]",args.output)
with h5py.File(args.output, 'w') as fp:
	fp.create_group('0')
	d=fp['0'].create_dataset('data',data=data)
	if args.pair:
		d.attrs['placeholders'] = np.array([b"$placeholder1$",b"$placeholder2$",b"$placeholder3$",b"$placeholder4$",b"$placeholder5$",b"$placeholder6$"])
	else:
		d.attrs['placeholders'] = np.array([b"$placeholder1$",b"$placeholder2$",b"$placeholder3$"])

