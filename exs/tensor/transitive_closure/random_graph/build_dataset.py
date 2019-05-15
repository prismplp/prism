import numpy as np
import h5py
import argparse

np.random.seed(1234)
parser = argparse.ArgumentParser()
parser.add_argument('-n', type=int,
	default=100,
	help='the numeber of entities')
parser.add_argument('-p', type=float,
	default=0.001,
	help='the probability of occurence of an edge')
parser.add_argument('--normalize',
	action='store_true',
	help='')
parser.add_argument('--check_eigen',
	action='store_true',
	help='')


args=parser.parse_args()

N=args.n
pe=args.p
filename="tc_"+str(N)+"_"+str(pe)+".h5"

# construntion of random graph
X_train=np.random.rand(N,N)
X_train[X_train<1.0-pe]=0.0
X_train[X_train>=1.0-pe]=1.0
for i in range(N):
	X_train[i,i]=1.0

# normalization
if args.normalize:
	x=np.sum(X_train,axis=1)
	M=max(x)
	X_train=X_train/(M+1.0)

# save the matrix
print("save matrix:",X_train.shape)
print("[SAVE]",filename)
with h5py.File(filename, 'w') as fp:
	fp.create_group('train')
	fp['train'].create_dataset('tensor_rel1_',data=X_train)

# display eigen values
if args.check_eigen:
	w,v=np.linalg.eig(X_train)
	print("maximum of eigen values:",max(w))
	print("minimum of eigen values:",min(w))

