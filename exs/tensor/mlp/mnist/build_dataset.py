import numpy as np
import h5py

from sklearn.datasets import fetch_openml

"""
from tensorflow.examples.tutorials.mnist import input_data
data_path = "../dataset"
mnist = fetch_mldata('MNIST original', data_home=data_path)

mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)

X_train = np.vstack([img.reshape(-1,) for img in mnist.train.images])
y_train = mnist.train.labels

X_test = np.vstack([img.reshape(-1,) for img in mnist.test.images])
y_test = mnist.test.labels

del mnist
print(X_train.shape)
print(y_train.shape)

print(X_test.shape)
print(y_test.shape)
"""

mnist_X, mnist_y = fetch_openml('mnist_784', version=1, data_home=".", return_X_y=True)

x_all = mnist_X.astype(np.float32) / 255
y_all = mnist_y.astype(np.int32)

X_train=x_all[:60000,:]
y_train=y_all[:60000]
X_test =x_all[60000:,:]
y_test =y_all[60000:]

with h5py.File("mnist.h5", 'w') as fp:
	fp.create_group('train')
	fp['train'].create_dataset('tensor_in_',data=X_train+0.1)
	#fp['train'].create_dataset('label',data=y_train)
	fp.create_group('test')
	fp['test'].create_dataset('tensor_in_',data=X_test+0.1)
	#fp['test'].create_dataset('label',data=y_test)

fp=open("mnist.train.dat","w")
for i in range(X_train.shape[0]):
	line="output(%d,%d).\n"%(i,y_train[i])
	fp.write(line)

fp=open("mnist.test.dat","w")
for i in range(X_test.shape[0]):
	line="output(%d,%d).\n"%(i,y_test[i])
	fp.write(line)
