import numpy as np
import h5py
from sklearn.datasets import fetch_openml

mnist_X, mnist_y = fetch_openml('mnist_784', version=1, data_home=".", return_X_y=True, as_frame=False)

x_all = mnist_X.astype(np.float32) / 255
y_all = mnist_y.astype(np.int32)

X_train=x_all[:60000,:]
y_train=y_all[:60000]
X_test =x_all[60000:,:]
y_test =y_all[60000:]

X_train=X_train[y_train<3,:]
y_train=y_train[y_train<3]
X_test=X_test[y_test<3,:]
y_test=y_test[y_test<3]
X_train=X_train[:300]
y_train=y_train[:300]
X_test=X_test[:100]
y_test=y_test[:100]


with h5py.File("mnist.h5", 'w') as fp:
    fp.create_group('train')
    for i in range(len(X_train)):
        fp['train'].create_dataset('tensor_in_'+str(i)+'_',data=X_train[i,:])
    #fp['train'].create_dataset('label',data=y_train)
    fp.create_group('test')
    for i in range(len(X_test)):
        fp['test'].create_dataset('tensor_in_'+str(i)+'_',data=X_test[i,:])
    #fp['test'].create_dataset('label',data=y_test)

fp=open("mnist.train.dat","w")
for i in range(X_train.shape[0]):
    line="output(%d,%d).\n"%(y_train[i],i)
    fp.write(line)

fp=open("mnist.test.dat","w")
for i in range(X_test.shape[0]):
    line="output(%d,%d).\n"%(y_test[i],i)
    fp.write(line)

