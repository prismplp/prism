import numpy as np
import h5py
from sklearn.datasets import fetch_openml

mnist_X, mnist_y = fetch_openml('mnist_784', version=1, data_home=".", return_X_y=True)

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

with h5py.File("mnist.h5", "w") as fp:
    fp.create_group("train")
    fp["train"].create_dataset("tensor_in_", data=X_train)
    # fp['train'].create_dataset('label',data=y_train)
    fp.create_group("test")
    fp["test"].create_dataset("tensor_in_", data=X_test)
    # fp['test'].create_dataset('label',data=y_test)

fp = open("mnist.train.dat", "w")
n=X_train.shape[0]
for _ in range(500):
    i=np.random.randint(0,n)
    j=np.random.randint(0,n)
    #line = "output(%d,%d,%d,%d,%d).\n" % (i, j, y_train[i], y_train[j], y_train[i] + y_train[j])
    yi,yj = y_train[i], y_train[j]
    y=yi+yj
    line = "output(%d,%d,%d).\n" % (y, i, j)
    fp.write(line)

fp = open("mnist.test.dat", "w")
fp_a = open("mnist.test_ans.dat", "w")
n=X_test.shape[0]
for _ in range(100):
    i=np.random.randint(0,n)
    j=np.random.randint(0,n)
    yi,yj = y_test[i], y_test[j]
    y=yi+yj
    line = "output(%d,%d,%d).\n" % (y, i, j)
    fp.write(line)
    line = "output(%d,%d,%d,%d,%d).\n" % (y,yi,yj,i,j)
    fp_a.write(line)
