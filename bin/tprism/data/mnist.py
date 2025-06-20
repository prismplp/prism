from sklearn.datasets import fetch_openml
import numpy as np
import h5py
import os

def get_mnist(out_filename_base, N_train=1000,N_test=0,individual=True,addition=False,cnn_input=False):
  path=os.path.dirname(out_filename_base)
  if len(path)>0:
    os.makedirs(path,exist_ok=True)
  #  Preparation of MNIST dataset
  mnist_X, mnist_y = fetch_openml('mnist_784', version=1, data_home=".", return_X_y=True)

  #  Creation of numpy object of MNIST
  x_all = mnist_X.astype(np.float32) / 255
  y_all = mnist_y.astype(np.int32)
  if N_train>0:
    if cnn_input:
      X_train=x_all.values[:N_train,:].reshape(-1,1,28,28)
    else:
      X_train=x_all.values[:N_train,:]
    y_train=y_all.values[:N_train]
  if N_test>0:
    if cnn_input:
      X_test=x_all.values[N_train:N_train+N_test,:].reshape(-1,1,28,28)
    else:
      X_test=x_all.values[N_train:N_train+N_test,:]
    y_test=y_all.values[N_train:N_train+N_test]

  # save MNIST dataset as h5 format
  with h5py.File(out_filename_base+".h5", 'w') as fp:
    if individual:
      if N_train>0:
        fp.create_group('train')
        for i in range(len(X_train)):
            fp['train'].create_dataset('tensor_in_'+str(i)+'_',data=X_train[i,:])
      if N_test>0:
        fp.create_group('test')
        for i in range(len(X_test)):
            fp['test'].create_dataset('tensor_in_'+str(i)+'_',data=X_test[i,:])
    else:
      if N_train>0:
        fp.create_group('train')
        fp['train'].create_dataset('tensor_in_',data=X_train)
      if N_test>0:
        fp.create_group('test')
        fp['test'].create_dataset('tensor_in_',data=X_test)
  ###
  # save MNIST dataset as dat format
  if addition:
    if N_train>0:
      with open(out_filename_base+".train.dat","w") as fp:
        for k in range(X_train.shape[0]//2):
          i=2*k
          j=2*k+1
          line="output_add(%d,%d,%d).\n"%(y_train[i]+y_train[j],i,j)
          fp.write(line)
    if N_test>0:
      with open(out_filename_base+".test.dat","w") as fp:
        for k in range(X_test.shape[0]//2):
          i=2*k
          j=2*k+1
          line="output_add(%d,%d,%d).\n"%(y_test[i]+y_test[j],i,j)
          fp.write(line)
  else:
    if N_train>0:
      with open(out_filename_base+".train.dat","w") as fp:
        for i in range(X_train.shape[0]):
            line="output(%d,%d).\n"%(y_train[i],i)
            fp.write(line)
    if N_test>0:
      with open(out_filename_base+".test.dat","w") as fp:
        for i in range(X_test.shape[0]):
            line="output(%d,%d).\n"%(y_test[i],i)
            fp.write(line)



