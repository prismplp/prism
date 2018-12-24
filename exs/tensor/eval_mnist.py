import numpy as np
import h5py

print("====== evaluation =======")
output_file="mnist_output.npy"
print("[LOAD]",output_file)
output=np.load(output_file)

in_file = './mnist_data.test.h5'
print("[LOAD]",in_file)
infh = h5py.File(in_file, 'r')

rank_list_data=[]
for i,k in enumerate(infh):
	prob=output[i]
	print("placeholders: ",infh[k]["data"].attrs.get("placeholders"))
	print("data: ",infh[k]["data"].value.shape)
	data=infh[k]["data"].value
	y=data[:,1]
	pred=np.argmax(prob,axis=1)
	print("prediction: ",pred)
	print("answer:",y)
	c=np.sum(y==pred)
	print("accuracy: ",c,"/",len(y))
#np.save("eval/rank_mnist.npy",rank_list_data)
#mr=1.0/np.array(rank_list_data)
#mrr=np.mean(mr)
#print(mrr)
infh.close()



	

