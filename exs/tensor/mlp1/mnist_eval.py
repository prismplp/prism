import numpy as np
import h5py

print("====== evaluation =======")
output_file="mnist_output.npy"
print("[LOAD]",output_file)
output=np.load(output_file)
print("output:",output.shape)

in_file = './mnist/mnist.test.dat'
print("[LOAD]",in_file)
for i,line in enumerate(open(in_file, 'r')):
    prob=output[i]
    pred=np.argmax(prob)
    print("prediction: ",prob)
    print("prediction: ",pred)
    print("answer:",line)
    #c=np.sum(y==pred)
    #print("accuracy: ",c,"/",len(y))
#np.save("eval/rank_mnist.npy",rank_list_data)
#mr=1.0/np.array(rank_list_data)
#mrr=np.mean(mr)
#print(mrr)
#infh.close()



	

