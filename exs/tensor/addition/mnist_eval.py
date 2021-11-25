import numpy as np
import h5py
from sklearn.metrics import confusion_matrix


print("====== evaluation =======")
output_file="mnist_output.npy"
print("[LOAD]",output_file)
output=np.load(output_file)
print("output:",output.shape)

in_file = './mnist/mnist.test.dat'
in_file_a = './mnist/mnist.test_ans.dat'
print("[LOAD]",in_file)
print("[LOAD]",in_file_a)
pred_y=[]
true_y=[]
fp1=open(in_file, 'r')
fp2=open(in_file_a, 'r')

for i,el in enumerate(zip(fp1,fp2)):
    line, line_a =el
    prob=output[i]
    pred=np.argmax(prob)
    #print("prediction: ",prob)
    print("prediction: ",pred)
    print("answer:",line.strip())
    #print("answer:",line_a.strip())
    print("")
    #c=np.sum(y==pred)
    y=line.split(",")[0].split("(")[1]
    pred_y.append(pred)
    true_y.append(int(y))
    #print("accuracy: ",c,"/",len(y))
#np.save("eval/rank_mnist.npy",rank_list_data)
#mr=1.0/np.array(rank_list_data)
#mrr=np.mean(mr)
#print(mrr)
#infh.close()
m = confusion_matrix(true_y, pred_y)
print("confusion matrix")
print(m)


