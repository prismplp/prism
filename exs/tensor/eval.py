import numpy as np
import h5py

label_file="fb15k/fb15k.all1.check.npy"
print("[LOAD]",label_file)
label=np.load(label_file)

output_file="fb15k_output.npy"
print("[LOAD]",output_file)
output=np.load(output_file)

in_file = './fb15k_data.all1.h5'
print("[LOAD]",in_file)
infh = h5py.File(in_file, 'r')


test_file = 'fb15k_data.test.h5'
print("[LOAD]",test_file)
test_fh = h5py.File(test_file, 'r')


rank_list_data=[]
for i,k in enumerate(infh):
	print(output[i])
	print(output[i].shape)
	prob=output[i]
	test_data=test_fh[k]["data"].value
	#print(infh[k]["data"].value)
	#print(test_data)
	print(infh[k]["data"].value.shape)
	print(infh[k]["data"].attrs.get("placeholders"))
	x=infh[k]["data"].value
	rels=set(x[:,1])
	print(rels)
	print(len(rels))
	result_data={}
	result_prob={}
	result_label={}
	for r in rels:
		y_data=x[x[:,1]==r,:]
		y_prob=prob[x[:,1]==r]
		y_label=label[x[:,1]==r]
		#np.save("eval/"+str(k)+"-"+str(r)+".npy",y_data)
		result_data[r]=y_data
		result_prob[r]=y_prob
		result_label[r]=y_label
		print("*",end="")
	####
	####
	####
	print("============")
	count=0
	rank_list=[]
	for t in test_data:
		r=t[1]
		s=t[0]
		o=t[2]
		y_data =result_data[r]
		y_prob =result_prob[r]
		y_label=result_label[r]
		
		c_data =y_data[y_data[:,2]==o,:]
		c_prob =y_prob[y_data[:,2]==o]
		c_label=y_label[y_data[:,2]==o]
		
		l =[(bp,c_data[i,0],c_label[i]) for i,bp in enumerate(c_prob)]
		sorted_ll=sorted(l,reverse=True)
		rank=1
		for el in sorted_ll:
			if el[2]==0:
				rank+=1
			if el[1]==s:
				break
		rank_list.append(rank)
	rank_list_data.append(rank_list)
rank_list_data=np.array(rank_list_data)
rank_filename="eval/rank_fb15k.npy"
print("[SAVE]",rank_filename)
np.save(rank_filename,rank_list_data)
mr=1.0/rank_list_data
mrr=np.mean(mr)
print("MRR:",mrr)
a=np.sum(rank_list_data<=10)
b=rank_list_data.shape[1]
print("HIT@10:",a*1.0/b)
infh.close()


