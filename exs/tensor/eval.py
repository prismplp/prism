import numpy as np
import h5py

o=np.load("output.npy")
in_file = './fb15k_data.all1.h5'
infh = h5py.File(in_file, 'r')

test_in_file="fb15k_data.test.h5"
test_file = h5py.File(test_in_file, 'r')
rank_list_data=[]
for i,k in enumerate(infh):
	print(o[i])
	print(o[i].shape)
	prob=o[i]
	print(infh[k]["data"].value)
	print(infh[k]["data"].value.shape)
	print(infh[k]["data"].attrs.get("placeholders"))
	x=infh[k]["data"].value
	rels=set(x[:,1])
	print(rels)
	print(len(rels))
	result_data={}
	result_prob={}
	for r in rels:
		y_data=x[x[:,1]==r,:]
		y_prob=prob[x[:,1]==r]
		#np.save("eval/"+str(k)+"-"+str(r)+".npy",y_data)
		result_data[r]=y_data
		result_prob[r]=y_prob
	####
	####
	####
	print("============")
	test_data=test_file[k]["data"].value
	count=0
	rank_list=[]
	for t in test_data:
		r=t[1]
		s=t[0]
		y_data=result_data[r]
		y_prob=result_prob[r]
		ad=y_data[y_data[:,0]==s,:]
		ap=y_prob[y_data[:,0]==s]
		l=[(bp,ad[i,2]) for i,bp in enumerate(ap)]
		ll=[v for k,v in sorted(l,reverse=True)]
		rank=ll.index(t[2])+1.0
		rank_list.append(rank)
	rank_list_data.append(rank_list)
np.save("eval/rank.npy",rank_list_data)
mr=1.0/np.array(rank_list_data)
mrr=np.mean(mr)
print(mrr)
infh.close()



	

