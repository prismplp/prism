import numpy as np
with open("fb15k.train_pos.dat") as fp:
	train_set=set([l.strip() for l in fp.readlines()])
#with open("fb15k.valid_pos.dat") as fp:
#	valid_set=set([l.strip() for l in fp.readlines()])
with open("fb15k.test_pos.dat") as fp:
	test_set=set([l.strip() for l in fp.readlines()])
valid_set=set()
x=[]
for l in open("fb15k.all_o.dat"):
	k=l.strip()
	if k in train_set:
		x.append(1)
	elif k in valid_set:
		x.append(2)
	elif k in test_set:
		x.append(3)
	else:
		x.append(0)
print("[SAVE] fb15k.all_o.check.npy")
np.save("fb15k.all_o.check.npy",x)
x=[]
for l in open("fb15k.all_s.dat"):
	k=l.strip()
	if k in train_set:
		x.append(1)
	elif k in valid_set:
		x.append(2)
	elif k in test_set:
		x.append(3)
	else:
		x.append(0)
print("[SAVE] fb15k.all_s.check.npy")
np.save("fb15k.all_s.check.npy",x)
