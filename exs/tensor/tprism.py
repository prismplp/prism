import tensorflow as tf
import json
import os
import expl_pb2
import re
import numpy as np
from google.protobuf import json_format
from itertools import chain
import collections
import argparse

import expl_graph
import draw_graph

class Flags(object):
	def __init__(self, args, options):
		self.internal_config = dict()
		self.args = args
		self.flags = {f.key:f.value for f in options.flags}
	def __getattr__(self, k):
		return dict.get(self.internal_config, k) \
			or getattr(self.args, k, None) \
			or dict.get(self.flags, k)
	def add(self, k, v):
		self.internal_config[k]=v
	def update(self):
		##
		batch_size=10
		if self.sgd_minibatch_size=="default":
			pass
		else:
			batch_size=int(self.sgd_minibatch_size)
		self.sgd_minibatch_size=batch_size
		self.add("sgd_minibatch_size",batch_size)
		##
		if self.max_iterate=="default":
			self.max_iterate=100
		else:
			self.max_iterate=int(self.max_iterate)
		##
		self.sgd_learning_rate=float(self.sgd_learning_rate)

def optimize(sess,goal_dataset,loss,flags):
	optimizer = tf.train.GradientDescentOptimizer(flags.sgd_learning_rate)
	train = [optimizer.minimize(l) for l in loss]
	init = tf.initialize_all_variables()
	sess.run(init)
	saver = tf.train.Saver()
	#print("starting at", "loss:", sess.run(total_loss))
	
	batch_size=flags.sgd_minibatch_size
	total_loss=[[] for _ in range(len(goal_dataset))]
	for step in range(flags.max_iterate):  
		for j,goal in enumerate(goal_dataset):
			ph_vars=goal["placeholders"]
			dataset=goal["dataset"]
			num=dataset.shape[1]
			num_itr=num//batch_size
			progbar=tf.keras.utils.Progbar(num_itr)
			for itr in range(num_itr):
				progbar.update(itr)
				feed_dict={ph:dataset[i,itr*batch_size:(itr+1)*batch_size] for i,ph in enumerate(ph_vars)}
				sess.run([train[j]], feed_dict=feed_dict)
				batch_loss=sess.run(loss[j],feed_dict=feed_dict)
				total_loss[j].extend(batch_loss)
			print("step", step, "loss:", np.mean(total_loss[j]))
	print("[SAVE]",flags.save_model)
	saver.save(sess, flags.save_model)

def save_draw_graph(g,base_name):
	html=draw_graph.show_graph(g)
	fp=open(base_name+".html","w")
	fp.write(html)
	dot=draw_graph.tf_to_dot(g)
	dot.render(base_name)
	
def build_goal_dataset(input_data,tensor_provider):
	goal_dataset=[]
	def to_index(item,ph_name):
		vocab_name=tensor_provider.ph_vocab[ph_name]
		vocab_name=list(vocab_name)[0]
		index=tensor_provider.vocab_set.get_values_index(vocab_name,item)
		return index
	to_index_func=np.vectorize(to_index)
	for d in input_data:
		ph_names=d["placeholders"]
		# TODO: multiple with different placeholders
		ph_vars=[tensor_provider.ph_var[ph_name] for ph_name in ph_names]
		dataset=[None for _ in ph_names]
		goal_data={"placeholders":ph_vars,"dataset":dataset}
		goal_dataset.append(goal_data)
		for i,ph_name in enumerate(ph_names):
			rec = d["records"]
			dataset[i]=to_index_func(rec[:,i],ph_name)
			print("*")
	for obj in goal_dataset:
		obj["dataset"]=np.array(obj["dataset"])
	return goal_dataset
		
# loss: goal x minibatch
def compute_preference_loss(graph,goal_inside):
	loss=[]
	output=[]
	gamma=1.00
	
	beta=1.0e-4
	reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
	reg_loss=beta*tf.reduce_mean(reg_losses)
	for rank_root in graph.root_list:
		goal_ids=[el.sorted_id for el in rank_root.roots]
		l1=goal_inside[goal_ids[0]]["inside"]
		l2=goal_inside[goal_ids[1]]["inside"]
		#l=tf.nn.relu(l2-l1+gamma)+reg_loss
		#l=tf.exp(l2-l1)+reg_loss
		l = tf.nn.softplus(1 * l2)+tf.nn.softplus(-1 * l1) + reg_loss
		loss.append(l)
		output.append([l1,l2])
	return loss,output
# loss: goal x minibatch
def compute_logll_score(graph,goal_inside):
	loss=[]
	output=[]
	gamma=1.00
	
	beta=1.0e-4
	reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
	reg_loss=beta*tf.reduce_mean(reg_losses)
	for rank_root in graph.root_list:
		goal_ids=[el.sorted_id for el in rank_root.roots]
		for sid in goal_ids:
			l1=goal_inside[sid]["inside"]
			output.append(l1)
		ll=-1*tf.reduce_mean(tf.log(output),axis=0)
		loss.append(ll)
	return loss,output


def run_training(g,sess,args):
	input_data = expl_graph.load_input_data(args.data)
	graph,options = expl_graph.load_explanation_graph(args.expl_graph,args.flags)
	flags=Flags(args,options)
	flags.update()
	##
	tensor_provider=expl_graph.SwitchTensorProvider()
	tensor_embedding = tensor_provider.build(graph,options,input_data,flags,load_embeddings=False)
	goal_inside = expl_graph.build_explanation_graph(graph,tensor_provider)
	goal_dataset=build_goal_dataset(input_data,tensor_provider)
	save_draw_graph(g,"test")
	loss,output=compute_preference_loss(graph,goal_inside)
	
	optimize(sess,goal_dataset,loss,flags)

def run_test(g,sess,args):
	input_data = expl_graph.load_input_data(args.data)
	graph,options = expl_graph.load_explanation_graph(args.expl_graph,args.flags)
	flags=Flags(args,options)
	flags.update()
	##
	tensor_provider=expl_graph.SwitchTensorProvider()
	tensor_embedding = tensor_provider.build(graph,options,input_data,flags,load_embeddings=True)
	goal_inside = expl_graph.build_explanation_graph(graph,tensor_provider)
	goal_dataset=build_goal_dataset(input_data,tensor_provider)
	save_draw_graph(g,"test")
	loss,output=compute_logll_score(graph,goal_inside)
	saver = tf.train.Saver()
	saver.restore(sess,flags.load_model)

	batch_size=flags.sgd_minibatch_size
	total_loss=[[] for _ in range(len(goal_dataset))]
	total_output=[[] for _ in range(len(goal_dataset))]
	for j,goal in enumerate(goal_dataset):
		ph_vars=goal["placeholders"]
		dataset=goal["dataset"]
		num=dataset.shape[1]
		num_itr=(num+batch_size-1)//batch_size
		progbar=tf.keras.utils.Progbar(num_itr)
		idx=list(range(num))
		for itr in range(num_itr):
			progbar.update(itr)
			temp_idx=idx[itr*batch_size:(itr+1)*batch_size]
			if len(temp_idx)<batch_size:
				padding_idx=np.zeros((batch_size,),dtype=np.int32)
				padding_idx[:len(temp_idx)]=temp_idx
				feed_dict={ph:dataset[i,padding_idx] for i,ph in enumerate(ph_vars)}
			else:
				feed_dict={ph:dataset[i,temp_idx] for i,ph in enumerate(ph_vars)}
			batch_loss,batch_output=sess.run([loss[j],output[j]],feed_dict=feed_dict)
			batch_output=np.transpose(batch_output)
			total_loss[j].extend(batch_loss[:len(temp_idx)])
			total_output[j].extend(batch_output[:len(temp_idx)])
		print("loss:", np.mean(total_loss[j]))
		print("output:", np.array(total_output[j]).shape)
	###
	"""
	total_output=np.array(total_output)
	print(total_output)
	acc=total_output[0,:,0]>total_output[0,:,1]
	print(np.sum(acc))
	acc=total_output[0,:,0]<=total_output[0,:,1]
	print(np.sum(acc))
	"""
	np.save(flags.output,total_output)


if __name__ == '__main__':
	# set random seed
	seed = 1234
	np.random.seed(seed)
	
	parser = argparse.ArgumentParser()
	parser.add_argument('mode', type=str,
			help='train/test')
	parser.add_argument('--config', type=str,
			default=None,
			help='config json file')
	
	parser.add_argument('--data', type=str,
			default="data.json",
			nargs='+',
			help='[from prolog] data json file')
	parser.add_argument('--expl_graph', type=str,
			default="expl.json",
			help='[from prolog] explanation graph json file')
	parser.add_argument('--flags', type=str,
			default="flags.json",
			help='[from prolog] flags json file')
	
	parser.add_argument('--save_model', type=str,
			default="./model.ckpt",
			help='model file')
	parser.add_argument('--load_model', type=str,
			default="./model.ckpt",
			help='model file')
	
	parser.add_argument('--save_embedding', type=str,
			default="./embedding.pkl",
			help='model file')
	parser.add_argument('--load_embedding', type=str,
			default="./embedding.pkl",
			help='model file')

	parser.add_argument('--output', type=str,
			default="./output.npy",
			help='model file')

	parser.add_argument('--gpu', type=str,
			default=None,
			help='constraint gpus (default: all) (e.g. --gpu 0,2)')
	parser.add_argument('--cpu',
			action='store_true',
			help='cpu mode (calcuration only with cpu)')

	parser.add_argument('--sgd_minibatch_size', type=str,
			default=None,
			help='[prolog flag]')
	parser.add_argument('--max_iterate', type=str,
			default=None,
			help='[prolog flag]')
	parser.add_argument('--sgd_learning_rate', type=float,
			default=0.01,
			help='[prolog flag]')
	


	args=parser.parse_args()
	# config
	if args.config is None:
		pass
	else:
		print("[LOAD] ",args.config)
		fp = open(args.config, 'r')
		config.update(json.load(fp))
	
	# gpu/cpu
	if args.cpu:
		os.environ['CUDA_VISIBLE_DEVICES'] = ""
	elif args.gpu is not None:
		os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
	# setup
	g = tf.Graph()
	with g.as_default():
		seed = 1234
		tf.set_random_seed(seed)
		with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
			# mode
			if args.mode=="train":
				run_training(g,sess,args)
			if args.mode=="test":
				run_test(g,sess,args)
			elif args.mode=="cv":
				run_train_cv(g,sess,args)

