#!/usr/bin/env python

import tensorflow as tf
import json
import os
import re
import numpy as np
from google.protobuf import json_format
from itertools import chain
import collections
import argparse
import time

import tprism.expl_pb2 as expl_pb2
import tprism.expl_graph as expl_graph
import tprism.draw_graph as draw_graph

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

def optimize_solve(sess,goal_dataset,goal_inside,flags,embedding_generators):
	
	print("============")
	inside=[]
	for goal in goal_inside:
		print(goal)
		l1=goal["inside"]
		inside.append(l1)
	
	init = tf.global_variables_initializer()
	sess.run(init)
	feed_dict={}
	prev_loss=None
	for step in range(flags.max_iterate):  
		feed_dict={}
		for embedding_generator in embedding_generators:
			if embedding_generator is not None:
				feed_dict=embedding_generator.build_feed(feed_dict)
		out_inside=sess.run(inside,feed_dict=feed_dict)
		
		loss=0
		for embedding_generator in embedding_generators:
			if embedding_generator is not None:
				loss=embedding_generator.update(out_inside)
		print("step", step, "loss:", loss)
		if loss<1.0e-20:
			break
		if prev_loss is not None and not loss<prev_loss:
			pass
		prev_loss=loss
	for a in out_inside:
		print("~~~~~")
		print(a)
		print(np.sum(a>0))
		#print("step", step, "loss:", sess.run(total_loss,feed_dict=feed_dict))

def optimize_sgd(sess,goal_dataset,loss,flags,embedding_generators):
	total_loss=tf.reduce_sum(loss)
	optimizer = tf.train.AdamOptimizer(flags.sgd_learning_rate)
	train = optimizer.minimize(total_loss)

	init = tf.global_variables_initializer()
	sess.run(init)
	feed_dict={}
	for embedding_generator in embedding_generators:
		if embedding_generator is not None:
			feed_dict=embedding_generator.build_feed(feed_dict)
	print(feed_dict)
	print("starting at", "loss:", sess.run(loss,feed_dict=feed_dict))
	for step in range(flags.max_iterate):  
		feed_dict={}
		for embedding_generator in embedding_generators:
			if embedding_generator is not None:
				feed_dict=embedding_generator.build_feed(feed_dict)
		sess.run(train,feed_dict=feed_dict)
		print("step", step, "loss:", sess.run(total_loss,feed_dict=feed_dict))

		


def optimize(sess,goal_dataset,loss,flags,embedding_generators):
	
	#optimizer = tf.train.GradientDescentOptimizer(flags.sgd_learning_rate)
	optimizer = tf.train.AdamOptimizer(flags.sgd_learning_rate)
	#train = [optimizer.minimize(l) for l in loss]
	train=[]
	for l in loss:
		gradients, variables = zip(*optimizer.compute_gradients(l))
		gradients = [
		    None if gradient is None else tf.clip_by_norm(gradient, 5.0)
		    for gradient in gradients]
		optimize = optimizer.apply_gradients(zip(gradients, variables))
		train.append(optimize)
	
	init = tf.global_variables_initializer()
	sess.run(init)
	saver = tf.train.Saver()
	best_valid_loss=[None for _ in range(len(goal_dataset))]
	stopping_step=0
	batch_size=flags.sgd_minibatch_size
	for step in range(flags.max_iterate):  
		start_t = time.time()
		total_train_loss=[0.0 for _ in range(len(goal_dataset))]
		total_valid_loss=[0.0 for _ in range(len(goal_dataset))]
		for j,goal in enumerate(goal_dataset):
			ph_vars=goal["placeholders"]
			valid_ratio=0.1
			all_num=goal["dataset"].shape[1]
			train_num=int(all_num-valid_ratio*all_num)
			train_dataset=goal["dataset"][:,:train_num]
			num=train_dataset.shape[1]
			num_itr=num//batch_size
			if not flags.no_verb:
				progbar=tf.keras.utils.Progbar(num_itr)
			for itr in range(num_itr):
				feed_dict={ph:train_dataset[i,itr*batch_size:(itr+1)*batch_size] for i,ph in enumerate(ph_vars)}
				for embedding_generator in embedding_generators:
					if embedding_generator is not None:
						feed_dict=embedding_generator.build_feed(feed_dict)
				batch_loss,_=sess.run([loss[j],train[j]], feed_dict=feed_dict)
				if not flags.no_verb:
					bl=np.mean(batch_loss)
					progbar.update(itr,values=[("loss",bl)])
				total_train_loss[j]+=np.mean(batch_loss)/num_itr
			# valid
			valid_dataset=goal["dataset"][:,train_num:]
			num=valid_dataset.shape[1]
			num_itr=num//batch_size
			for itr in range(num_itr):
				feed_dict={ph:valid_dataset[i,itr*batch_size:(itr+1)*batch_size] for i,ph in enumerate(ph_vars)}
				for embedding_generator in embedding_generators:
					if embedding_generator is not None:
						feed_dict=embedding_generator.build_feed(feed_dict)
				batch_loss,_=sess.run([loss[j],train[j]], feed_dict=feed_dict)
				total_valid_loss[j]+=np.mean(batch_loss)/num_itr
			#
			print(": step", step, "train loss:", total_train_loss[j],"valid loss:", total_valid_loss[j])
			#
			if best_valid_loss[j] is None or best_valid_loss[j] > total_valid_loss[j]:
				best_valid_loss[j]=total_valid_loss[j]
				stopping_step=0
			else:
				stopping_step+=1
				if stopping_step==flags.sgd_patience:
					print("[SAVE]",flags.save_model)
					saver.save(sess, flags.save_model)
					return
		train_time = time.time() - start_t
		print("time:{0}".format(train_time) + "[sec]")
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
			if 0<len(tensor_provider.ph_vocab[ph_name]):
				dataset[i]=to_index_func(rec[:,i],ph_name)
			else: # goal placeholder
				dataset[i]=rec[:,i]
				print("goal_placeholder?")
				print(rec.shape)
				print(ph_name)
			print("*")
	for obj in goal_dataset:
		obj["dataset"]=np.array(obj["dataset"])
	return goal_dataset
	
def run_preparing(g,sess,args):
	input_data = expl_graph.load_input_data(args.data)
	graph,options = expl_graph.load_explanation_graph(args.expl_graph,args.flags)
	flags=Flags(args,options)
	flags.update()
	##
	loss_loader=expl_graph.LossLoader()
	loss_loader.load_all("loss/")
	loss_cls=loss_loader.get_loss(flags.sgd_loss)
	##
	tensor_provider=expl_graph.SwitchTensorProvider()
	embedding_generator=None
	if flags.embedding:
		embedding_generator=expl_graph.EmbeddingGenerator()
		embedding_generator.load(flags.embedding)
	tensor_embedding = tensor_provider.build(graph,options,input_data,flags,load_embeddings=False,embedding_generator=embedding_generator)
	#goal_inside = expl_graph.build_explanation_graph(graph,tensor_provider)
	#goal_dataset=build_goal_dataset(input_data,tensor_provider)
	#save_draw_graph(g,"test")
	
	#loss,output=loss_cls().call(graph,goal_inside,tensor_provider)
	

def run_training(g,sess,args):
	if args.data is not None:
		input_data = expl_graph.load_input_data(args.data)
	else:
		input_data = None
	graph,options = expl_graph.load_explanation_graph(args.expl_graph,args.flags)
	flags=Flags(args,options)
	flags.update()
	print(flags)
	##
	loss_loader=expl_graph.LossLoader()
	loss_loader.load_all("loss/")
	loss_cls=loss_loader.get_loss(flags.sgd_loss)
	##
	tensor_provider=expl_graph.SwitchTensorProvider()
	embedding_generator=None
	if flags.embedding:
		embedding_generator=expl_graph.EmbeddingGenerator()
		embedding_generator.load(flags.embedding)
	cycle_embedding_generator=None
	if flags.cycle:
		cycle_embedding_generator=expl_graph.CycleEmbeddingGenerator()
		cycle_embedding_generator.load(options)
	tensor_embedding = tensor_provider.build(graph,options,input_data,flags,load_embeddings=False,embedding_generator=embedding_generator)
	goal_inside = expl_graph.build_explanation_graph(graph,tensor_provider,cycle_embedding_generator)
	if input_data is not None:
		goal_dataset=build_goal_dataset(input_data,tensor_provider)
	else:
		goal_dataset=None
	save_draw_graph(g,"test")

	loss,output=loss_cls().call(graph,goal_inside,tensor_provider)
	
	with tf.name_scope('summary'):
		tf.summary.scalar('loss', loss)
		merged = tf.summary.merge_all()
		writer = tf.summary.FileWriter('./tf_logs', sess.graph)
	##	
	print("traing start") 
	vars_to_train = tf.trainable_variables()
	for var in vars_to_train:
		print(var,var.shape)
	##
	start_t = time.time()
	if flags.cycle:
		optimize_solve(sess,goal_dataset,goal_inside,flags,[embedding_generator,cycle_embedding_generator])
	elif goal_dataset is not None:
		optimize(sess,goal_dataset,loss,flags,[embedding_generator,cycle_embedding_generator])
	else:
		optimize_sgd(sess,goal_dataset,loss,flags,[embedding_generator,cycle_embedding_generator])
	train_time = time.time() - start_t
	print("traing time:{0}".format(train_time) + "[sec]")
	
def run_test(g,sess,args):
	input_data = expl_graph.load_input_data(args.data)
	graph,options = expl_graph.load_explanation_graph(args.expl_graph,args.flags)
	flags=Flags(args,options)
	flags.update()
	##
	loss_loader=expl_graph.LossLoader()
	loss_loader.load_all("loss/")
	loss_cls=loss_loader.get_loss(flags.sgd_loss)
	##
	tensor_provider=expl_graph.SwitchTensorProvider()
	embedding_generator=None
	if flags.embedding:
		embedding_generator=expl_graph.EmbeddingGenerator()
		embedding_generator.load(flags.embedding,key="test")
	tensor_embedding = tensor_provider.build(graph,options,input_data,flags,load_embeddings=True,embedding_generator=embedding_generator)
	goal_inside = expl_graph.build_explanation_graph(graph,tensor_provider)
	goal_dataset=build_goal_dataset(input_data,tensor_provider)
	save_draw_graph(g,"test")
	loss,output=loss_cls().call(graph,goal_inside,tensor_provider)
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
		if not flags.no_verb:
			progbar=tf.keras.utils.Progbar(num_itr)
		idx=list(range(num))
		for itr in range(num_itr):
			temp_idx=idx[itr*batch_size:(itr+1)*batch_size]
			if len(temp_idx)<batch_size:
				padding_idx=np.zeros((batch_size,),dtype=np.int32)
				padding_idx[:len(temp_idx)]=temp_idx
				feed_dict={ph:dataset[i,padding_idx] for i,ph in enumerate(ph_vars)}
			else:
				feed_dict={ph:dataset[i,temp_idx] for i,ph in enumerate(ph_vars)}
			
			if embedding_generator:
				feed_dict=embedding_generator.build_feed(feed_dict)
			batch_loss,batch_output=sess.run([loss[j],output[j]],feed_dict=feed_dict)
			if not flags.no_verb:
				progbar.update(itr)
			#print(batch_output.shape)
			#batch_output=np.transpose(batch_output)
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
			#default="data.json",
			default=None,
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
	parser.add_argument('--embedding', type=str,
			default=None,
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

	parser.add_argument('--no_verb',
			action='store_true',
			help='verb')
	
	parser.add_argument('--sgd_minibatch_size', type=str,
			default=None,
			help='[prolog flag]')
	parser.add_argument('--max_iterate', type=str,
			default=None,
			help='[prolog flag]')
	parser.add_argument('--sgd_learning_rate', type=float,
			default=0.01,
			help='[prolog flag]')
	parser.add_argument('--sgd_loss', type=str,
			default="preference_pair",
			help='[prolog flag] nll/preference_pair')
	parser.add_argument('--sgd_patience', type=int,
			default=3,
			help='[prolog flag] ')
	
	parser.add_argument('--cycle',
			action='store_true',
			help='cycle')

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
			if args.mode=="prepare":
				run_preparing(g,sess,args)
			if args.mode=="test":
				run_test(g,sess,args)
			elif args.mode=="cv":
				run_train_cv(g,sess,args)

