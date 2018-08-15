import tensorflow as tf
import json
import expl_pb2
import re
import numpy as np
from google.protobuf import json_format

from itertools import chain
import collections

import inspect
import importlib
import glob
import os
import re
import pickle
import h5py

import op.base
   



def load_input_data(data_filename_list):
	input_data_list=[]
	for filename in data_filename_list:
		_,ext=os.path.splitext(filename)
		if ext==".h5":
			print("[LOAD]",filename)
			datasets=load_input_h5(filename)
		elif ext==".json":
			print("[LOAD]",filename)
			datasets=load_input_json(filename)
		elif ext[:5]==".json":
			print("[LOAD]",filename)
			datasets=load_input_json(filename)
		else:
			print("[ERROR]",data_filename)
		input_data_list.append(datasets)
	return merge_input_data(input_data_list)
	#return input_data_list

def load_input_json(filename):	
	input_data = expl_pb2.PlaceholderData()
	with open(filename, "r") as fp:
		input_data = json_format.Parse(fp.read(), input_data)
	datasets=[]
	for g in input_data.goals:
		phs=[ph.name for ph in g.placeholders]
		rs=[]
		for r in g.records:
			rs.append([item for item in items])
		dataset={"goal_id":g.id,"placeholders":phs,"records":rs}
		datasets.append(dataset)
	return datasets

def load_input_h5(filename):	
	infh = h5py.File(filename, 'r')
	datasets=[]
	for k in infh:
		goal_id=int(k)
		phs=[ph.decode() for ph in infh[k]["data"].attrs.get("placeholders")]
		rs=infh[k]["data"].value
		dataset={"goal_id":goal_id,"placeholders":phs,"records":rs}
		datasets.append(dataset)
	infh.close()
	return datasets
	
def merge_input_data(input_data_list):
	merged_data={}
	for datasets in input_data_list:
		for data in datasets:
			goal_id=data["goal_id"]
			if goal_id not in merged_data:
				merged_data[goal_id]=data
			else:
				merged_data[goal_id]["records"].extend(data[goal_id]["records"])
				
	return list(merged_data.values())

def load_explanation_graph(expl_filename,option_filename):
	graph = expl_pb2.ExplGraph()
	options = expl_pb2.Option()

	with open(expl_filename, "r") as fp:
		graph = json_format.Parse(fp.read(), graph)
	#f = open("expl.bin", "rb")
		#graph.ParseFromString(f.read())
	with open(option_filename, "r") as fp:
		options = json_format.Parse(fp.read(), options)
	return graph,options

def make_var_name(name):
	return re.sub(r'[\)\(\'$]+', "_", name)

def get_unique_list(seq):
	seen = []
	return [x for x in seq if x not in seen and not seen.append(x)]


#[['i'], ['i','l', 'j'], ['j','k']] => ['l','k']
def compute_output_template(template):
	counter=collections.Counter(chain.from_iterable(template))
	out_template=[k for k,cnt in counter.items() if cnt==1 and k!="b"]
	return out_template

class OperatorLoader:
	def __init__(self):
		self.operators={}
		self.base_module_name="op."
		self.module=None
	# a snake case operator name to class name
	def to_class_name(self,snake_str):
		components = snake_str.split('_')
		return ''.join(x.title() for x in components)
	# class name to a snake case operator name
	def to_op_name(self,cls_name):
		_underscorer1 = re.compile(r'(.)([A-Z][a-z]+)')
		_underscorer2 = re.compile('([a-z0-9])([A-Z])')
		subbed = _underscorer1.sub(r'\1_\2', cls_name)
		return _underscorer2.sub(r'\1_\2', subbed).lower()

	def get_operator(self,name):
		if name in self.operators:
			cls=self.operators[name]
			return cls
		else:
			return None
		
	def load_all(self,path):
		for fpath in glob.glob(os.path.join(path, '*.py')):
			print("[LOAD]",fpath)
			module_name = os.path.splitext(fpath)[0].replace(os.path.sep, '.')
			self.load_module(module_name)
			
	def load_module(self,module_name):
		#module_name=self.base_module_name+name
		module=importlib.import_module(module_name)
		for cls_name, cls in inspect.getmembers(module, inspect.isclass):
			if(issubclass(cls,op.base.BaseOperator)):
				print("[IMPORT]",cls_name)
				op_name=self.to_op_name(cls_name)
				self.operators[op_name]=cls
		return module

def build_explanation_graph(graph,tensor_provider):
	tensor_embedding=tensor_provider.tensor_embedding
	operator_loader=OperatorLoader()
	operator_loader.load_all("op")
	# converting explanation graph to computational graph
	goal_inside=[None]*len(graph.goals)
	for i in range(len(graph.goals)):
		g=graph.goals[i]
		path_inside=[]
		path_template=[]
		path_batch_flag=False
		for path in g.paths:
			## build template and inside for switches in the path
			sw_template=[]
			sw_inside=[]
			for sw in path.tensor_switches:
				ph=tensor_provider.get_placeholder_name(sw.name)
				if len(ph)>0:
					sw_template.append(['b']+list(sw.values))
					path_batch_flag=True
				else:
					sw_template.append(list(sw.values))
				sw_var=tensor_embedding[sw.name]
				sw_inside.append(sw_var)
				tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(sw_var))
			prob_sw_inside=1.0
			for sw in path.prob_switches:
				prob_sw_inside*=sw.inside
				
			## building template and inside for nodes in the path
			node_template=[]
			node_inside=[]
			for node in path.nodes:
				temp_goal=goal_inside[node.sorted_id]
				if temp_goal is None:
					print("[ERROR]")
					temp_goal=goal_inside[node.sorted_id]
					print(g.node.sorted_id)
					print(node.sorted_id)
					print(temp_goal)
					quit()
				temp_goal_inside=temp_goal["inside"]
				temp_goal_template=temp_goal["template"]
				if temp_goal["batch_flag"]:
					path_batch_flag=True
				node_inside.append(temp_goal_inside)
				node_template.append(temp_goal_template)
			## building template and inside for all elements (switches and nodes) in the path
			sw_node_template=sw_template+node_template
			sw_node_inside=sw_inside+node_inside
			path_v=sorted(zip(sw_node_template,sw_node_inside),key=lambda x: x[0])
			template=[x[0] for x in path_v]
			inside=[x[1] for x in path_v]
			# constructing einsum operation using template and inside
			out_template=compute_output_template(template)
			print(template,out_template)
			lhs=",".join(map(lambda x: "".join(x) ,template))
			rhs="".join(out_template)
			if path_batch_flag:
				rhs="b"+rhs
				out_template=['b']+out_template
			einsum_eq=lhs+"->"+rhs
			print(einsum_eq)
			print(inside)
			temp_inside=tf.einsum(einsum_eq,*inside)*prob_sw_inside
			## computing operaters
			for op in path.operators:
				#print(">>>",op.name)
				#print(">>>",op.values)
				cls=operator_loader.get_operator(op.name)
				#print(">>>",cls)
				op_obj=cls(op.values)
				temp_inside=op_obj.call(temp_inside)
				out_template=op_obj.get_output_template(out_template)
			##
			path_inside.append(temp_inside)
			path_template.append(out_template)
			##
		##
		path_template_list=get_unique_list(path_template)
		if len(path_template_list)!=1:
			print("[ERROR]")
		goal_inside[i]={
			"template":path_template_list[0],
			"inside":tf.reduce_sum(tf.stack(path_inside),axis=0),
			"batch_flag":path_batch_flag
			}
	return goal_inside

def build_variables(graph,options):
	index_range={el.index:el.range for el in options.index_range}
	tensors={}
	for g in graph.goals:
		for path in g.paths:
			for sw in path.tensor_switches:
				if sw.name not in tensors:
					tensors[sw.name]=set([])
				tensors[sw.name].add(tuple([el for el in sw.values]))
	# converting PRISM switches to Tensorflow Variables
	tensor_embedding={}
	dtype = tf.float32
	initializer=tf.contrib.layers.xavier_initializer(uniform = True)
	for sw_name,v in tensors.items():
		var_name=make_var_name(sw_name)
		shape_set=set()
		for e in v:
			shape=tuple([index_range[i] for i in e])
			shape_set.add(shape)
		if len(shape_set)!=1:
			print("[ERROR] missmatch")
		tensor_embedding[sw_name]= tf.get_variable(var_name, shape=shape, initializer=initializer, dtype=dtype)
	return tensor_embedding


class SwitchTensor:
	def __init__(self,sw_name):
		self.name=sw_name
		self.placeholder_names=self.get_placeholder_name(sw_name)
		self.vocab_name=self.get_vocab_name(sw_name)
		self.var_name=self.make_var_name(sw_name)
		self.shape_set=set([])
	
	def enabled_placeholder(self):
		return len(self.placeholder_names)==0

	def add_shape(self,shape):
		self.shape_set.add(shape)
		
	def get_placeholder_name(self,name):
		pattern=r'(\$placeholder[0-9]+\$)'
		m=re.finditer(pattern,name)
		names=[el.group(1) for el in m]
		return names

	def get_vocab_name(self,name):
		pattern=r'\$(placeholder[0-9]+)\$'
		m=re.sub(pattern,"",name)
		return self.make_var_name(m)

	def make_var_name(self,name):
		return re.sub(r'[\)\(\'$]+', "_", name)

class VocabSet:
	def __init__(self):
		pass
	def build(self,vocab_ph,ph_vocab,ph_values):
		vocab_group={vocab_name: None for vocab_name,_ in vocab_ph.items()}
		group_count=0
		vocab_group_values={}
		for vocab_name,phs in vocab_ph.items():
			if vocab_group[vocab_name] is None:
				next_vocabs=set([vocab_name])
				while len(next_vocabs)!=0:
					temp_vocabs=set([])
					for now_vocab_name in next_vocabs:
						vocab_group[now_vocab_name]=group_count
						phs=vocab_ph[now_vocab_name]
						for ph in phs:
							if group_count not in vocab_group_values:
								vocab_group_values[group_count] = set()
							vocab_group_values[group_count] = vocab_group_values[group_count] | ph_values[ph]
							for vocab_name2 in ph_vocab[ph]:
								if vocab_group[vocab_name2] is None:
									temp_vocabs.add(vocab_name2)
					next_vocabs=temp_vocabs
				group_count+=1
		self.vocab_group=vocab_group
		self.group_count=group_count
		self.vocab_ph=vocab_ph
		self.vocab_group_values=vocab_group_values
		self.group_value_index=self.build_group_value_index()
	def build_group_value_index(self):
		group_value_index={}
		for group,values in self.vocab_group_values.items():
			for i,v in enumerate(values):
				group_value_index[(group,v)]=i
		return group_value_index
		
	def get_values_index(self,vocab_name,value):
		g=self.vocab_group[vocab_name]
		key=(g,value)
		if key in self.group_value_index:
			return self.group_value_index[key]
		else:
			#print("[WARN] unknown value:",key)
			return 0
	
	def get_values(self,vocab_name):
		g=self.vocab_group[vocab_name]
		return self.get_group_values(g)
	
	def get_group_values(self,group):
		if group in self.vocab_group_values:
			return self.vocab_group_values[group]
		else:
			return None

	def get_group_vocabs(self,group):
		return [k for k,g in self.vocab_group if g==group]

	def get_group_placeholders(self,group):
		vocabs=self.get_vocabs(group)
		return [self.vocab_ph[v] for v in vocabs]
		

class SwitchTensorProvider:
	def __init__(self):
		pass

	def get_placeholder_name(self,name):
		return self.sw_info[name].placeholder_names

	def get_placeholder_var_name(self,name):
		return re.sub(r'\$',"",name)

	def build_ph_values(self,input_data):
		ph_values={}
		for g in input_data:
			for ph in g["placeholders"]:
				if ph not in ph_values:
					ph_values[ph]=set()
			placeholders=[ph for ph in g["placeholders"]]
			rt=np.transpose(g["records"])
			for i,item in enumerate(rt):
				ph_values[placeholders[i]]|=set(item)
		return ph_values

	def build(self,graph,options,input_data,flags,load_embeddings=False):
		index_range={el.index:el.range for el in options.index_range}
		# switch name => 
		sw_info={}
		for g in graph.goals:
			for path in g.paths:
				for sw in path.tensor_switches:
					if sw.name not in sw_info:
						sw_obj=SwitchTensor(sw.name)
						sw_info[sw.name]=sw_obj
					else:
						sw_obj=sw_info[sw.name]
					value_list=[el for el in sw.values]
					shape=tuple([index_range[i] for i in value_list])
					sw_obj.add_shape(shape)
		# build placeholders
		ph_values=self.build_ph_values(input_data)
		ph_var={}
		batch_size=flags.sgd_minibatch_size
		for ph_name,_ in ph_values.items():
			ph_var_name=self.get_placeholder_var_name(ph_name)
			ph_var[ph_name]=tf.placeholder(name=ph_var_name,shape=(batch_size,),dtype=tf.int32)
		# 
		vocab_shape={}
		vocab_ph={}
		ph_vocab={ph_name:set() for ph_name,_ in ph_values.items()}
		for sw_name,sw in sw_info.items():
			##
			## build vocab. shape
			
			if sw.vocab_name not in vocab_shape:
				vocab_shape[sw.vocab_name]=set()
			shape=list(sw.shape_set)[0]
			vocab_shape[sw.vocab_name].add(shape)
	
			if len(sw.shape_set)!=1:
				print("[ERROR] missmatch")
			if len(vocab_shape[sw.vocab_name])!=1:
				print("[ERROR] missmatch")
			##
			## build
			ph_list=sw.placeholder_names
			if len(ph_list)==0:
				if sw.vocab_name not in vocab_ph:
					vocab_ph[sw.vocab_name] =set()
			elif len(ph_list)==1:
				if sw.vocab_name not in vocab_ph:
					vocab_ph[sw.vocab_name] =set()
				vocab_ph[sw.vocab_name].add(ph_list[0])
				ph_vocab[ph_list[0]].add(sw.vocab_name)
			elif len(ph_list)>1:
				print("[ERROR] not supprted")
		for k,el in vocab_shape.items():
			vocab_shape[k]=list(el)[0]
		##
		## build vocab group
		if load_embeddings:
			with open(flags.load_embedding, mode="rb") as f:
				vocab_set = pickle.load(f)
		else:
			vocab_set=VocabSet()
			vocab_set.build(vocab_ph,ph_vocab,ph_values)
			with open(flags.save_embedding, mode="wb") as f:
				pickle.dump(vocab_set, f)
		##
		vocab_var={}
		dtype = tf.float32
		initializer=tf.contrib.layers.xavier_initializer()
		for vocab_name,shape in vocab_shape.items():
			values=vocab_set.get_values(vocab_name)
			if values is not None:
				s=[len(values)]+list(shape)
			else:
				s=list(shape)
			var_name=vocab_name
			var= tf.get_variable(var_name, shape=s, initializer=initializer, dtype=dtype)
			vocab_var[vocab_name]=var
		# converting PRISM switches to Tensorflow Variables
		tensor_embedding={}
		for sw_name,sw in sw_info.items():
			v_name=sw.vocab_name
			var_name=sw.var_name
			ph_list=sw.placeholder_names
			if len(ph_list)==0:
				var=vocab_var[v_name]
				tensor_embedding[sw_name]= var
			elif len(ph_list)==1:
				var=vocab_var[v_name]
				ph=ph_var[ph_list[0]]
				tensor_embedding[sw_name]=tf.gather(var,ph)
		self.vocab_var=vocab_var
		self.vocab_set=vocab_set
		self.ph_var=ph_var
		self.ph_vocab=ph_vocab
		self.tensor_embedding=tensor_embedding
		self.sw_info=sw_info
		return tensor_embedding


