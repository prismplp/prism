import tensorflow as tf
import json
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

import tprism_module.expl_pb2 as expl_pb2
import tprism_module.op.base
import tprism_module.loss.base
   


class CycleEmbeddingGenerator():
	def __init__(self):
		self.embedding={}
		self.index_range={}
		self.tensor_shape={}
		self.feed_verb=False
	def load(self,options):
		self.index_range={el.index:el.range for el in options.index_range}
		self.tensor_shape={el.tensor_name:[d for d in el.shape] for el in options.tensor_shape}
	##
	def template2shape(self,template):
		return [self.index_range[t] for t in template]
	def is_dataset_embedding(self,vocab_name):
		return (vocab_name in self.dataset)
	def get_embedding(self,name,shape,node_id):
		ph_name=name+"_cyc"
		if ph_name in self.embedding:
			print("[GET]>",ph_name,":",self.embedding[ph_name]["tensor"])
			return self.embedding[ph_name]["tensor"]
		else:
			print("[CREATE]>",ph_name,":",shape)
			self.embedding[ph_name]={}
			self.embedding[ph_name]["tensor"]=tf.placeholder(name=ph_name,shape=shape,dtype=tf.float32)
			self.embedding[ph_name]["data"]=np.zeros(shape=shape,dtype=np.float32)
			self.embedding[ph_name]["id"]=node_id
			return self.embedding[ph_name]["tensor"]
	def build_feed(self,feed_dict):
		for ph_name,data in self.embedding.items():
			#batch_data=data[idx]
			batch_data=data["data"]
			ph_var=data["tensor"]
			if self.feed_verb:
				print("[INFO: feed]","node_id:",data["id"],"=>",ph_name)
			feed_dict[ph_var]=batch_data
		return feed_dict
	def update(self,out_inside):
		total_loss=0
		for ph_name,data in self.embedding.items():
			node_id=data["id"]
			print("[INFO: update] node_id:",node_id,"=>",ph_name)
			##
			loss=self.embedding[ph_name]["data"]-out_inside[node_id]
			total_loss+=np.sum(loss**2)	
			##
			self.embedding[ph_name]["data"]=out_inside[node_id]
			#a=0.5
			#self.embedding[ph_name]["data"]=(1.0-a)*self.embedding[ph_name]["data"]+a*out_inside[node_id]
		return total_loss


class EmbeddingGenerator():
	def __init__(self):
		self.feed_verb=False
		self.gather_in_flow=False
		self.dataset={}
		self.ph_var={}
		self.vocabset_ph_var=None
		self.vocabset_vocab_ph=None
	def load(self,filename,key="train"):
		print("[LOAD]",filename)
		infh = h5py.File(filename, 'r')
		if key in infh:
			for vocab_name in infh[key]:
				rs=infh[key][vocab_name].value
				self.dataset[vocab_name]=rs
				print("[LOAD VOCAB]",vocab_name)
		infh.close()
	def init(self,vocab_ph,ph_var):
		self.vocabset_ph_var=ph_var
		self.vocabset_vocab_ph=vocab_ph
	def is_dataset_embedding(self,vocab_name):
		return (vocab_name in self.dataset)
	
	def get_embedding(self,vocab_name,shape):
		ph_name=vocab_name+"_ph"
		if ph_name in self.ph_var:
			print("[GET]>",ph_name,":",self.vocabset_ph_var[ph_name])
			return self.ph_var[ph_name]
		elif ph_name in self.vocabset_ph_var:
			print("[GET]>",ph_name,":",self.vocabset_ph_var[ph_name])
			return self.vocabset_ph_var[ph_name]
		else:
			self.ph_var[ph_name]=tf.placeholder(name=ph_name,
				shape=shape,dtype=tf.float32)
			print("[CREATE]>",ph_name,":",shape)
			return self.ph_var[ph_name]

	def build_feed(self,feed_dict):
		for vocab_name,data in self.dataset.items():
			ph_name=vocab_name+"_ph"
			l=list(self.vocabset_vocab_ph[vocab_name])
			if len(l)>0:
				idx_ph_name=l[0]
				if self.feed_verb:
					print("[INFO: feed]",vocab_name,"=>",idx_ph_name)
				idx_ph_var=self.vocabset_ph_var[idx_ph_name]
				idx=feed_dict[idx_ph_var]
				batch_data=data[idx]
				if ph_name in self.ph_var:
					ph_var=self.ph_var[ph_name]
					feed_dict[ph_var]=batch_data
			else:
				if ph_name in self.ph_var:
					ph_var=self.ph_var[ph_name]
					feed_dict[ph_var]=data
				if self.feed_verb:
					print("[INFO: feed]",vocab_name,"=>",ph_name)
		return feed_dict
	def update(self,out_inside):
		pass

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
	print("[LOAD]",expl_filename)
	with open(expl_filename, "r") as fp:
		graph = json_format.Parse(fp.read(), graph)
	#f = open("expl.bin", "rb")
		#graph.ParseFromString(f.read())
	with open(option_filename, "r") as fp:
		options = json_format.Parse(fp.read(), options)
	return graph,options


class OperatorLoader:
	def __init__(self):
		self.operators={}
		self.base_module_name="tprism_module.op."
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
		assert name in self.operators, "%s is not found"%(name)
		cls=self.operators[name]
		assert cls is not None, "%s is not found"%(name)
		return cls
		
	def load_all(self,path):
		search_path=os.path.dirname(__file__)+"/"+path
		for fpath in glob.glob(os.path.join(search_path, '*.py')):
			print("[LOAD]",fpath)
			name = os.path.basename(os.path.splitext(fpath)[0])
			module_name=self.base_module_name+name
			module = importlib.machinery.SourceFileLoader(module_name,fpath).load_module()
			self.load_module(module)
				
	def load_module(self,module):
		for cls_name, cls in inspect.getmembers(module, inspect.isclass):
			if(issubclass(cls,tprism_module.op.base.BaseOperator)):
				print("[IMPORT]",cls_name)
				op_name=self.to_op_name(cls_name)
				self.operators[op_name]=cls

class LossLoader:
	def __init__(self):
		self.module=None
		self.base_module_name="tprism_module.loss."
		self.losses={}
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

	def get_loss(self,name):
		if name in self.losses:
			cls=self.losses[name]
			return cls
		else:
			return None
	
	def load_all(self,path):
		search_path=os.path.dirname(__file__)+"/"+path
		for fpath in glob.glob(os.path.join(search_path, '*.py')):
			print("[LOAD]",fpath)
			name = os.path.basename(os.path.splitext(fpath)[0])
			module_name=self.base_module_name+name
			module = importlib.machinery.SourceFileLoader(module_name,fpath).load_module()
			self.load_module(module)
		
	def load_module(self,module):
		for cls_name, cls in inspect.getmembers(module, inspect.isclass):
			if(issubclass(cls,tprism_module.loss.base.BaseLoss)):
				print("[IMPORT]",cls_name)
				op_name=self.to_op_name(cls_name)
				self.losses[op_name]=cls



def get_unique_list(seq):
	seen = []
	return [x for x in seq if x not in seen and not seen.append(x)]


#[['i'], ['i','l', 'j'], ['j','k']] => ['l','k']
def compute_output_template(template):
	counter=collections.Counter(chain.from_iterable(template))
	out_template=[k for k,cnt in counter.items() if cnt==1 and k!="b"]
	return sorted(out_template)

def build_explanation_graph_template(graph,tensor_provider,operator_loader=None,cycle_node=[]):
	tensor_embedding=tensor_provider.tensor_embedding
	# checking template
	goal_template=[None]*len(graph.goals)
	for i in range(len(graph.goals)):
		g=graph.goals[i]
		path_template=[]
		path_batch_flag=False
		for path in g.paths:
			## build template and inside for switches in the path
			sw_template=[]
			for sw in path.tensor_switches:
				ph=tensor_provider.get_placeholder_name(sw.name)
				if len(ph)>0:
					sw_template.append(['b']+list(sw.values))
					path_batch_flag=True
				else:
					sw_template.append(list(sw.values))
			## building template and inside for nodes in the path
			node_template=[]
			cycle_detected=False
			for node in path.nodes:
				temp_goal=goal_template[node.sorted_id]
				if temp_goal is None:
					#cycle
					if node.sorted_id not in cycle_node:
						 cycle_node.append(node.sorted_id)
					cycle_detected=True
					continue
				if len(temp_goal["template"])>0:
					temp_goal_template=temp_goal["template"]
					if temp_goal["batch_flag"]:
						path_batch_flag=True
					node_template.append(temp_goal_template)
			if cycle_detected:
				continue
			sw_node_template=sw_template+node_template
			#template=[x for x in sorted(sw_node_template)]
			template=sw_node_template
			# constructing einsum operation using template and inside
			out_template=compute_output_template(template)
			if len(template)>0: # condition for einsum
				if path_batch_flag:
					out_template=['b']+out_template
			## computing operaters
			for op in path.operators:
				print(op.name)
				cls=operator_loader.get_operator(op.name)
				op_obj=cls(op.values)
				out_template=op_obj.get_output_template(out_template)
			path_template.append(out_template)
			##
		##
		path_template_list=get_unique_list(path_template)
		if len(path_template_list)==0:
			goal_template[i]={"template":[],"batch_flag":False}
		else:
			if len(path_template_list)!=1:
				print("[WARNING] missmatch indices:",path_template_list)
			goal_template[i]={"template":path_template_list[0],"batch_flag":path_batch_flag}
	##
	return goal_template,cycle_node
	

def build_explanation_graph(graph,tensor_provider,cycle_embedding_generator=None):
	tensor_embedding=tensor_provider.tensor_embedding
	operator_loader=OperatorLoader()
	operator_loader.load_all("op")
	goal_template,cycle_node=build_explanation_graph_template(graph,tensor_provider,operator_loader)
	print(">>",goal_template)
	print(">>",cycle_node)
	#goal_template
	# converting explanation graph to computational graph
	goal_inside=[None]*len(graph.goals)
	for i in range(len(graph.goals)):
		g=graph.goals[i]
		print("=== tensor equation (node_id:%d, %s) ==="%(g.node.sorted_id,g.node.goal.name))
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
			node_scalar_inside=[]
			for node in path.nodes:
				temp_goal=goal_inside[node.sorted_id]
				
				if node.sorted_id in cycle_node:
					name=node.goal.name
					template=goal_template[node.sorted_id]["template"]
					shape=cycle_embedding_generator.template2shape(template)
					temp_goal_inside=cycle_embedding_generator.get_embedding(name,shape,node.sorted_id)
					temp_goal_template=template
					node_inside.append(temp_goal_inside)
					node_template.append(temp_goal_template)
				elif temp_goal is None:
					print("  [ERROR] cycle node is detected")
					temp_goal=goal_inside[node.sorted_id]
					print(g.node.sorted_id)
					print(node)
					print(node.sorted_id)
					print(temp_goal)
					quit()
				elif len(temp_goal["template"])>0:
					#tensor
					
					temp_goal_inside=temp_goal["inside"]
					temp_goal_template=temp_goal["template"]
					if temp_goal["batch_flag"]:
						path_batch_flag=True
					node_inside.append(temp_goal_inside)
					node_template.append(temp_goal_template)
				else:# scalar
					node_scalar_inside.append(temp_goal["inside"])
			## building template and inside for all elements (switches and nodes) in the path
			sw_node_template=sw_template+node_template
			sw_node_inside=sw_inside+node_inside
			path_v=sorted(zip(sw_node_template,sw_node_inside),key=lambda x: x[0])
			template=[x[0] for x in path_v]
			inside=[x[1] for x in path_v]
			# constructing einsum operation using template and inside
			out_template=compute_output_template(template)
			#print(template,out_template)
			out_inside=prob_sw_inside
			if len(template)>0: # condition for einsum
				lhs=",".join(map(lambda x: "".join(x) ,template))
				rhs="".join(out_template)
				if path_batch_flag:
					rhs="b"+rhs
					out_template=['b']+out_template
				einsum_eq=lhs+"->"+rhs
				print("  ",einsum_eq)
				print("  ",inside)
				out_inside=tf.einsum(einsum_eq,*inside)*out_inside
			for scalar_inside in node_scalar_inside:
				out_inside=scalar_inside*out_inside
			## computing operaters
			for op in path.operators:
				print(">>>",op.name)
				print(">>>",op.values)
				cls=operator_loader.get_operator(op.name)
				print(">>>",cls)
				op_obj=cls(op.values)
				out_inside=op_obj.call(out_inside)
				out_template=op_obj.get_output_template(out_template)
			##
			path_inside.append(out_inside)
			path_template.append(out_template)
			##
		##
		path_template_list=get_unique_list(path_template)
		
		if len(path_template_list)==0:
			goal_inside[i]={
				"template":[],
				"inside":np.array(1),
				"batch_flag":False
				}
		else:
			if len(path_template_list)!=1:
				print("[WARNING] missmatch indices:",path_template_list)
			goal_inside[i]={
				"template":path_template_list[0],
				"inside":tf.reduce_sum(tf.stack(path_inside),axis=0),
				"batch_flag":path_batch_flag
				}
	return goal_inside

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
		return re.sub(r'[\[\],\)\(\'$]+', "_", name)

class VocabSet:
	def __init__(self):
		# vocab name => vocab group index
		self.vocab_group=None
		# maximum of vocab index
		self.group_count=None
		# vocab name => a set of placeholder names
		self.vocab_ph=None
		# vocab_group index => a list of values
		self.vocab_group_values=None
		# (vocab_group index, value) => value index i
		self.group_value_index=None
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
	
		self.vocab_group_values={k:list(v) for k,v in vocab_group_values.items()}
		self.group_value_index=self.build_group_value_index()
	def build_group_value_index(self):
		group_value_index={}
		for group,values in self.vocab_group_values.items():
			for i,v in enumerate(sorted(values)):
				#group_value_index[(group,v)]=i
				group_value_index[(group,v)]=int(v)
				#print((group,v,i))
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

	def build(self,graph,options,input_data,flags,load_embeddings=False,embedding_generator=None):
		index_range={el.index:el.range for el in options.index_range}
		tensor_shape={el.tensor_name:[d for d in el.shape] for el in options.tensor_shape}
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
					if sw.name in tensor_shape:
						shape=tuple(tensor_shape[sw.name])
					else:
						shape=tuple([index_range[i] for i in value_list])
					sw_obj.add_shape(shape)
		# build placeholders
		if input_data is not None:
			ph_values=self.build_ph_values(input_data)
			ph_var={}
			batch_size=flags.sgd_minibatch_size
			for ph_name,_ in ph_values.items():
				ph_var_name=self.get_placeholder_var_name(ph_name)
				ph_var[ph_name]=tf.placeholder(name=ph_var_name,shape=(batch_size,),dtype=tf.int32)
			ph_vocab={ph_name:set() for ph_name,_ in ph_values.items()}
		else:
			ph_vocab={}
			ph_values={}
			ph_var={}
		vocab_ph={}
		# 
		vocab_shape={}
		for sw_name,sw in sw_info.items():
			##
			## build vocab. shape
			
			if sw.vocab_name not in vocab_shape:
				vocab_shape[sw.vocab_name]=set()
			vocab_shape[sw.vocab_name]|=sw.shape_set
	
			"""
			if len(sw.shape_set)!=1:
				print("[ERROR] missmatch")
			if len(vocab_shape[sw.vocab_name])!=1:
				print("[ERROR] missmatch")
			"""
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
				print("[ERROR] not supprted: one placeholder for one term")
		##
		if embedding_generator:
			embedding_generator.init(vocab_ph,ph_var)
		## build vocab group
		if load_embeddings:
			print("[LOAD]",flags.vocab)
			with open(flags.vocab, mode="rb") as f:
				vocab_set = pickle.load(f)
		else:
			vocab_set=VocabSet()
			vocab_set.build(vocab_ph,ph_vocab,ph_values)
			print("[SAVE]",flags.vocab)
			with open(flags.vocab, mode="wb") as f:
				pickle.dump(vocab_set, f)
		##
		vocab_var={}
		dtype = tf.float32
		initializer=tf.contrib.layers.xavier_initializer()
		for vocab_name,shapes in vocab_shape.items():
			values=vocab_set.get_values(vocab_name)
			if len(shapes)==1:
				shape=list(shapes)[0]
				if values is not None:
					#s=[len(values)]+list(shape)
					s=[max(values)+1]+list(shape)
				else:
					s=list(shape)
			else:
				shape=sorted(list(shapes),key=lambda x:len(x),reverse=True)[0]
				s=list(shape)
			var_name=vocab_name
			if embedding_generator and embedding_generator.is_dataset_embedding(vocab_name):
				print(">> dataset >>",var_name,":",s)
				pass
			elif(var_name[:14]=="tensor_onehot_"):
				print(">> onehot>>",var_name,":",s)
				m=re.match(r'tensor_onehot_([\d]*)_',var_name)
				if m:
					d=int(m.group(1))
					if len(s)==1:
						var=tf.one_hot(d,s[0])
						print(var,d,s[0])
					else:
						print("[ERROR]")
				else:
					print("[ERROR]")
				vocab_var[vocab_name]=var
			else:
				print(">> variable>>",var_name,":",s)
				var= tf.get_variable(var_name, shape=s, initializer=initializer, dtype=dtype)
				vocab_var[vocab_name]=var
		# converting PRISM switches to Tensorflow Variables
		tensor_embedding={}
		for sw_name,sw in sw_info.items():
			vocab_name=sw.vocab_name
			var_name=sw.var_name
			ph_list=sw.placeholder_names
			if len(ph_list)==0:
				if embedding_generator and embedding_generator.is_dataset_embedding(vocab_name):
					shape=list(list(sw.shape_set)[0])
					var=embedding_generator.get_embedding(vocab_name,shape)
					tensor_embedding[sw_name]=var
				else:
					var=vocab_var[vocab_name]
					tensor_embedding[sw_name]= var
			elif len(ph_list)==1:
				if embedding_generator and embedding_generator.is_dataset_embedding(vocab_name):
					shape=[batch_size]+list(list(sw.shape_set)[0])
					var=embedding_generator.get_embedding(vocab_name,shape)
					tensor_embedding[sw_name]=var
				else:
					var=vocab_var[vocab_name]
					ph=ph_var[ph_list[0]]
					tensor_embedding[sw_name]=tf.gather(var,ph)
					
		self.vocab_var=vocab_var
		self.vocab_set=vocab_set
		self.ph_var=ph_var
		self.ph_vocab=ph_vocab
		self.tensor_embedding=tensor_embedding
		self.sw_info=sw_info
		return tensor_embedding


