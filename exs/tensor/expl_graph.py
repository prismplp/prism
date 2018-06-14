import tensorflow as tf
import json
import expl_pb2
import re
from google.protobuf import json_format

from itertools import chain
import collections

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
	return re.sub(r'[\)\(]+', "_", name)

def get_unique_list(seq):
	seen = []
	return [x for x in seq if x not in seen and not seen.append(x)]

#[['i'], ['i','l', 'j'], ['j','k']] => ['l','k']
def compute_output_template(template):
	counter=collections.Counter(chain.from_iterable(template))
	out_template=[k for k,cnt in counter.items() if cnt==1]
	print(out_template)
	return out_template

def build_explanation_graph(graph,tensor_embedding):
	# converting explanation graph to computational graph
	goal_inside=[None]*len(graph.goals)
	for i in range(len(graph.goals)):
		g=graph.goals[i]
		path_inside=[]
		path_template=[]
		for path in g.paths:
			## build template and inside for switches in the path
			sw_template=[]
			sw_inside=[]
			for sw in path.sws:
				sw_template.append(list(sw.value.list))
				sw_inside.append(tensor_embedding[sw.name])
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
			lhs=",".join(map(lambda x: "".join(x) ,template))
			rhs="".join(out_template)
			einsum_eq=lhs+"->"+rhs
			print(einsum_eq)
			print(inside)
			temp_inside=tf.einsum(einsum_eq,*inside)
			path_inside.append(temp_inside)
			path_template.append(out_template)
			##
		##
		path_template_list=get_unique_list(path_template)
		if len(path_template_list)!=1:
			print("[ERROR]")
		goal_inside[i]={
			"template":path_template_list[0],
			"inside":tf.reduce_sum(tf.stack(path_inside),axis=0)}
	return goal_inside

def build_variables(graph,options):
	index_range={el.index:el.range for el in options.index_range}
	tensors={}
	for g in graph.goals:
		for path in g.paths:
			for sw in path.sws:
				if sw.name not in tensors:
					tensors[sw.name]=set([])
				tensors[sw.name].add(tuple([el for el in sw.value.list]))
	# converting PRISM switches to Tensorflow Variables
	tensor_embedding={}
	dtype = tf.float32
	initializer=tf.contrib.layers.xavier_initializer()
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


