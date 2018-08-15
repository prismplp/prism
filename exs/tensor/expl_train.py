import tensorflow as tf
import json
import expl_pb2
import re
from google.protobuf import json_format
from itertools import chain
import collections

import expl_graph

graph,options = expl_graph.load_explanation_graph("expl.json","flags.json")
tensor_embedding = expl_graph.build_variables(graph,options)
goal_inside = expl_graph.build_explanation_graph(graph,tensor_embedding)

loss=[]
for rank_root in graph.root_list:
	goal_ids=[el.sorted_id for el in rank_root.roots]
	l1=goal_inside[goal_ids[0]]["inside"]
	l2=goal_inside[goal_ids[1]]["inside"]
	loss.append(tf.nn.relu(l2-l1+0.1))
total_loss=tf.reduce_sum(loss)
#all_vars=tf.trainable_variables()
#dg=tf.gradients(total_loss,all_vars)
#print(dg)

optimizer = tf.train.GradientDescentOptimizer(0.001)
train = optimizer.minimize(total_loss)

init = tf.initialize_all_variables()

def optimize():
	with tf.Session() as session:
		session.run(init)
		print("starting at", "loss:", session.run(total_loss))
		for step in range(10):  
			session.run(train)
			print("step", step, "loss:", session.run(total_loss))
		

optimize()


