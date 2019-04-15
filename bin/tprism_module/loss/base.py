import numpy as np
import tensorflow as tf
class BaseLoss:
	def __init__(self,parameters=None):
		pass
	def call(self,output):
		print("[ERROR] not implemened")
		return None

class PreferencePair(BaseLoss):
	def __init__(self,parameters=None):
		pass
	# loss: goal x minibatch
	def call(self,graph,goal_inside,tensor_provider):
		loss=[]
		output=[]
		gamma=1.00
		
		beta=5.0e-6
		reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
		reg_loss=beta*tf.reduce_mean(reg_losses)
		for rank_root in graph.root_list:
			goal_ids=[el.sorted_id for el in rank_root.roots]
			l1=goal_inside[goal_ids[0]]["inside"]
			l2=goal_inside[goal_ids[1]]["inside"]
			l=tf.nn.relu(l2-l1+gamma)+reg_loss
			#l=tf.exp(l2-l1)+reg_loss
			#l=- 1.0/(tf.exp(l2-l1)+1)+reg_loss
			#l=tf.log(tf.exp(l2-l1)+1)+reg_loss
			#l = tf.nn.softplus(1 * l2)+tf.nn.softplus(-1 * l1) + reg_loss
			loss.append(l)
			output.append([l1,l2])
		return loss,output


class SigmoidNll(BaseLoss):
	def __init__(self,parameters=None):
		pass
	# loss: goal x minibatch
	def call(self,graph,goal_inside,tensor_provider):
		loss=[]
		output=[]
		gamma=1.00
		
		beta=1.0e-4
		reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
		reg_loss=beta*tf.reduce_mean(reg_losses)
		for rank_root in graph.root_list:
			goal_ids=[el.sorted_id for el in rank_root.roots]
			for sid in goal_ids:
				l1=tf.nn.sigmoid(goal_inside[sid]["inside"])
				nll=-1.0*tf.log(l1+1.0e-10)
				output.append(nll)
			ll=tf.reduce_mean(output,axis=0)
			loss.append(ll)
		return loss,output



class Nll(BaseLoss):
	def __init__(self,parameters=None):
		pass
	# loss: goal x minibatch
	def call(self,graph,goal_inside,tensor_provider):
		loss=[]
		output=[]
		gamma=1.00
		
		beta=1.0
		reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
		print(reg_losses)
		reg_loss=beta*tf.reduce_mean(reg_losses)
		for rank_root in graph.root_list:
			goal_ids=[el.sorted_id for el in rank_root.roots]
			for sid in goal_ids:
				l1=goal_inside[sid]["inside"]
				nll=-1.0*tf.log(l1+1.0e-10)
				output.append(l1)
			ll=tf.reduce_mean(output,axis=0)+reg_loss
			loss.append(ll)
		return loss,output

class Ce(BaseLoss):
	def __init__(self,parameters=None):
		pass
	# loss: goal x minibatch
	def call(self,graph,goal_inside,tensor_provider):
		loss=[]
		output=[]
		gamma=1.00
		#label_ph_var=tensor_provider.ph_var["$placeholder2$"]
		beta=1.0e-4
		reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
		reg_loss=beta*tf.reduce_mean(reg_losses)
		for rank_root in graph.root_list:
			goal_ids=[el.sorted_id for el in rank_root.roots]
			for sid in goal_ids:
				#print(graph.expl[sid])
				label=int(graph.goals[sid].node.goal.args[0])
				l1=goal_inside[sid]["inside"]
				lo=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label,logits=l1)
				lo=lo+reg_loss
				loss.append(lo)
				output.append(l1)
		return loss,output

class Ce1(BaseLoss):
	def __init__(self,parameters=None):
		pass
	# loss: goal x minibatch
	def call(self,graph,goal_inside,tensor_provider):
		loss=[]
		output=[]
		gamma=1.00
		label_ph_var=tensor_provider.ph_var["$placeholder1$"]
		print(label_ph_var.shape)
		
		beta=1.0e-5
		reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
		reg_loss=beta*tf.reduce_mean(reg_losses)
		for rank_root in graph.root_list:
			goal_ids=[el.sorted_id for el in rank_root.roots]
			for sid in goal_ids:
				l1=goal_inside[sid]["inside"]
				l1=tf.clip_by_value(l1,1e-10,1.0-1e-10)
				#lo=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_ph_var,logits=l1)
				#numerator=tf.gather(l1,label_ph_var,axis=1)
				print(">????>>",l1.shape)
				print(">????>>",label_ph_var.shape)
				print(label_ph_var.shape)
				v=tf.one_hot(label_ph_var,depth=l1.shape[1])
				numerator=tf.reduce_sum(l1*v,axis=1)
				print(">????>>",numerator.shape)
				denom=tf.reduce_logsumexp(l1,axis=1)
				loss.append(-numerator+denom)
				output.append(l1)
		return loss,output


class Ce2(BaseLoss):
	def __init__(self,parameters=None):
		pass
	# loss: goal x minibatch
	def call(self,graph,goal_inside,tensor_provider):
		loss=[]
		output=[]
		gamma=1.00
		label_ph_var=tensor_provider.ph_var["$placeholder3$"]
		beta=1.0e-5
		num_neg_sample=100
		reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
		reg_loss=beta*tf.reduce_mean(reg_losses)
		for rank_root in graph.root_list:
			goal_ids=[el.sorted_id for el in rank_root.roots]
			for sid in goal_ids:
				l1=goal_inside[sid]["inside"]
				#l1=tf.clip_by_value(l1,1e-10,1.0-1e-10)
				batch_size=int(l1.shape[0])
				dim=int(l1.shape[1])
				label_one_hot=tf.one_hot(label_ph_var,depth=dim)
				pos_numerator=tf.reduce_sum(l1*label_one_hot,axis=1)
				pos_numerator=tf.reshape(pos_numerator,(batch_size,1))
				#
				neg_l1=tf.reshape(l1,(batch_size,1,dim))
				neg_sample=tf.random_uniform(shape=(batch_size,num_neg_sample),minval=0,maxval=dim,dtype=tf.int32)
				neg_one_hot=tf.one_hot(neg_sample,depth=dim)
				neg_numerator=tf.reduce_sum(neg_l1*neg_one_hot,axis=2)
				#
				# log(exp(neg_num)/sumexp(denom))-log(exp(pos_num)/sumexp(denom))
				#=neg_num - logsumexp(denom)-pos_num + logsumexp(denom)
				#=neg_num -pos_num 
				l=tf.nn.relu(neg_numerator-pos_numerator+gamma)
				l=tf.reduce_sum(l,axis=1)+reg_loss
				loss.append(l)
				output.append(l1)
		return loss,output

class Ce3(BaseLoss):
	def __init__(self,parameters=None):
		pass
	# loss: goal x minibatch
	def call(self,graph,goal_inside,tensor_provider):
		loss=[]
		output=[]
		gamma=1.00
		label_ph_var=tensor_provider.ph_var["$placeholder3$"]
		beta=1.0e-5
		num_neg_sample=100
		reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
		reg_loss=beta*tf.reduce_mean(reg_losses)
		for rank_root in graph.root_list:
			goal_ids=[el.sorted_id for el in rank_root.roots]
			for sid in goal_ids:
				l1=goal_inside[sid]["inside"]
				#l1=tf.clip_by_value(l1,1e-10,1.0-1e-10)
				batch_size=int(l1.shape[0])
				dim=int(l1.shape[1])
				label_one_hot=tf.one_hot(label_ph_var,depth=dim)
				pos_numerator=tf.reduce_sum(l1*label_one_hot,axis=1)
				#
				neg_l1=tf.reshape(l1,(batch_size,1,dim))
				neg_sample=tf.random_uniform(shape=(batch_size,num_neg_sample),minval=0,maxval=dim,dtype=tf.int32)
				neg_one_hot=tf.one_hot(neg_sample,depth=dim)
				neg_numerator=tf.reduce_sum(neg_l1*neg_one_hot,axis=2)
				neg_numerator=tf.reduce_logsumexp(neg_numerator,axis=1)
				# -log(exp(pos_num)/sumexp(neg_num))
				#=-pos_num + logsumexp(neg_num)
				#l=tf.nn.relu(neg_numerator-pos_numerator+gamma)
				l=neg_numerator-pos_numerator
				l=l+reg_loss
				loss.append(l)
				output.append(l1)
		return loss,output


class Ce4(BaseLoss):
	def __init__(self,parameters=None):
		pass
	# loss: goal x minibatch
	def call(self,graph,goal_inside,tensor_provider):
		loss=[]
		output=[]
		gamma=1.00
		label_ph_var=tensor_provider.ph_var["$placeholder3$"]
		beta=1.0e-5
		num_neg_sample=100
		reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
		reg_loss=beta*tf.reduce_mean(reg_losses)
		for rank_root in graph.root_list:
			goal_ids=[el.sorted_id for el in rank_root.roots]
			for sid in goal_ids:
				l1=goal_inside[sid]["inside"]
				#l1=tf.clip_by_value(l1,1e-10,1.0-1e-10)
				batch_size=int(l1.shape[0])
				dim=int(l1.shape[1])
				label_one_hot=tf.one_hot(label_ph_var,depth=dim)
				pos_numerator=tf.reduce_sum(l1*label_one_hot,axis=1)
				pos_numerator=tf.nn.softplus(pos_numerator)
				#
				neg_l1=tf.reshape(l1,(batch_size,1,dim))
				neg_sample=tf.random_uniform(shape=(batch_size,num_neg_sample),minval=0,maxval=dim,dtype=tf.int32)
				neg_one_hot=tf.one_hot(neg_sample,depth=dim)
				neg_numerator=tf.reduce_sum(neg_l1*neg_one_hot,axis=2)
				neg_numerator=tf.nn.softplus(neg_numerator)
				neg_numerator=tf.reduce_logsumexp(neg_numerator,axis=1)
				#
				# -log(exp(f(pos_num))/sumexp(f(neg_num)))
				#=-f(pos_num) + logsumexp(f(neg_num))
				l=tf.nn.relu(neg_numerator-pos_numerator+gamma)
				l=l+reg_loss
				loss.append(l)
				output.append(l1)
		return loss,output


class Ce5(BaseLoss):
	def __init__(self,parameters=None):
		pass
	# loss: goal x minibatch
	def call(self,graph,goal_inside,tensor_provider):
		loss=[]
		output=[]
		gamma=1.00
		label_ph_var=tensor_provider.ph_var["$placeholder3$"]
		beta=1.0e-5
		num_neg_sample=100
		reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
		reg_loss=beta*tf.reduce_mean(reg_losses)
		for rank_root in graph.root_list:
			goal_ids=[el.sorted_id for el in rank_root.roots]
			for sid in goal_ids:
				l1=goal_inside[sid]["inside"]
				#l1=tf.clip_by_value(l1,1e-10,1.0-1e-10)
				batch_size=int(l1.shape[0])
				dim=int(l1.shape[1])
				label_one_hot=tf.one_hot(label_ph_var,depth=dim)
				pos_numerator=tf.reduce_sum(l1*label_one_hot,axis=1)
				pos_numerator=tf.reshape(pos_numerator,(batch_size,1))
				#
				neg_l1=tf.reshape(l1,(batch_size,1,dim))
				neg_sample=tf.random_uniform(shape=(batch_size,num_neg_sample),minval=0,maxval=dim,dtype=tf.int32)
				neg_one_hot=tf.one_hot(neg_sample,depth=dim)
				neg_numerator=tf.reduce_sum(neg_l1*neg_one_hot,axis=2)
				#
				# log(exp(neg_num)/sumexp(denom))-log(exp(pos_num)/sumexp(denom))
				#=neg_num - logsumexp(denom)-pos_num + logsumexp(denom)
				#=neg_num -pos_num 
				l=tf.nn.relu(neg_numerator-pos_numerator+gamma)
				l=tf.reduce_sum(l,axis=1)+reg_loss
				loss.append(l)
				output.append(l1)
		return loss,output


