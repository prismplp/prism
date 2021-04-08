import numpy as np
import torch
import torch.nn.functional as F
from tprism_module.loss.base import BaseLoss
import sklearn.metrics


class Nll(BaseLoss):
    def __init__(self, parameters=None):
        pass

    # loss: goal x minibatch
    def call(self, graph, goal_inside, tensor_provider):
        loss = []
        output = []
        gamma = 1.00

        #beta = 1.0e-4
        #reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        #reg_loss = beta * torch.mean(reg_losses)
        for rank_root in graph.root_list:
            goal_ids = [el.sorted_id for el in rank_root.roots]
            o=[]
            for sid in goal_ids:
                l1 = goal_inside[sid]["inside"]
                nll = -1.0 * torch.log(l1 + 1.0e-10)
                o.append(l1)
            o=torch.stack(o)
            ll = torch.mean(o, dim=0)
            loss.append(ll)
            output.append(o)
        return loss, output, None

class Ce_pl2(BaseLoss):
    def __init__(self, parameters=None):
        pass

    # loss: goal x minibatch
    def call(self, graph, goal_inside, tensor_provider):
        loss = []
        output = []
        label = []
        gamma = 1.00
        label_ph = tensor_provider.ph_var["$placeholder2$"]
        label_ph_var =tensor_provider.get_embedding(label_ph)
        beta = 1.0e-4
        #reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        #reg_loss = beta * tf.reduce_mean(reg_losses)
        for rank_root in graph.root_list:
            goal_ids = [el.sorted_id for el in rank_root.roots]
            for sid in goal_ids:
                l1 = goal_inside[sid]["inside"]
                #print(F.softmax(l1,dim=1).shape, label_ph_var.shape)
                #lo = F.nll_loss(F.softmax(l1,dim=1), label_ph_var)
                lo = F.nll_loss(F.log_softmax(l1,dim=1), label_ph_var)
                loss.append(lo)
                output.append(l1)
                label.append(label_ph_var)
        return loss, output, label
    def metrics(self, output, label):
        pred=np.argmax(output,axis=1)
        acc=sklearn.metrics.accuracy_score(label,pred)
        return {"*accuracy":acc}

class Mse(BaseLoss):
    def __init__(self, parameters=None):
        pass

    # loss: goal x minibatch
    def call(self, graph, goal_inside, tensor_provider):
        loss = []
        output = []
        label = []
        gamma = 1.00
        label_ph = tensor_provider.ph_var["$placeholder1$"]
        label_ph_var =tensor_provider.get_embedding(label_ph)
        beta = 1.0e-4
        for rank_root in graph.root_list:
            goal_ids = [el.sorted_id for el in rank_root.roots]
            l1 = goal_inside[goal_ids[0]]["inside"]
            l2 = goal_inside[goal_ids[1]]["inside"]
            lo = (l1-l2)**2
            loss.append(lo.sum())
            output.append(l1)
            label.append(l2)
        return loss, output, label
    def metrics(self, output, label):
        mse=np.mean((label-output)**2,axis=0)
        mse=np.sum(mse)
        return {"*mse":mse}



class PreferencePair(BaseLoss):
    def __init__(self, parameters=None):
        pass

    # loss: goal x minibatch
    def call(self, graph, goal_inside, tensor_provider):
        loss = []
        output = []
        gamma = 1.00

        for rank_root in graph.root_list:
            goal_ids = [el.sorted_id for el in rank_root.roots]
            l1 = goal_inside[goal_ids[0]]["inside"]
            l2 = goal_inside[goal_ids[1]]["inside"]
            l = torch.nn.functional.relu(l2 - l1 + gamma)
            # l=tf.exp(l2-l1)
            # l=- 1.0/(tf.exp(l2-l1)+1)+reg_loss
            # l=tf.log(tf.exp(l2-l1)+1)+reg_loss
            # l = tf.nn.softplus(1 * l2)+tf.nn.softplus(-1 * l1) + reg_loss
            loss.append(torch.sum(l))
            output.append(torch.stack([l1, l2]))
        return loss, output, None



