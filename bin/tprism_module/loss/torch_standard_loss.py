import numpy as np
import torch
import torch.nn.functional as F
from tprism_module.loss.base import BaseLoss


class Nll(BaseLoss):
    def __init__(self, parameters=None):
        pass

    # loss: goal x minibatch
    def call(self, graph, goal_inside, tensor_provider):
        loss = []
        output = []
        gamma = 1.00

        beta = 1.0e-4
        #reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_loss = beta * torch.mean(reg_losses)
        for rank_root in graph.root_list:
            goal_ids = [el.sorted_id for el in rank_root.roots]
            for sid in goal_ids:
                l1 = goal_inside[sid]["inside"]
                nll = -1.0 * torch.log(l1 + 1.0e-10)
                output.append(l1)
            ll = torch.mean(output, dim=0)# + reg_loss
            loss.append(ll)
        return loss, output


class Ce_pl2(BaseLoss):
    def __init__(self, parameters=None):
        pass

    # loss: goal x minibatch
    def call(self, graph, goal_inside, tensor_provider):
        loss = []
        output = []
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
                lo = F.nll_loss(F.softmax(l1), label_ph_var)
                loss.append(lo)
                output.append(l1)
        return loss, output


