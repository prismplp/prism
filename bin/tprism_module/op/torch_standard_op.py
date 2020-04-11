import numpy as np
import torch
from tprism_module.op.base import BaseOperator

class Reindex(BaseOperator):
    def __init__(self, parameters):
        index = parameters[0].strip("[]").split(",")
        self.out = index
        pass

    def call(self, x):
        return x

    def get_output_template(self, input_template):
        if len(input_template) > 0 and input_template[0] == "b":
            return ["b"] + self.out
        else:
            return self.out


class Sigmoid(BaseOperator):
    def __init__(self, parameters):
        pass

    def call(self, x):
        return torch.sigmoid(x)

    def get_output_template(self, input_template):
        return input_template


class Relu(BaseOperator):
    def __init__(self, parameters):
        pass

    def call(self, x):
        return torch.relu(x)

    def get_output_template(self, input_template):
        return input_template


class Softmax(BaseOperator):
    def __init__(self, parameters):
        pass

    def call(self, x):
        return torch.softmax(x,dim=-1)

    def get_output_template(self, input_template):
        return input_template

