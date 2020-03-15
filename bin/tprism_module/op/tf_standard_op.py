import numpy as np
import tensorflow as tf

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
        return tf.nn.sigmoid(x)

    def get_output_template(self, input_template):
        return input_template


class Relu(BaseOperator):
    def __init__(self, parameters):
        pass

    def call(self, x):
        return tf.nn.relu(x)

    def get_output_template(self, input_template):
        return input_template


class Softmax(BaseOperator):
    def __init__(self, parameters):
        pass

    def call(self, x):
        return tf.nn.softmax(x)

    def get_output_template(self, input_template):
        return input_template


class Min1(BaseOperator):
    def __init__(self, parameters):
        pass

    def call(self, x):
        return tf.clip_by_value(x, 0.0, 1.0)

    def get_output_template(self, input_template):
        return input_template
