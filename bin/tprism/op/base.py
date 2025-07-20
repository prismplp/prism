
class BaseOperator:
    def __init__(self, parameters):
        pass

    def call(self, x):
        print("[ERROR] not implemened")
        return None

    def get_output_template(self, input_template:List[str])-> List[str]:
        print("[ERROR] not implemened")
        return None

    def get_output_shape(self, input_shape):
        #print("[ERROR] not implemened")
        return input_shape
