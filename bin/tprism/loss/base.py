
from typing import List
#from tprism.expl_graph import ComputationalExplGraph, SwitchTensorProvider
#from tprism.torch_expl_graph import GoalInsideEntry

class BaseLoss:
    def __init__(self, parameters=None):
        pass

    def call(self, graph:'ComputationalExplGraph', goal_inside:List['GoalInsideEntry'], tensor_provider:'SwitchTensorProvider'):
        print("[WARN] loss is not implemened")
        output = []
        for rank_root in graph.root_list:
            goal_ids = [el.sorted_id for el in rank_root.roots]
            for sid in goal_ids:
                l1 = goal_inside[sid].inside
                output.append(l1)
        return None, output, None
    def metrics(self, output, label):
        return {}


