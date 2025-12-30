
from typing import Any, List, Optional, Tuple
import tprism.expl_pb2 as expl_pb2
from torch import Tensor
import torch
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from tprism.expl_graph import SwitchTensorProvider
    from tprism.torch_expl_graph import GoalInsideEntry

class BaseLoss:
    def __init__(self, parameters=None):
        pass

    def call(self, graph:'expl_pb2.ExplGraph', goal_inside:List[Optional['GoalInsideEntry']], tensor_provider:'SwitchTensorProvider')-> Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
        print("[WARN] loss is not implemened")
        output = []
        for rank_root in graph.root_list:
            goal_ids = [el.sorted_id for el in rank_root.roots]
            for sid in goal_ids:
                gnode=goal_inside[sid]
                if gnode is None:
                    raise RuntimeError("goal_inside[%d] is None" % (sid,))
                l1 = gnode.inside
                output.append(l1)
        # loss, output, label
        return None, torch.stack(output), None
    def metrics(self, output, label):
        return {}


