import json
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple


GoalNode = Dict[str, Any]
Goals = List[Dict[str, Any]]
MappingIndex = MutableMapping[Any, List[str]]
MappingValues = MutableMapping[str, int]


def node2str(node: GoalNode, verb_index: bool = False, mapping_index: MappingIndex | None = None) -> str:
    """
    Render a goal node as a string.

    When verbose mode is enabled, include either the resolved indices or the sorted_id for debugging.
    """
    if verb_index:
        if mapping_index is not None:
            return f"{node['goal']['name']}({','.join(node['goal']['args'])})<{','.join(mapping_index[node['sorted_id']])}>"

        print(node)
        return f"{node['goal']['name']}({','.join(node['goal']['args'])})[id={node['sorted_id']}]"

    return f"{node['goal']['name']}({','.join(node['goal']['args'])})"


def sw2str(sw: Dict[str, Any], verb_index: bool = False) -> str:
    """Render a switch as a human-readable string."""
    if verb_index:
        return f"{sw['name']}<{','.join(sw['values'])}>"
    return sw["name"]


def print_expl(goals: Goals, verb_index: bool = False, mapping_index: MappingIndex | None = None) -> None:
    """Print an explanation tree in human-readable format."""
    for g in goals[::-1]:
        gg = g["node"]
        print(node2str(gg, verb_index, mapping_index), "<=>")
        for path in g["paths"]:
            nodes = [node2str(node, verb_index, mapping_index) for node in path["nodes"]]
            switches = [sw2str(sw, verb_index) for sw in path["tensor_switches"]]
            print("  ", "&".join(nodes + switches))


def out_template(temp_list: Sequence[Iterable[str]]) -> List[str]:
    """
    Flatten the first path's indices and return a deduplicated list.

    This is used to compute the new index list for a goal node.
    """
    path_temp = temp_list[0]
    arr: list[str] = []
    for element in path_temp:
        arr.extend(element)
    return list(dict.fromkeys(arr))


def remap_index(goals):
  # subgoal id=>インデックスリストの対応
  mapping_index={}
  for gindex,g in enumerate(goals):
    gg=g["node"]
    temp_list=[]
    for pindex,path in enumerate(g["paths"]):
      arr=[]
      for i,node in enumerate(path["nodes"]):
        new_index=mapping_index[node['sorted_id']]
        path["nodes"][i]["new_index"]=new_index
        arr.extend(new_index)
      for sw in path["tensor_switches"]:
        new_index=sw["values"]
        sw["new_index"]=new_index
        arr.extend(new_index)
      temp_list.append(arr)
    gid=gg['sorted_id']
    mapping_index[gid]=out_template(temp_list)
    gg["new_index"]=mapping_index[gid]
  return mapping_index




if __name__ == "__main__":
    with open("./data.expl.json", "r", encoding="utf-8") as expl_file:
        obj = json.load(expl_file)
    goals = obj["goals"]
    print_expl(goals)
    mapping_index = remap_index(goals)
    print_expl(goals, mapping_index=mapping_index)
