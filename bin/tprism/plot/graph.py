import networkx as nx
from typing import List, Optional, Tuple, Set, Dict, Any
from tprism.torch_expl_graph import GoalInsideEntry

def build_and_or_graph(
    goal_inside: List[Optional[GoalInsideEntry]],
    and_label: str = "AND",
    or_label: str = "OR",
    leaf_label: str = "tensor",
) -> Tuple[
    List[Tuple[str, str]],
    List[Tuple[str, str]],
    Set[str],
    Set[str],
    Set[str],
    Dict[str, str],
]:
    or_and_edges: List[Tuple[str, str]] = []
    inside_edges: List[Tuple[str, str]] = []
    or_nodes: Set[str] = set()
    and_nodes: Set[str] = set()
    t_nodes: Set[str] = set()
    factor_mapping: Dict[str, str] = {}
    # for each all subgoals
    for i, goal in enumerate(goal_inside):
        if goal is None:
            continue
        # Only meaningful in dryrun mode (inside holds descriptors)
        if getattr(goal, "dryrun", False) is False:
            continue
        # make OR label
        or_id = str(goal.id) if goal.id is not None else str(i)
        or_key = or_id  # keep node id as label as before
        or_nodes.add(or_key)
        # subgoal without path (constant)
        inside_list = goal.inside if isinstance(goal.inside, list) else []
        if len(inside_list) == 0:
            continue
        # for each paths(OR) in a subgoal
        for j, path_desc in enumerate(inside_list):
            if not isinstance(path_desc, dict):
                continue
            # make AND label
            and_key = and_label + str(len(and_nodes))
            and_nodes.add(and_key)
            or_and_edges.append((and_key, or_key))
            # try to show einsum eq if present; otherwise fallback to name
            factor_mapping[and_key] = str(path_desc.get("einsum_eq", path_desc.get("name", "")))
            # for each node (AND) in a path
            path_nodes = list(path_desc.get("path", [])) + list(path_desc.get("path_scalar", []))
            for node in path_nodes:
                if isinstance(node, dict) and node.get("type") == "tensor_atom":
                    nm = node.get("name", "")
                    # keep previous convention of trimming 'tensor' prefix in label display
                    n = leaf_label + (nm[6:] if nm.startswith("tensor") else nm)
                    inside_edges.append((n, and_key))
                    t_nodes.add(n)
                elif isinstance(node, dict) and node.get("type") == "goal":
                    gid = str(node.get("id"))
                    inside_edges.append((gid, and_key))
                    or_nodes.add(gid)
                else:
                    pass  # dummy or unsupported
    return or_and_edges, inside_edges, or_nodes, and_nodes, t_nodes, factor_mapping

def plot_and_or_graph(
    goal_inside: List[Optional[GoalInsideEntry]],
    and_label: str = "AND",
    or_label: str = "OR",
    leaf_label: str = "tensor",
    pos: Any = None,
):
    or_and_edges, inside_edges, or_nodes, and_nodes, t_nodes, factor_mapping = build_and_or_graph(
        goal_inside, and_label, or_label, leaf_label
    )

    G = nx.DiGraph()
    G.add_edges_from(or_and_edges)
    G.add_edges_from(inside_edges)

    if pos is None:
        pos = nx.kamada_kawai_layout(G)
        #pos = nx.circular_layout(G)

    nx.draw_networkx_nodes(G, pos, nodelist=or_nodes, node_color="blue")
    nx.draw_networkx_nodes(G, pos, nodelist=t_nodes, node_color="green")
    nx.draw_networkx_nodes(G, pos, nodelist=and_nodes, node_color="red",node_shape="s")
    nx.draw_networkx_edges(G,pos)
    nx.draw_networkx_labels(G,pos)
    return G,pos

