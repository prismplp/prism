import networkx as nx

def build_and_or_graph(goal_inside, and_label="AND", or_label="OR", leaf_label="tensor"):
    or_and_edges=[]
    inside_edges=[]
    or_nodes=set()
    and_nodes=set()
    t_nodes=set()
    # for each all subgoals
    for i,goal in enumerate(goal_inside):
        ## make OR label
        or_key=str(goal["id"])
        or_nodes.add(or_key)
        # subgoal without path (constant)
        if 0==len(goal["inside"]):
            pass
        # for each paths(OR) in a subgoal
        for j in range(len(goal["inside"])):
            # make AND label
            and_key=and_label + str(len(and_nodes))
            and_nodes.add(and_key)
            or_and_edges.append((and_key,or_key))
            # for each node (AND) in a path
            for node in goal["inside"][j]["path"]+goal["inside"][j]["path_scalar"]:
                if "type" in node and node["type"]=="tensor_atom":
                    n=leaf_label+node["name"][6:]
                    inside_edges.append((n,and_key))
                    t_nodes.add(n)
                elif "type" in node and node["type"]=="goal":
                    gid=str(node["id"])
                    inside_edges.append((gid,and_key))
                    or_nodes.add(str(node["id"]))
                else:
                    pass #dummy
    return or_and_edges,inside_edges,or_nodes,and_nodes,t_nodes

def plot_and_or_graph(goal_inside, and_label="AND", or_label="OR", leaf_label="tensor", pos=None):
    or_and_edges,inside_edges,or_nodes,and_nodes,t_nodes = build_and_or_graph(
            goal_inside, and_label, or_label, leaf_label)

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

