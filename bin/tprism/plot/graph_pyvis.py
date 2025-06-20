from IPython.core.display import display, HTML
from pyvis.network import Network

from tprism.plot.graph import build_and_or_graph

def plot_and_or_graph(goal_inside, and_label="AND", or_label="OR", leaf_label="tensor"):
    or_and_edges,inside_edges,or_nodes,and_nodes,t_nodes,factor_mapping = build_and_or_graph(
            goal_inside, and_label, or_label, leaf_label)
    ##
    nt = Network('320px', '480px', notebook=True,cdn_resources="remote",directed=True)

    for node in or_nodes:
      nt.add_node(node, node, title=node, color='red', font='42px arial black')
    for node in t_nodes:
      nt.add_node(node, node, title=node, color='green', font='42px arial black')
    for node in and_nodes:
      nt.add_node(node, " E ", title=factor_mapping[node], color='blue',shape="box", font='42px arial white')

    for src, dst in inside_edges:
      nt.add_edge(src, dst,weight=1.0)
    for src, dst in or_and_edges:
      if dst in t_nodes:
        nt.add_edge(src, dst ,weight=0.1)
      else:
        nt.add_edge(src, dst ,weight=1.0)
    nt.barnes_hut(central_gravity=0,spring_strength=0.5,spring_length=5)
    nt.show_buttons(filter_=['physics'])
    nt.show('nx.html')
    display(HTML('nx.html'))

