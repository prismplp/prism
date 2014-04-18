% Transform the prism explanation graph into a hypergraph
$pp_trans_graph(ExpGraph,HGraph,NodesId,ExpTableId) :- !,
  length(ExpGraph,L),
  new_bigarray(NodesAcc,L),
  new_hashtable(NodesId,L),
  new_bigarray(ExpTableId,L),
  $pp_create_node_array(ExpGraph,1,NodesId,ExpTableId),
  $pp_transform_graph(ExpGraph,HGraph,NodesAcc,NodesId).

$pp_create_node_array([],_,_,_).
$pp_create_node_array([node(H,_,_)|ExpGraph],I,NodesId,ExpTableId) :-
  hashtable_register(NodesId,H,I),
  bigarray_put(ExpTableId,I,H),
  I1 is I + 1,
  $pp_create_node_array(ExpGraph,I1,NodesId,ExpTableId).

$pp_transform_graph([],HGraph,HGraph,_).
% Traverse node H <=> B_1 + ... + B_n
% where P(H) = P(B_1) + ... + P(B_n)
$pp_transform_graph([node(H,Paths,_)|ExpGraph],HGraph,HGraphAcc,NodesId) :-
% Traverse edges B_i = C_1 ^ ... ^ C_m ^ msw_1 ^ ... ^ msw_n
% where P(B_i) = P(C_1)*...*P(C_m)*P(msw_1)*...*P(msw_n)
  $pp_transform_edges(NodesId,Paths,Edges),
  hashtable_get(NodesId,H,I),
  bigarray_put(HGraphAcc,I,Edges),
  $pp_transform_graph(ExpGraph,HGraph,HGraphAcc,NodesId).

% Transform the paths into multiedges
$pp_transform_edges(NodesId,Paths,Edges) :-
  $pp_transform_edges(NodesId,Paths,Edges,[]).
$pp_transform_edges(_,[],Edges,Edges).
$pp_transform_edges(NodesId,[path(Nodes,Switches,_)|Paths],Edges,EdgesAcc) :-
  % Get children of edge
  Children @= [C : N in Nodes,[C,P,G],(N = gnode(G,P),hashtable_get(NodesId,G,C))],
  % Accumulate probability from switches
  % AccProb = P(msw_1)*...*P(msw_n)
  ($pp_in_log_scale ->
    foreach(snode(_,Prob) in Switches,ac(AccProb,0),(AccProb^1 is Prob + AccProb^0))
    ;foreach(snode(_,Prob) in Switches,ac(AccProb,1),AccProb^1 is Prob * AccProb^0)
  ),
  $pp_transform_edges(NodesId,Paths,Edges,[edge(AccProb,Children)|EdgesAcc]).

