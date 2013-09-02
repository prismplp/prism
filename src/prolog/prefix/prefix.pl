prprobg(Goal):-
    (get_prism_flag(log_scale,on)->Text='Log-probability';Text='Probability'),
    prprobg(Goal,L),
    foreach(S in L,[G,P],
    ([P,G]=S,format("~w of ~w is: ~15f~n",[Text,G,P]))).

prprobg(Goal,Probs):-
    vars_set(Goal,Vars),
    probf(Goal,N),
    N=[node(_,Z)|_],
    foreach(S in Z,ac(Ps,[]),[P,G|Vars],
    (S=path([G],[]),Goal=G->prprob(G,P),Ps^1=[[P,G]|Ps^0];true)),
	sort(>,Ps,Probs).
probg(Goal,Probs):-
    vars_set(Goal,Vars),
    probf(Goal,N),
    N=[node(_,Z)|_],
    foreach(S in Z,ac(Ps,[]),[P,G|Vars],
    (S=path([G],[]),Goal=G->prob(G,P),Ps^1=[[P,G]|Ps^0];true)),
	sort(>,Ps,Probs).


prprob(Goal) :-
  prprob(Goal,P),
  (get_prism_flag(log_scale,on)->Text='Log-probability';Text='Probability'),
  format("~w of ~w is: ~15f~n",[Text,Goal,P]).

prprob(Goal,Prob) :-
  % Testing goal
  probefi(Goal,ExpGraph),
  % Transforming graph
  $pp_trans_graph(ExpGraph,HGraph,_,_),
  % Finding SCC
  $pp_find_scc(HGraph,Components,CompTable),
  % Solving graph
  $pp_solve_graph(HGraph,Components,CompTable,ProbTable),
  bigarray_get(ProbTable,1,Prob),!.

prprobfi(Goal):-
  prprobfi(Goal,Expls),print_graph(Expls,[lr('<=>')]).
prprobefi(Goal):-
  prprobefi(Goal,Expls),print_graph(Expls,[lr('<=>')]).

prprobfi(Goal,Expls) :-
  $prprobfi(Goal,_,1,Expls).
prprobefi(Goal,Expls) :-
  $prprobfi(Goal,_,0,Expls).

find_scc(Goal,Components,CompT) :-
  % Testing goal
  probefi(Goal,ExpGraph),
  % Transforming graph
  $pp_trans_graph(ExpGraph,HGraph,_,_),
  % Finding SCC
  $pp_find_scc(HGraph,Components,CompT).

replace_prob(Id,Mapping,P):-
  !,(X=(Id,P),member(X,Mapping)),!.

$prprobfi(Goal,OrgExpls,Decode,NewExpls) :-
  % Testing goal
  (Decode=0->probefi(Goal,OrgExpls);probfi(Goal,OrgExpls)),
  % Transforming graph
  $pp_trans_graph(OrgExpls,HGraph,_,_),
  % Finding SCC
  $pp_find_scc(HGraph,Components,CompTable),
  % Solving graph
  $pp_solve_graph(HGraph,Components,CompTable,ProbTable),
  bigarray_length(ProbTable,Size),
  % Creating mapping from ProbTableIndex to NodeID
  %new_bigarray(Mapping,Size),
  %foreach((E,I) in (OrgExpls,1..Size),[Id,T1,T2],
  %  (E=node(Id,T1,T2),bigarray_put(Mapping,I,Id))),
  % Creating mapping ProbTableIndex and NodeID
  Src @= [Index : Index in 1..Size],
  maplist(I,Ex,Pair,(
      bigarray_get(ProbTable,I,Temp),
      Ex=node(Id,_,_),
      %bigarray_get(Mapping,I,Index2),
      Pair=(Id,Temp)
    ),Src,OrgExpls,IMapping),!,
  $pc_import_sorted_graph_size(ESize),
  % Replacing probabilities
  maplist(E,NewExpl,(E=node(Id,Paths,P),
  maplist(Path,NewPath,(Path=path(GNodes,SNodes,PP),
  maplist(GNode,NewGNode,(GNode=gnode(GID,GP),replace_prob(GID,IMapping,NewGP),NewGNode=gnode(GID,NewGP)),GNodes,NewGNodes)
  ,NewPath=path(NewGNodes,SNodes,PP)),Paths,NewPaths)
  ,replace_prob(Id,IMapping,NewP),NewExpl = node(Id, NewPaths ,NewP) ),OrgExpls,NewExpls),
  % TODO:Re-calculate path-probabilities
  %
  $pp_garbage_collect.


