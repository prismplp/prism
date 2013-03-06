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
  $prprobfi(Goal,Expls,1).
prprobefi(Goal,Expls) :-
  $prprobfi(Goal,Expls,0).

find_scc(Goal,Components) :-
  % Testing goal
  probefi(Goal,ExpGraph),
  % Transforming graph
  $pp_trans_graph(ExpGraph,HGraph,_,_),
  % Finding SCC
  $pp_find_scc(HGraph,Components,_).

$prprobfi(Goal,Expls,Decode) :-
  % Testing goal
  probefi(Goal,ExpGraph),
  % Transforming graph
  $pp_trans_graph(ExpGraph,HGraph,_,_),
  % Finding SCC
  $pp_find_scc(HGraph,Components,CompTable),
  % Solving graph
  $pp_solve_graph(HGraph,Components,CompTable,ProbTable),
  bigarray_length(ProbTable,Size),
  % Call Gabow algorithm on each node
  foreach(I in 1..Size,
    [Index,Index2,Temp],
    (
      Index is Size - I+1,
      Index2 is I-1,
      bigarray_get(ProbTable,Index,Temp),
      $pc_set_gnode_inside(Index2,Temp)
    )
  ),!,
  $pc_import_sorted_graph_size(ESize),
  $pp_build_expls(ESize,Decode,1,Goal,Expls).


