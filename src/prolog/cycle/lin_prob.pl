$prprobg(Goal):-
    (get_prism_flag(log_scale,on)->Text='Log-probability';Text='Probability'),
    prprobg(Goal,L),
    foreach(S in L,[G,P],
    ([P,G]=S,format("~w of ~w is: ~15f~n",[Text,G,P]))).

$prprobg(Goal,Probs):-
    vars_set(Goal,Vars),
    probf(Goal,N),
    N=[node(_,Z)|_],
    foreach(S in Z,ac(Ps,[]),[P,G|Vars],
    (S=path([G],[]),Goal=G->prprob(G,P),Ps^1=[[P,G]|Ps^0];true)),
	sort(>,Ps,Probs).
$probg(Goal,Probs):-
    vars_set(Goal,Vars),
    probf(Goal,N),
    N=[node(_,Z)|_],
    foreach(S in Z,ac(Ps,[]),[P,G|Vars],
    (S=path([G],[]),Goal=G->prob(G,P),Ps^1=[[P,G]|Ps^0];true)),
	sort(>,Ps,Probs).


lin_prob(Goal) :-
  lin_prob(Goal,P),
  (get_prism_flag(log_scale,on)->Text='Log-probability';Text='Probability'),
  format("~w of ~w is: ~15f~n",[Text,Goal,P]).

find_scc(Goal,Components,CompT) :-
  % Testing goal
  probefi(Goal,ExpGraph),
  % Transforming graph
  $pp_trans_graph(ExpGraph,HGraph,_,_),
  % Finding SCC
  $pp_find_scc(HGraph,Components,CompT).

lin_prob(Goal,Prob) :-
  % Testing goal
  probefi(Goal,ExpGraph),
  % Transforming graph
  $pp_trans_graph(ExpGraph,HGraph,_,_),
  % Finding SCC
  $pp_find_scc(HGraph,Components,CompTable),
  % Solving graph
  $pp_solve_graph(HGraph,Components,CompTable,ProbTable),
  bigarray_get(ProbTable,1,Prob),!.



lin_probfi(Goal):-
  lin_probfi(Goal,Expls),print_graph(Expls,[lr('<=>')]).
lin_probefi(Goal):-
  lin_probefi(Goal,Expls),print_graph(Expls,[lr('<=>')]).

lin_probfi(Goal,Expls) :-
  $pp_cyc_probfi(Goal,_,1,Expls).
lin_probefi(Goal,Expls) :-
  $pp_cyc_probfi(Goal,_,0,Expls).

$pp_cyc_replace_prob(Id,Mapping,P):-
  !,(X=(Id,P),member(X,Mapping)),!.

$pp_cyc_probfi(Goal,OrgExpls,Decode,NewExpls2) :-
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
  maplist(GNode,NewGNode,(GNode=gnode(GID,GP),$pp_cyc_replace_prob(GID,IMapping,NewGP),NewGNode=gnode(GID,NewGP)),GNodes,NewGNodes)
  ,NewPath=path(NewGNodes,SNodes,PP)),Paths,NewPaths)
  ,$pp_cyc_replace_prob(Id,IMapping,NewP),NewExpl = node(Id, NewPaths ,NewP) ),OrgExpls,NewExpls),
  % TODO:Re-calculate path-probabilities
  get_prism_flag(log_scale,LogScale),
  %( LogScale == on -> Vi is Vg + Vs ; Vi is Vg * Vs),
  maplist(E,NewExpl,(E=node(Id,Paths,P),
  maplist(Path,NewPath,(Path=path(GNodes,SNodes,PP),
  maplist(GNode,GP,(GNode=gnode(GID,GP)),GNodes,GNodeProbs),
  maplist(SNode,SP,(SNode=snode(SID,SP)),SNodes,SNodeProbs),
  ( LogScale == on -> (
    reducelist(Y0,X,Y1,(Y1 is Y0+X),GNodeProbs,0.0,TempP),
    reducelist(Y0,X,Y1,(Y1 is Y0+X),SNodeProbs,TempP,PathP)
  );(
    reducelist(Y0,X,Y1,(Y1 is Y0*X),GNodeProbs,1.0,TempP),
    reducelist(Y0,X,Y1,(Y1 is Y0*X),SNodeProbs,TempP,PathP)
  )),
  NewPath=path(GNodes,SNodes,PathP)),Paths,NewPaths)
  ,NewExpl = node(Id, NewPaths ,P) ),NewExpls,NewExpls2),
  %
  $pp_garbage_collect.


$cyc_learn(Goals) :-
$pp_cyc_learn_core(ml,Goals).

$pp_cyc_learn_core(Mode,Goals) :-
    $pp_learn_check_goals(Goals),
    $pp_learn_message(MsgS,MsgE,MsgT,MsgM),
    $pc_set_em_message(MsgE),
    cputime(Start),
    $pp_clean_learn_info,
    $pp_learn_reset_hparams(Mode),
    $pp_trans_goals(Goals,GoalCountPairs,AllGoals),!,
    global_set($pg_observed_facts,GoalCountPairs),
    cputime(StartExpl),
    global_set($pg_num_goals,0),
    $pp_find_explanations(AllGoals),!,
    $pp_print_num_goals(MsgS),
    cputime(EndExpl),
    statistics(table,[TableSpace,_]),
    $pp_format_if(MsgM,"Exporting switch information to the EM routine ... "),
    flush_output,
    $pp_export_sw_info,
    $pp_format_if(MsgM,"done~n"),
    format("~w\n",[GoalCountPairs,GidCountPairs,0,Len,0,NGoals,-1,FailRootIndex]),
    $pp_observed_facts(GoalCountPairs,GidCountPairs,
                       0,Len,0,NGoals,-1,FailRootIndex),
    format("a"),
    $pc_prism_prepare(GidCountPairs,Len,NGoals,FailRootIndex),
    format("b"),
    cputime(StartEM),
    %%%$pp_em(Mode,Output),
    $pp_cyc_em(Mode,Output),
    %%%
    cputime(EndEM),
    $pc_import_occ_switches(NewSws,NSwitches,NSwVals),
    $pp_decode_update_switches(Mode,NewSws),
    $pc_import_graph_stats(NSubgraphs,NGoalNodes,NSwNodes,AvgShared),
    cputime(End),
    $pp_assert_graph_stats(NSubgraphs,NGoalNodes,NSwNodes,AvgShared),
    $pp_assert_learn_stats(Mode,Output,NSwitches,NSwVals,TableSpace,
                           Start,End,StartExpl,EndExpl,StartEM,EndEM,1000),
    $pp_print_learn_stats_message(MsgT),
    $pp_print_learn_end_message(MsgM,Mode),!.

$pp_cyc_em(ml,Output) :-
    format("c"),
    $pc_cyc_em(Iterate,LogPost,LogLike,BIC,CS,ModeSmooth),
    format("d"),
    Output = [Iterate,LogPost,LogLike,BIC,CS,ModeSmooth].

