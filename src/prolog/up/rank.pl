
$pp_slice_push(List,[],N,Step):-
	length(List,L),L<N.
$pp_slice_push(List,[Sub|Dest],N,Step):-
	length(List,L),
	sublist(Sub,List,0,N),
	sublist(Rest,List,Step,L),
	$pp_slice_push(Rest,Dest,N,Step).
slice(List,Dest,N,Step):-$pp_slice_push(List,Dest,N,Step).
	
$pp_build_pair_list(GoalList,GoalPairList):-
	maplist(Gs,Y,
		slice(Gs,Y,2,1),GoalList,Ys),
	reducelist(X,Y,Z,append(X,Y,Z),Ys,[],GoalPairList).

rank(Goals,Gs):-rank(Goals,Gs,_).
rank(Goals,Gs,Probs):-
	get_prism_flag(clean_table,Backup),
	set_prism_flag(clean_table,off),
	(maplist(G,Y,
			(prob(G,P),Y=[P,G]),
			Goals,Ys)),
	sort(>,Ys,Rank),
	zip(Probs,Gs,Rank).

rank_learn(Goals) :-
	$pp_build_pair_list(Goals,GoalPairs),
	get_prism_flag(learn_mode,Mode0),
	$pp_conv_learn_mode(Mode0,Mode,VT),
    $pp_rank_learn_main(Mode,GoalPairs).
	%( VT = 0 -> $pp_rank_learn_main(Mode,Goals)
	%; VT = 1 -> $pp_rank_vlearn_main(Mode,Goals))
	

$pp_rank_learn_main(Mode,Goals) :- $pp_rank_learn_core(Mode,Goals,0).

$pp_rank_learn_core(Mode,GoalList,Debug) :-
	flatten(GoalList,Goals),
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
    $pp_observed_facts(GoalCountPairs,GidCountPairs,
                       0,Len,0,NGoals,-1,FailRootIndex),
	$pc_set_goal_rank(GoalList),
	%$pc_prism_goal_id_get(Goal,Gid)
    $pc_prism_prepare(GidCountPairs,Len,NGoals,FailRootIndex),
    cputime(StartEM),
    %%%
    $pp_rank_learn(Mode,Output,Debug),
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

$pp_rank_learn(ml,Output,Debug) :-
    $pc_rank_learn(Iterate,LogPost,LogLike,BIC,CS,ModeSmooth,Debug),
    Output = [Iterate,LogPost,LogLike,BIC,CS,ModeSmooth].

