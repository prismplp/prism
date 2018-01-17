
$pp_trans_phase_vec(Prog0,Prog_vec,Info):-
	maplist(X,Y,(
	%format("~w\n",X),
		X=pred(A0,A1,A2,A3,A4,Clauses)->($pp_vec_parse_clauses(Clauses,C),Y=pred(A0,A1,A2,A3,A4,C));Y=X
	),Prog0,Prog_vec).

$pp_vec_parse_clauses(Clauses,NewClauses):-
	maplist(C,NC,(format(">>~w\n",C),
	$pp_vec_get_msws(C,NC),format(">>>>~w\n",NC)
		),Clauses,NewClauses).

$pp_vec_get_msws(Clause,MswClause):-
	Clause=(H:-Body) -> (
		Clause=(H:-Body),
		and_to_list(Body,LBody),
		$pp_vec_msw_filter(LBody,MswLBody),
		list_to_and(MswLBody,MswBody),
		MswClause=(H:-(MswBody)))
	;MswClause=Clause.
	
$pp_vec_msw_filter([],[]).
$pp_vec_msw_filter([vec(A0,A1)|Atoms],[msw(A0,A1)|Msws]):-$pp_vec_msw_filter(Atoms,Msws).
$pp_vec_msw_filter([A|Atoms],[A|Msws]):-$pp_vec_msw_filter(Atoms,Msws).



vec_core:-$pp_vec_core(em).
vec_core(Goals) :-$pp_vec_core(em,Goals).

$pp_vec_core(Mode) :-
	$pp_learn_data_file(FileName),
	load_clauses(FileName,Goals,[]),!,
	$pp_vec_core(Mode,Goals).

$pp_vec_core(Mode,Goals) :-
	$pp_learn_check_goals(Goals),
	$pp_learn_message(MsgS,MsgE,MsgT,MsgM),
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
	$pc_prism_prepare(GidCountPairs,Len,NGoals,FailRootIndex),
	cputime(StartEM),
	%$pp_em(Mode,Output),
	format("=======================\n"),
	cputime(EndEM),
	$pc_import_occ_switches(NewSws,NSwitches,NSwVals),
	format("=======================\n"),
	%$pp_decode_update_switches(Mode,NewSws),
	$pc_import_graph_stats(NSubgraphs,NGoalNodes,NSwNodes,AvgShared),
	cputime(End),
	format("======================\n"),
	%$pp_assert_graph_stats(NSubgraphs,NGoalNodes,NSwNodes,AvgShared),
	%$pp_assert_learn_stats(Mode,Output,NSwitches,NSwVals,TableSpace,
	%					   Start,End,StartExpl,EndExpl,StartEM,EndEM,1000),
	$pp_print_learn_stats_message(MsgT),
	format("=======================\n"),
	$pp_print_learn_end_message(MsgM,Mode),!.


