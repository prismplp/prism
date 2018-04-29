
$pp_trans_phase_vec(Prog0,Prog_vec,Info):-
	maplist(X,Y,Vecs,(
		X=pred(indeces,2,A2,A3,A4,Clauses)->
			($pp_vec_parse_clauses(Clauses,C,Vecs),
			Y=pred(values,2,A2,A3,A4,C));
		X=pred(A0,A1,A2,A3,A4,Clauses)->
			($pp_vec_parse_clauses(Clauses,C,Vecs),
			Y=pred(A0,A1,A2,A3,A4,C));
		Y=X,Vecs[]
	),Prog0,Prog_vec0,VecLists),
	%$pp_vec_value_generator(VecLists,[],Values),
	format("~w \n",Values),
	Prog_vec0=Prog_vec.
	%append(Prog_vec0,Values,Prog_vec),
	%format("~w",Prog_vec).

$pp_vec_parse_clauses(Clauses,NewClauses,VecList):-
	maplist(C,NC,Vecs,(format(">>~w\n",C),
	$pp_vec_get_msws(C,NC,Vecs),format(">>>>~w\n",NC),format("~w\n",Vecs)
		),Clauses,NewClauses,VecList).

$pp_vec_get_msws(Clause,MswClause,VecList):-
	Clause=(H:-Body) -> (
		Clause=(H:-Body),
		and_to_list(Body,LBody),
		$pp_vec_msw_filter(LBody,MswLBody,VecList),
		list_to_and(MswLBody,MswBody),
		MswClause=(H:-(MswBody)));
	Clause=indeces(_,_) -> (
		Clause=indeces(A0,A1),
		VecList=[],
		MswClause=values(vector(A0),A1));
	MswClause=Clause,VecList=[].
	
$pp_vec_msw_filter([],[],[]).
$pp_vec_msw_filter([vec(A0,A1)|Atoms],[msw(vector(A0),A1)|Msws],[vec(A0,A1)|VecList]):-$pp_vec_msw_filter(Atoms,Msws,VecList).
$pp_vec_msw_filter([A|Atoms],[A|Msws],VecList):-$pp_vec_msw_filter(Atoms,Msws,VecList).

$pp_vec_values_filter([],[],[]).
$pp_vec_values_filter([indeces(A0,A1)|Atoms],[values(vector(A0),A1)|Msws],[indeces(A0,A1)|VecList]):-$pp_vec_values_filter(Atoms,Msws,VecList).
$pp_vec_values_filter([A|Atoms],[A|Msws],VecList):-$pp_vec_values_filter(Atoms,Msws,VecList).


%%%%%%%%%%%

$pp_vec_value_generator([],V,V).
$pp_vec_value_generator([Vec|VecLists],InValues,OutValues):-
	$pp_vec_value_generator1(Vec,InValues,Values),
	$pp_vec_value_generator(VecLists,Values,OutValues).

$pp_vec_value_generator1([],V,V).
$pp_vec_value_generator1([Vec|VecLists],InValues,OutValues):-
	$pp_vec_value_generator2(Vec,InValues,Values),
	$pp_vec_value_generator1(VecLists,Values,OutValues).


$pp_vec_value_generator2([],V,V).
$pp_vec_value_generator2([Vec|VecLists],InValues,OutValues):-
	$pp_vec_conv_vecterm(Vec,NewVec),
	$pp_vec_conv_valterm(NewVec,NewVal),
	%InValues,Values
	Values=[NewVal | InValues],
	$pp_vec_value_generator2(VecLists,Values,OutValues).

$pp_vec_conv_vecterm(Pred,NewPred):-
	Pred=vec(A1term,A2term),
	%A1term=..[F|Args],
	%$pp_vec_conv_args(Args,NewArgs),
	%A1newterm=..[F|NewArgs],
	%%
	term2atom(A2term,A2atom),
	atom_concat('dummy_',A2atom,A2newatom),
	NewPred=vec(A1term,A2newatom).
	
$pp_vec_conv_valterm(Pred,NewPred):-
	Pred=vec(A1term,A2term),
	NewPred=pred(values,2,_,_,_,[values(A1term,[A2term])]).

$pp_vec_conv_args([],[]).
$pp_vec_conv_args([Arg|Args],[NewArg|NewArgs]):-
	nonvar(Arg)->Arg=NewArg;(
		term2atom(Arg,A1atom),
		atom_concat('dummy_',A1atom,NewArg)),
	$pp_vec_conv_args(Args,NewArgs).

%%%%%%%%%%%%

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
	$pc_prism_vec(_),
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


