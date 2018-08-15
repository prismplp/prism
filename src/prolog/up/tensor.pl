%%%%
%%%% utility
%%%%

set_index_range(Index,R):-
	$pc_set_index_range(Index,R).

save_placeholder_goals(PlaceholderGoals,Goals):-save_placeholder_goals('data.h5',hdf5,PlaceholderGoals,Goals).
save_placeholder_goals(Filename,PlaceholderGoals,Goals):-save_placeholder_goals(Filename,hdf5,PlaceholderGoals,Goals).
save_placeholder_goals(Filename,Mode,PlaceholderGoals,Goals):-
	$pp_generate_placeholder_goals(PlaceholderGoals,Goals,0,GG,Var),
	save_placeholder_data(Filename,Mode,Var,GG).

$pp_generate_placeholder([],N,N).
$pp_generate_placeholder([V|VV],N,O):-
	M is N+1,
	term2atom(M,AM),
	atom_concat($placeholder,AM,V0),
	atom_concat(V0,$,V),
	$pp_generate_placeholder(VV,M,O).

$pp_generate_placeholder_goals([],_,_,[],[]).
$pp_generate_placeholder_goals([GP0|GP],Goals,N,[X|XX],[V0|Var]):-
	copy_term(GP0,GP1),
	term_variables(GP0,V0),
	term_variables(GP1,V1),
	$pp_generate_placeholder(V0,N,M),
	findall(V1,member(GP1,Goals),X),
	$pp_generate_placeholder_goals(GP,Goals,M,XX,Var).
	

save_placeholder_data(Placeholders,Data):-save_placeholder_data('data.h5',hdf5,Placeholders,Data).
save_placeholder_data(Filename,Placeholders,Data):-save_placeholder_data(Filename,hdf5,Placeholders,Data).
save_placeholder_data(Filename,Mode,Placeholders,Data):-
	(Mode==json ->Mode0=0
	;Mode==pb   ->Mode0=1
	;Mode==pbtxt->Mode0=2
	;Mode==hdf5 ->Mode0=3
	;$pp_raise_runtime_error($msg(9804),unknown_save_format,save_placeholder_data/4)),
	$pc_save_placeholder_data(Filename,Mode0,Placeholders,Data,20000000).

%%%
%%% save prism flags
%%%

save_flags:-save_flags('flags.json',json).
save_flags(Filename):-save_flags(Filename,json).
save_flags(Filename,Mode):-findall([X,F],get_prism_flag(X,F),G),
	$pc_set_export_flags(G),
	(Mode==json ->Mode0=0
	;Mode==pb   ->Mode0=1
	;Mode==pbtxt->Mode0=2
	;Mode==hdf5 ->Mode0=$pp_raise_runtime_error($msg(9806),hdf5_is_not_supportted_for_saving_flags,save_flags/2)
	;$pp_raise_runtime_error($msg(9804),unknown_save_format,save_flags/2)),
	$pc_save_options(Filename,Mode0).

%%%%
%%%% save explanation graph
%%%%

unique([],[]).
unique([X],[X]).
unique([H|T],[H|S]) :- not(member(H, T)), unique(T,S).
unique([H|T],S) :- member(H, T), unique(T,S).

$pp_trans_phase_tensor(Prog0,Prog_tensor,Info):-
	maplist(X,Y,CollectList,(
		X=pred(index,2,A2,A3,A4,Clauses)->
			($pp_tensor_parse_clauses(Clauses,C,CollectList),
			Y=pred(values,2,A2,A3,A4,C));
		X=pred(A0,A1,A2,A3,A4,Clauses)->
			($pp_tensor_parse_clauses(Clauses,C,CollectList),
			Y=pred(A0,A1,A2,A3,A4,C));
		Y=X,CollectList=[]
	),Prog0,Prog_tensor0,CollectLists),
	flatten(CollectLists,FlatCList),
	maplist(C,X,C=data(_,X),FlatCList,IndexList),
	flatten(IndexList,IndexAtoms),
	unique(IndexAtoms,UIndexAtoms),
	assert(index_atoms(UIndexAtoms)),
	Pred1=pred(values,3,_,_,_,[values($operator(_),[$operator],fix@[1.0])]),
	Prog_tensor=[Pred1|Prog_tensor0].

$pp_tensor_parse_clauses(Clauses,NewClauses,CollectLists):-
	maplist(C,NC,CollectList,(
		$pp_tensor_get_msws(C,NC,CollectList)
		),Clauses,NewClauses,CollectLists).

$pp_tensor_get_msws(Clause,MswClause,CollectList):-
	Clause=(H:-Body) -> (
		Clause=(H:-Body),
		and_to_list(Body,LBody),
		$pp_tensor_msw_filter(LBody,MswLBody,VecList),
		list_to_and(MswLBody,MswBody),
		CollectList=data(VecList,[]),
		MswClause=(H:-(MswBody)));
	Clause=index(_,_) -> (
		Clause=index(A0,A1),
		CollectList=data([],A1),
		MswClause=values(tensor(A0),A1));
	MswClause=Clause,VecList=data([],[]).
	
$pp_tensor_msw_filter([],[],[]).
$pp_tensor_msw_filter([tensor(A0,A1)|Atoms],[msw(tensor(A0),A1)|Msws],[tensor(A0,A1)|VecList]):-$pp_tensor_msw_filter(Atoms,Msws,VecList).
$pp_tensor_msw_filter([vector(A0,A1)|Atoms],[msw(tensor(A0),A1)|Msws],[tensor(A0,A1)|VecList]):-$pp_tensor_msw_filter(Atoms,Msws,VecList).
$pp_tensor_msw_filter([matrix(A0,A1)|Atoms],[msw(tensor(A0),A1)|Msws],[tensor(A0,A1)|VecList]):-$pp_tensor_msw_filter(Atoms,Msws,VecList).
$pp_tensor_msw_filter([operator(A0)|Atoms],[msw($operator(A0),$operator)|Msws],VecList):-$pp_tensor_msw_filter(Atoms,Msws,VecList).
$pp_tensor_msw_filter([A|Atoms],[A|Msws],VecList):-$pp_tensor_msw_filter(Atoms,Msws,VecList).

$pp_tensor_values_filter([],[],[]).
$pp_tensor_values_filter([index(A0,A1)|Atoms],[values(tensortor(A0),A1)|Msws],[index(A0,A1)|VecList]):-$pp_tensor_values_filter(Atoms,Msws,VecList).
$pp_tensor_values_filter([A|Atoms],[A|Msws],VecList):-$pp_tensor_values_filter(Atoms,Msws,VecList).


%%%%%%%%%%%

$pp_tensor_value_generator([],V,V).
$pp_tensor_value_generator([Vec|VecLists],InValues,OutValues):-
	$pp_tensor_value_generator1(Vec,InValues,Values),
	$pp_tensor_value_generator(VecLists,Values,OutValues).

$pp_tensor_value_generator1([],V,V).
$pp_tensor_value_generator1([Vec|VecLists],InValues,OutValues):-
	$pp_tensor_value_generator2(Vec,InValues,Values),
	$pp_tensor_value_generator1(VecLists,Values,OutValues).


$pp_tensor_value_generator2([],V,V).
$pp_tensor_value_generator2([Vec|VecLists],InValues,OutValues):-
	$pp_tensor_conv_tensorterm(Vec,NewVec),
	$pp_tensor_conv_valterm(NewVec,NewVal),
	%InValues,Values
	Values=[NewVal | InValues],
	$pp_tensor_value_generator2(VecLists,Values,OutValues).

$pp_tensor_conv_tensorterm(Pred,NewPred):-
	Pred=tensor(A1term,A2term),
	%A1term=..[F|Args],
	%$pp_tensor_conv_args(Args,NewArgs),
	%A1newterm=..[F|NewArgs],
	%%
	term2atom(A2term,A2atom),
	atom_concat('dummy_',A2atom,A2newatom),
	NewPred=tensor(A1term,A2newatom).
	
$pp_tensor_conv_valterm(Pred,NewPred):-
	Pred=tensor(A1term,A2term),
	NewPred=pred(values,2,_,_,_,[values(A1term,[A2term])]).

$pp_tensor_conv_args([],[]).
$pp_tensor_conv_args([Arg|Args],[NewArg|NewArgs]):-
	nonvar(Arg)->Arg=NewArg;(
		term2atom(Arg,A1atom),
		atom_concat('dummy_',A1atom,NewArg)),
	$pp_tensor_conv_args(Args,NewArgs).

%%%%%%%%%%%%

save_expl_graph:-$pp_tensor_core('expl.json',0).
save_expl_graph(Goals) :-$pp_tensor_core('expl.json',0,Goals).
save_expl_graph(Filename,Goals) :-$pp_tensor_core(Filename,0,Goals).
save_expl_graph(Filename,Mode,Goals) :-$pp_tensor_core(Filename,Mode,Goals).

$pp_tensor_core(Filename,Mode) :-
	$pp_learn_data_file(DataName),
	load_clauses(DataName,Goals,[]),!,
	$pp_tensor_core(Mode,Goals).

$pp_tensor_core(Filename,Mode,GoalList) :-
	flatten(GoalList,Goals),
	$pp_learn_check_goals(Goals),
	$pp_learn_message(MsgS,MsgE,MsgT,MsgM),
	cputime(Start),
	$pp_clean_learn_info,
	$pp_trans_goals(Goals,GoalCountPairs,AllGoals),!,
	global_set($pg_observed_facts,GoalCountPairs),
	cputime(StartExpl),
	global_set($pg_num_goals,0),
	$pp_find_explanations(AllGoals),!,
	$pp_print_num_goals(MsgS),
	cputime(EndExpl),
	$pp_format_if(MsgM,"Exporting switch information ... "),
	flush_output,
	$pp_export_sw_info,
	$pp_format_if(MsgM,"done~n"),
	$pp_observed_facts(GoalCountPairs,GidCountPairs,
					   0,Len,0,NGoals,-1,FailRootIndex),
	$pc_clear_goal_rank,
	$pc_set_goal_rank(GoalList),
	$pc_prism_prepare(GidCountPairs,Len,NGoals,FailRootIndex),
	cputime(StartEM),
	%$pp_em(Mode,Output),
	$pc_prism_save_expl_graph(Filename,Mode),
	%format("=======================\n"),
	cputime(EndEM),
	$pc_import_occ_switches(NewSws,NSwitches,NSwVals),
	%format("=======================\n"),
	%$pp_decode_update_switches(Mode,NewSws),
	$pc_import_graph_stats(NSubgraphs,NGoalNodes,NSwNodes,AvgShared),
	cputime(End),!.
	%format("======================\n"),
	%$pp_assert_graph_stats(NSubgraphs,NGoalNodes,NSwNodes,AvgShared),
	%$pp_assert_learn_stats(Mode,Output,NSwitches,NSwVals,TableSpace,
	%					   Start,End,StartExpl,EndExpl,StartEM,EndEM,1000),
	%$pp_print_learn_stats_message(MsgT),!.


