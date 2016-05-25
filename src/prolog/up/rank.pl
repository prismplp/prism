
rank_learn(Goals) :-
    get_prism_flag(learn_mode,Mode0),
    $pp_conv_learn_mode(Mode0,Mode,VT),
    ( VT = 0 -> $pp_rank_learn_main(Mode,Goals)
    ; VT = 1 -> $pp_rank_vlearn_main(Mode,Goals)
    ).

$pp_rank_learn_main(Mode,Goals) :- $pp_rank_learn_core(Mode,Goals,0).

$pp_rank_learn_core(Mode,Goals,Debug) :-
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
	%format("dd~w\n",[GoalCountPairs,GidCountPairs,0,Len,0,NGoals,-1,FailRootIndex]),
    $pp_observed_facts(GoalCountPairs,GidCountPairs,
                       0,Len,0,NGoals,-1,FailRootIndex),
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

