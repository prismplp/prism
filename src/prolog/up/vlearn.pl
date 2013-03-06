vlearn_p :-
    $pp_vlearn_main(ml).
vlearn_p(Goals) :-
    $pp_vlearn_main(ml,Goals).
vlearn_h :-
    $pp_vlearn_main(vb).
vlearn_h(Goals) :-
    $pp_vlearn_main(vb,Goals).
vlearn_b :-
    $pp_vlearn_main(both).
vlearn_b(Goals) :-
    $pp_vlearn_main(both,Goals).

%% for the parallel version
$pp_vlearn_main(Mode) :- call($pp_vlearn_core(Mode)).
$pp_vlearn_main(Mode,Goals) :- call($pp_vlearn_core(Mode,Goals)).

$pp_vlearn_core(Mode) :-
    $pp_learn_data_file(FileName),
    load_clauses(FileName,Goals,[]),!,
    $pp_vlearn_core(Mode,Goals).

$pp_vlearn_core(Mode,Goals) :-
    $pp_vlearn_check_goals(Goals),
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
    $pp_format_if(MsgM,"Exporting switch information to the VT routine ... "),
    flush_output,
    $pp_export_sw_info,
    $pp_format_if(MsgM,"done~n"),
    $pp_observed_facts(GoalCountPairs,GidCountPairs,
                       0,Len,0,NGoals,-1,FailRootIndex),
    $pc_prism_prepare(GidCountPairs,Len,NGoals,FailRootIndex),
    cputime(StartVT),
    $pp_vt(Mode,Output),
    cputime(EndVT),
    $pc_import_occ_switches(NewSws,NSwitches,NSwVals),
    $pp_decode_update_switches(Mode,NewSws),
    $pc_import_graph_stats(NSubgraphs,NGoalNodes,NSwNodes,AvgShared),
    cputime(End),
    $pp_assert_graph_stats(NSubgraphs,NGoalNodes,NSwNodes,AvgShared),
    $pp_assert_vlearn_stats(Mode,Output,NSwitches,NSwVals,TableSpace,
                            Start,End,StartExpl,EndExpl,StartVT,EndVT,1000),
    $pp_print_learn_stats_message(MsgT),
    $pp_print_learn_end_message(MsgM,Mode),!.

$pp_vlearn_check_goals(Goals) :-
    $pp_require_observed_data(Goals,$msg(1302),$pp_vlearn_core/1),
    $pp_vlearn_check_goals1(Goals).

$pp_vlearn_check_goals1([]).
$pp_vlearn_check_goals1([G0|Gs]) :-
    ( (G0 = goal(G,Count) ; G0 = count(G,Count) ; G0 = (Count times G) ) ->
        $pp_require_positive_integer(Count,$msg(1306),$pp_vlearn_core/1)
    ; G = G0
    ),
    $pp_require_tabled_probabilistic_atom(G,$msg(1303),$pp_vlearn_core/1),!,
    $pp_learn_check_goals1(Gs).

$pp_vt(ml,Output) :-
    $pc_prism_vt(Iterate,LogPost,LogLike,ModeSmooth),
    Output = [Iterate,LogPost,LogLike,ModeSmooth].
$pp_vt(vb,Output) :-
    $pc_prism_vbvt(IterateVB,FreeEnergy),
    Output = [IterateVB,FreeEnergy].
$pp_vt(both,Output) :-
    $pc_prism_both_vt(IterateVB,FreeEnergy),
    Output = [IterateVB,FreeEnergy].

$pp_assert_vlearn_stats(Mode,Output,NSwitches,NSwVals,TableSpace,
                       Start,End,StartExpl,EndExpl,StartVT,EndVT,UnitsPerSec) :-
    assertz($ps_num_switches(NSwitches)),
    assertz($ps_num_switch_values(NSwVals)),
    ( integer(TableSpace) -> assertz($ps_learn_table_space(TableSpace)) ; true ),
    Time is (End - Start) / UnitsPerSec,
    assertz($ps_learn_time(Time)),
    TimeExpl is (EndExpl - StartExpl) / UnitsPerSec,
    assertz($ps_learn_search_time(TimeExpl)),
    TimeVT is (EndVT - StartVT) / UnitsPerSec,
    assertz($ps_em_time(TimeVT)),
    $pp_assert_vlearn_stats_sub(Mode,Output),!.

$pp_assert_vlearn_stats_sub(ml,Output) :-
    Output = [Iterate,LogPost,LogLike,ModeSmooth],
    assertz($ps_num_iterations(Iterate)),
    ( ModeSmooth > 0 -> assertz($ps_log_post(LogPost)) ; true ),
    assertz($ps_log_likelihood(LogLike)),!.

$pp_assert_vlearn_stats_sub(vb,Output) :-
    Output = [IterateVB,FreeEnergy],
    assertz($ps_num_iterations_vb(IterateVB)),
    assertz($ps_free_energy(FreeEnergy)),!.

$pp_assert_vlearn_stats_sub(both,Output) :-
    Output = [IterateVB,FreeEnergy],
    assertz($ps_num_iterations_vb(IterateVB)),
    assertz($ps_free_energy(FreeEnergy)),!.
