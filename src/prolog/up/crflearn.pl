crf_learn(Gs):- call($pp_crflearn_core(Gs)).

$pp_crflearn_check_goals(Goals) :-
    $pp_require_observed_data(Goals,$msg(1302),$pp_crflearn_core/1),
    $pp_crflearn_check_goals1(Goals).

$pp_crflearn_check_goals1([]).
$pp_crflearn_check_goals1([G0|Gs]) :-
    ( (G0 = goal(G,Count) ; G0 = count(G,Count) ; G0 = (Count times G) ) ->
        $pp_require_positive_integer(Count,$msg(1306),$pp_crflearn_core/1)
    ; G = G0
    ),
    $pp_require_tabled_probabilistic_atom(G,$msg(1303),$pp_crflearn_core/1),!,
    $pp_crflearn_check_goals1(Gs).

$pp_crflearn_core(Goals) :-
    $pp_crflearn_check_goals(Goals),
    $pp_learn_message(MsgS,MsgE,MsgT,MsgM),
    $pc_set_em_message(MsgE),
    cputime(Start),
    $pp_clean_crflearn_info,
    $pp_trans_crf_goals(Goals,Table,GoalCountPairs,AllGoals),
    $pp_trans_crf_countpairs(GoalCountPairs,GoalCountPairs1),
    global_set($pg_observed_facts,GoalCountPairs1),
    cputime(StartExpl),
    global_set($pg_num_goals,0),
    $pp_find_explanations(AllGoals),!,
    $pp_print_num_goals(MsgS),
    cputime(EndExpl),
    statistics(table,[TableSpace,_]),
    $pp_format_if(MsgM,"Exporting switch information to the CRF-learn routine ... "),
    flush_output,
    $pp_export_sw_info,
    $pp_format_if(MsgM,"done~n"),
    $pp_crf_observed_facts(GoalCountPairs,GidCountPairs,Table,
                       0,Len,0,NGoals,-1,FailRootIndex),
    $pp_check_failure_in_crflearn(Goals,FailRootIndex,crflearn/1),
    $pc_crf_prepare(GidCountPairs,Len,NGoals,FailRootIndex),
    cputime(StartGrd),
    $pp_grd(Output),
    cputime(EndGrd),
    $pc_import_occ_crf_switches(NewSws,NSwitches,NSwVals),
    $pp_decode_update_switches(ml,NewSws),
    $pc_import_graph_stats(NSubgraphs,NGoalNodes,NSwNodes,AvgShared),
    cputime(End),
    $pp_assert_graph_stats(NSubgraphs,NGoalNodes,NSwNodes,AvgShared),
    $pp_assert_crf_learn_stats(Output,NSwitches,NSwVals,TableSpace,
                           Start,End,StartExpl,EndExpl,StartGrd,EndGrd,1000),
    $pp_print_learn_stats_message(MsgT),
    $pp_print_crf_learn_end_message(MsgM),!.

$pp_print_crf_learn_end_message(Msg) :-
    ( Msg == 0 -> true
    ; format("Type show_sw to show the lambdas.~n",[])
    ).

$pp_grd(Output):-
    $pc_prism_grd(Iterate,Likelihood),
    Output = [Iterate,Likelihood].

$pp_clean_crflearn_info :-
    $pp_clean_dummy_goal_table,
    $pp_clean_graph_stats,
    $pp_clean_learn_stats,
    $pp_init_tables_aux,
    $pp_init_tables_if_necessary,!.

$pp_assert_crf_learn_stats(Output,NSwitches,NSwVals,TableSpace,
                       Start,End,StartExpl,EndExpl,StartEM,EndEM,UnitsPerSec) :-
    assertz($ps_num_switches(NSwitches)),
    assertz($ps_num_switch_values(NSwVals)),
    ( integer(TableSpace) -> assertz($ps_learn_table_space(TableSpace)) ; true ),
    Time is (End - Start) / UnitsPerSec,
    assertz($ps_learn_time(Time)),
    TimeExpl is (EndExpl - StartExpl) / UnitsPerSec,
    assertz($ps_learn_search_time(TimeExpl)),
    TimeEM is (EndEM - StartEM) / UnitsPerSec,
    assertz($ps_em_time(TimeEM)),
    Output = [Iterate,Likelihood],
    assertz($ps_num_iterations(Iterate)),
    assertz($ps_log_likelihood(Likelihood)),!.

$pp_trans_crf_countpairs([],[]).
$pp_trans_crf_countpairs([goal(Goal,Count,_PGidx)|GoalCountPairs],GoalCountPairs1):-
    GoalCountPairs1 = [goal(Goal,Count)|GoalCountPairs0],!,
    $pp_trans_crf_countpairs(GoalCountPairs,GoalCountPairs0).

$pp_trans_crf_goals(Goals,ParentTable,GoalCountPairs,AllGoals) :-
    $pp_build_crf_count_pairs(Goals,Pairs,ParentPairs),
    new_hashtable(PCountTable),
    $pp_trans_crf_count_pairs(Pairs,GoalCountPairs0,PCountTable,AllGoals0),
    new_hashtable(ParentTable),
    $pp_trans_crf_parent_count_pairs(ParentPairs,PCountTable,ParentTable,PGoalCountPairs,AllGoals1),
    append(GoalCountPairs0,PGoalCountPairs,GoalCountPairs),
    append(AllGoals0,AllGoals1,AllGoals).

$pp_build_crf_count_pairs(Goals,Pairs,ParentPairs) :-
    new_hashtable(Table),
    new_hashtable(Table2),
    $pp_count_crf_goals(Goals,Table,Table2,0),
    hashtable_to_list(Table,Pairs0),
    hashtable_to_list(Table2,ParentPairs0),
    sort(Pairs0,Pairs),
    sort(ParentPairs0,ParentPairs).

$pp_count_crf_goals([],_,_,_).
$pp_count_crf_goals([G0|Goals],Table,Table2,N) :-
    ( G0 = goal(Goal,Count)  -> true
    ; G0 = count(Goal,Count) -> true
    ; G0 = (Count times Goal) -> true
    ; Goal = G0, Count = 1
    ),
    $pp_require_ground(Goal,$msg(1601),$pp_crflearn_core),
    ( $pp_hashtable_get(Table,Goal,(Count0,Pid)) ->
        Count1 is Count0 + Count,
        $pp_hashtable_put(Table,Goal,(Count1,Pid)),
        N1 = N
    ; $pp_build_parent_goal(Goal,ParentGoal0),
        copy_term(ParentGoal0,ParentGoal),
        ( $pp_hashtable_get(Table2,ParentGoal,Pid) ->
            $pp_hashtable_put(Table,Goal,(Count,Pid)),N1 = N
        ; N1 is N + 1,
            $pp_hashtable_put(Table2,ParentGoal,N),
            $pp_hashtable_put(Table,Goal,(Count,N))
        )
    ),!,
    $pp_count_crf_goals(Goals,Table,Table2,N1).

$pp_build_parent_goal(Goal,ParentGoal) :-
    ( Goal =.. [F,X,_] -> ParentGoal =.. [F,X]
    ; $pp_raise_runtime_error($msg(1602),Goal,$pp_crf_learn_core)
    ),
    $pp_require_tabled_probabilistic_atom(ParentGoal,$msg(1604),$pp_crf_learn_core).

$pp_trans_crf_count_pairs([],[],_,[]).
$pp_trans_crf_count_pairs([Goal=(Count,PGidx)|Pairs],GoalCountPairs,PCountTable,AllGoals) :-
    $pp_build_dummy_goal(Goal,DummyGoal),
    GoalCountPairs = [goal(DummyGoal,Count,PGidx)|GoalCountPairs1],
    AllGoals = [DummyGoal|AllGoals1],
    ( $pp_hashtable_get(PCountTable,PGidx,Count0) ->
        Count1 is Count0 + Count,
        $pp_hashtable_put(PCountTable,PGidx,Count1)
    ; $pp_hashtable_put(PCountTable,PGidx,Count)
    ),!,
    $pp_trans_crf_count_pairs(Pairs,GoalCountPairs1,PCountTable,AllGoals1).

$pp_trans_crf_parent_count_pairs([],_,_,[],[]).
$pp_trans_crf_parent_count_pairs([PGoal=PGidx|ParentPairs],PCountTable,Table,PGoalCountPairs,AllGoals) :-
    $pp_build_dummy_goal(PGoal,DummyGoal),
    $pp_hashtable_put(Table,PGidx,DummyGoal),
    $pp_hashtable_get(PCountTable,PGidx,Count),
    PGoalCountPairs = [goal(DummyGoal,Count,-1)|PGoalCountPairs1],
    AllGoals = [DummyGoal|AllGoals1],!,
    $pp_trans_crf_parent_count_pairs(ParentPairs,PCountTable,Table,PGoalCountPairs1,AllGoals1).

$pp_crf_observed_facts([],[],_Table,Len,Len,NGoals,NGoals,FailRootIndex,FailRootIndex).
$pp_crf_observed_facts([goal(Goal,Count,PGidx)|GoalCountPairs],GidCountPairs,Table,
                   Len0,Len,NGoals0,NGoals,FailRootIndex0,FailRootIndex) :-
    % fails if the goal is ground but has no proof
    ( $pc_prism_goal_id_get(Goal,Gid) ->
        ( Goal == failure ->
            NGoals1 = NGoals0,
            FailRootIndex1 = Len0
        ; NGoals1 is NGoals0 + Count,
          FailRootIndex1 = FailRootIndex0
        ),
        ( $pp_hashtable_get(Table,PGidx,PGoal) ->
            $pc_prism_goal_id_get(PGoal,PGid)
        ; PGid = PGidx
        ),
        GidCountPairs = [goal(Gid,Count,PGid)|GidCountPairs1],
        Len1 is Len0 + 1
    ; $pp_raise_unexpected_failure($pp_crf_observed_facts/8)
    ),!,
    $pp_crf_observed_facts(GoalCountPairs,GidCountPairs1,Table,
                       Len1,Len,NGoals1,NGoals,FailRootIndex1,FailRootIndex).

$pp_check_failure_in_crflearn(Gs,FailRootIndex,Source) :-
    ( FailRootIndex >= 0 ->
        $pp_raise_runtime_error($msg(1603),[Gs],failure_in_crf_learn,Source)
    ; true
    ).
