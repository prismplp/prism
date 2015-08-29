%%%%
%%%% Hindsight routine with C interface
%%%%

%%
%% hindsight(G,SubG,HProbs) :-
%%   output hindsight probs of subgoals that matches with SubG given G
%%
%% hindsight(G,SubG) :- print hindsight probs of SubG given G
%%

crf_hindsight(G) :- crf_hindsight(G,_).

crf_hindsight(G,SubG) :-
    crf_hindsight(G,SubG,HProbs),
    ( HProbs == [] -> $pp_raise_warning($msg(1404))
    ; format("hindsight weights:~n",[]),
      $pp_print_hindsight_probs(HProbs)
    ).

crf_hindsight(G,SubG,HProbs) :-
    $pp_require_tabled_probabilistic_atom(G,$msg(0006),hindsight/3),
    ( nonvar(SubG) -> $pp_require_callable(SubG,$msg(1403),hindsight/3)
    ; true
    ),
    $pp_clean_infer_stats,
    cputime(T0),
    $pp_crfhindsight_core(G,SubG,HProbs0),
    $pp_sort_hindsight_probs(HProbs0,HProbs),
    cputime(T1),
    InfTime is T1 - T0,
    $pp_garbage_collect,
    $pp_assert_hindsight_stats1(InfTime),!.

$pp_crfhindsight_core(G,SubG,HProbs) :-
    ground(G),!,
    $pp_init_tables_aux,
    $pp_clean_graph_stats,
    $pp_init_tables_if_necessary,!,
    cputime(T0),
    $pp_find_explanations(G),!,
    cputime(T1),
    $pp_compute_crfhindsight(G,SubG,HProbs),
    cputime(T2),
    $pc_import_graph_stats(NSubgraphs,NGoalNodes,NSwNodes,AvgShared),
    $pp_assert_graph_stats(NSubgraphs,NGoalNodes,NSwNodes,AvgShared),
    SearchTime  is T1 - T0,
    NumCompTime is T2 - T1,
    $pp_assert_hindsight_stats2(SearchTime,NumCompTime),!.

$pp_crfhindsight_core(G,SubG,HProbs) :-
    $pp_init_tables_aux,
    $pp_clean_graph_stats,
    $pp_init_tables_if_necessary,!,
    copy_term(G,GoalCp),
    ( $pp_trans_one_goal(GoalCp,CompGoal) -> BodyGoal = CompGoal
    ; BodyGoal = (savecp(CP),Depth=0,
                  $pp_expl_interp_goal(GoalCp,Depth,CP,[],_,[],_,[],_,[],_))
    ),
    $pp_create_dummy_goal(DummyGoal),
    Clause = (DummyGoal:-BodyGoal,
                         $pc_prism_goal_id_register(GoalCp,GId),
                         $pc_prism_goal_id_register(DummyGoal,HId),
                         $prism_eg_path(HId,[GId],[])),
    Prog = [pred(DummyGoal,0,_Mode,_Delay,tabled(_,_,_,_),[Clause]),
            pred('$damon_load',0,_,_,_,[('$damon_load':-true)])],
    $pp_consult_preds_cond([],Prog),!,
    cputime(T0),
    $pp_find_explanations(DummyGoal),!,
    cputime(T1),
    $pp_compute_crfhindsight(DummyGoal,SubG,HProbs),
    cputime(T2),
    $pc_import_graph_stats(NSubgraphs,NGoalNodes,NSwNodes,AvgShared),
    $pp_assert_graph_stats(NSubgraphs,NGoalNodes,NSwNodes,AvgShared),
    SearchTime  is T1 - T0,
    NumCompTime is T2 - T1,
    $pp_assert_hindsight_stats2(SearchTime,NumCompTime),!.

% Sws = [sw(Id,Instances,Probs,PseudoCs,Fixed,FixedH),...]
$pp_compute_crfhindsight(Goal,SubG,HProbs) :-
    $pp_export_sw_info,
    $pc_prism_goal_id_get(Goal,Gid),
    $pc_compute_crfhindsight(Gid,SubG,0,HProbs0), % "0" indicates "unconditional"
    $pp_decode_hindsight(HProbs0,HProbs),!.
