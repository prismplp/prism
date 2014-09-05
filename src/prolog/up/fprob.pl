%
crf_prob(Goal) :-
    fprob(Goal,P),
    ( $pp_in_log_scale -> Text = 'Log-weight' ; Text = 'Weight' ),
    format("~w of ~w is: ~15f~n",[Text,Goal,P]).

crf_prob(Goal,Prob) :-
    $pp_require_tabled_probabilistic_atom(Goal,$msg(0006),prob/2),
    $pp_fprob(Goal,Prob).

$pp_fprob(msw(Sw,V),Prob) :-
    $pp_require_ground(Sw,$msg(0101),prob/2),
    $pp_require_switch_outcomes(Sw,$msg(0102),prob/2),
    $pp_clean_infer_stats,
    ( var(V) ->
        cputime(T0),
        ( $pp_in_log_scale -> Prob = 0.0 ; Prob = 1.0 ),
        cputime(T1),
        InfTime is T1 - T0,
        $pp_assert_prob_stats1(InfTime)
    ; % else
        cputime(T0),
        $pp_get_value_prob(Sw,V,Prob0),
        ( $pp_in_log_scale -> Prob is log(Prob0) ; Prob = Prob0 ),
        cputime(T1),
        InfTime is T1 - T0,
        $pp_assert_prob_stats1(InfTime)
    ),
    $pp_assert_prob_stats2(0.0,0.0),!.

$pp_fprob(Goal,Prob) :-
    $pp_clean_infer_stats,
    cputime(T0),
    $pp_fprob_core(Goal,Prob),
    cputime(T1),
    InfTime is T1 - T0,
    $pp_assert_prob_stats1(InfTime),!.

log_crf_prob(Goal) :-
    log_fprob(Goal,P),format("Log-weight of ~w is: ~15f~n",[Goal,P]).
log_crf_prob(Goal,P) :-
    $pp_fprob(Goal,P0),( $pp_in_log_scale -> P = P0 ; P is log(P0) ).

$pp_fprob_core(Goal,Prob) :-
    ground(Goal),
    $pp_is_tabled_probabilistic_atom(Goal),!,
    $pp_init_tables_aux,
    $pp_clean_graph_stats,
    $pp_init_tables_if_necessary,!,
    cputime(T1),
    $pp_find_explanations(Goal),
    cputime(T2),
    $pp_compute_inside_feature(Goal,Prob),!,
    cputime(T3),
    $pc_import_graph_stats(NSubgraphs,NGoalNodes,NSwNodes,AvgShared),
    $pp_assert_graph_stats(NSubgraphs,NGoalNodes,NSwNodes,AvgShared),
    SearchTime  is T2 - T1,
    NumCompTime is T3 - T2,
    $pp_assert_prob_stats2(SearchTime,NumCompTime),!.

$pp_fprob_core(Goal,Prob) :-
    $pp_init_tables_aux,
    $pp_clean_graph_stats,
    $pp_init_tables_if_necessary,!,
    copy_term(Goal,GoalCp),
    ( $pp_trans_one_goal(GoalCp,CompGoal) -> BodyGoal = CompGoal
    ; BodyGoal = (savecp(CP),Depth=0,
                  $pp_expl_interp_goal(GoalCp,Depth,CP,[],_,[],_,[],_,[],_))
    ),
    $pp_create_dummy_goal(DummyGoal),
    Clause = (DummyGoal:-BodyGoal,
                         $pc_prism_goal_id_register(GoalCp,GId),
                         $pc_prism_goal_id_register(DummyGoal,HId),
                         $prism_eg_path(HId,[GId],[])),
    Prog = [pred(DummyGoal,0,_Mode,_Delay,tabled(_,_,_,_),[Clause])],
    $pp_consult_preds_cond([],Prog),!,
    cputime(T1),
    $pp_find_explanations(DummyGoal),
    cputime(T2),
    $pp_compute_inside_feature(DummyGoal,Prob),
    cputime(T3),
    $pc_import_graph_stats(NSubgraphs,NGoalNodes,NSwNodes,AvgShared),
    $pp_assert_graph_stats(NSubgraphs,NGoalNodes,NSwNodes,AvgShared),
    SearchTime  is T2 - T1,
    NumCompTime is T3 - T2,
    $pp_assert_prob_stats2(SearchTime,NumCompTime),!.

% Sws = [sw(Id,Instances,Probs,Deltas,FixedP,FixedH),...]
$pp_compute_inside_feature(Goal,Prob) :-
    $pp_export_sw_info,
    $pc_prism_goal_id_get(Goal,Gid),
    $pc_compute_feature(Gid,Prob),!.

crf_probf(Goal) :-
    $pp_fprobf(Goal,Expls,1,0), \+ \+ print_graph(Expls,[lr('<=>')]).
crf_probfi(Goal) :-
    $pp_fprobf(Goal,Expls,1,1), \+ \+ print_graph(Expls,[lr('<=>')]).
crf_probfo(Goal) :-
    $pp_fprobf(Goal,Expls,1,2), \+ \+ print_graph(Expls,[lr('<=>')]).
crf_probfio(Goal) :-
    $pp_fprobf(Goal,Expls,1,4), \+ \+ print_graph(Expls,[lr('<=>')]).

crf_probf(Goal,Expls) :-
    $pp_fprobf(Goal,Expls,1,0).
crf_probfi(Goal,Expls) :-
    $pp_fprobf(Goal,Expls,1,1).
crf_probfo(Goal,Expls) :-
    $pp_fprobf(Goal,Expls,1,2).
crf_probfio(Goal,Expls) :-
    $pp_fprobf(Goal,Expls,1,4).

crf_probef(Goal) :-
    $pp_fprobf(Goal,Expls,0,0), \+ \+ print_graph(Expls,[lr('<=>')]).
crf_probefi(Goal) :-
    $pp_fprobf(Goal,Expls,0,1), \+ \+ print_graph(Expls,[lr('<=>')]).
crf_probefo(Goal) :-
    $pp_fprobf(Goal,Expls,0,2), \+ \+ print_graph(Expls,[lr('<=>')]).
crf_probefio(Goal) :-
    $pp_fprobf(Goal,Expls,0,4), \+ \+ print_graph(Expls,[lr('<=>')]).

crf_probef(Goal,Expls) :-
    $pp_fprobf(Goal,Expls,0,0).
crf_probefi(Goal,Expls) :-
    $pp_fprobf(Goal,Expls,0,1).
crf_probefo(Goal,Expls) :-
    $pp_fprobf(Goal,Expls,0,2).
crf_probefio(Goal,Expls) :-
    $pp_fprobf(Goal,Expls,0,4).

crf_probef(Goal,Expls,GoalHashTab,SwHashTab) :-
    $pp_fprobf(Goal,Expls,0,0),
    $pp_get_subgoal_hashtable(GoalHashTab),
    $pp_get_switch_hashtable(SwHashTab).
crf_probefi(Goal,Expls,GoalHashTab,SwHashTab) :-
    $pp_fprobf(Goal,Expls,0,1),
    $pp_get_subgoal_hashtable(GoalHashTab),
    $pp_get_switch_hashtable(SwHashTab).
crf_probefo(Goal,Expls,GoalHashTab,SwHashTab) :-
    $pp_fprobf(Goal,Expls,0,2),
    $pp_get_subgoal_hashtable(GoalHashTab),
    $pp_get_switch_hashtable(SwHashTab).
crf_probefio(Goal,Expls,GoalHashTab,SwHashTab) :-
    $pp_fprobf(Goal,Expls,0,4),
    $pp_get_subgoal_hashtable(GoalHashTab),
    $pp_get_switch_hashtable(SwHashTab).

%% PrMode is one of 0 (none), 1 (inside + feature)

$pp_fprobf(Goal,Expls,Decode,PrMode) :-
    $pp_require_tabled_probabilistic_atom(Goal,$msg(0006),$pp_fprobf/4),
    $pp_compute_expls_feature(Goal,Expls,Decode,PrMode),
    $pp_garbage_collect.

$pp_compute_expls_feature(Goal,Expls,Decode,PrMode) :-
    Goal = msw(I,V),!,
    $pp_require_ground(I,$msg(0101),$pp_fprobf/4),
    $pp_require_switch_outcomes(I,$msg(0102),$pp_fprobf/4),
    $pp_clean_infer_stats,
    ( ground(V) -> V = VCp ; copy_term(V,VCp) ),
    $pp_create_dummy_goal(DummyGoal),
    DummyBody = ($prism_expl_msw(I,VCp,Sid),
                 $pc_prism_goal_id_register(DummyGoal,Hid),
                 $prism_eg_path(Hid,[],[Sid])),
    Prog = [pred(DummyGoal,0,_,_,tabled(_,_,_,_),[(DummyGoal:-DummyBody)])],
    $pp_consult_preds_cond([],Prog),
    cputime(T0),
    $pp_compute_expls_feature(DummyGoal,Goal,Expls,Decode,PrMode,T0),!.

$pp_compute_expls_feature(Goal,Expls,Decode,PrMode) :-
    $pp_is_tabled_probabilistic_atom(Goal),
    ground(Goal),!,
    $pp_clean_infer_stats,
    cputime(T0),
    $pp_compute_expls_feature(Goal,_,Expls,Decode,PrMode,T0),!.

$pp_compute_expls_feature(Goal,Expls,Decode,PrMode) :-
    $pp_clean_infer_stats,
    copy_term(Goal,GoalCp),
    ( $pp_trans_one_goal(GoalCp,CompGoal) ->
      BodyGoal = CompGoal
    ; BodyGoal = (savecp(CP),Depth=0,
                  $pp_expl_interp_goal(GoalCp,Depth,CP,[],_,[],_,[],_,[],_))
    ),
    $pp_create_dummy_goal(DummyGoal),
    DummyBody = (BodyGoal,
                 $pc_prism_goal_id_register(GoalCp,GId),
                 $pc_prism_goal_id_register(DummyGoal,HId),
                 $prism_eg_path(HId,[GId],[])),
    Prog = [pred(DummyGoal,0,_,_,tabled(_,_,_,_),[(DummyGoal:-DummyBody)])],
    $pp_consult_preds_cond([],Prog),
    cputime(T0),
    $pp_compute_expls_feature(DummyGoal,Goal,Expls,Decode,PrMode,T0),!.

$pp_compute_expls_feature(Goal,GLabel,Expls,Decode,PrMode,T0) :-
    $pp_init_tables_aux,
    $pp_clean_graph_stats,
    $pp_init_tables_if_necessary,!,
    cputime(T1),
    $pp_find_explanations(Goal),
    cputime(T2),
    $pc_prism_goal_id_get(Goal,Gid),
    $pc_alloc_sort_egraph(Gid),
    cputime(T3),
    ( $pp_export_sw_info,
      $pc_compute_fprobf(PrMode)
    ),
    cputime(T4),
    $pc_import_sorted_graph_size(Size),
    $pp_build_expls_feature(Size,Decode,PrMode,GLabel,Expls),
    $pc_import_graph_stats(NSubgraphs,NGoalNodes,NSwNodes,AvgShared),
    $pp_assert_graph_stats(NSubgraphs,NGoalNodes,NSwNodes,AvgShared),
    cputime(T5),
    SearchTime  is T2 - T1,
    NumCompTime is T4 - T3,
    InfTime     is T5 - T0,
    $pp_assert_prob_stats2(SearchTime,NumCompTime),
    $pp_assert_prob_stats1(InfTime),!.

$pp_build_expls_feature(I0,_,_,_,Expls), I0 =< 0 =>
    Expls = [].
$pp_build_expls_feature(I0,Decode,PrMode,GLabel,Expls), I0 > 0 =>
    I is I0 - 1,
    $pc_import_sorted_graph_gid(I,Gid),
    $pc_import_sorted_graph_paths(I,Paths0),
    ( Decode == 0    -> Label = Gid
    ; nonvar(GLabel) -> Label = GLabel
    ; $pc_prism_goal_term(Gid,Label)
    ),
    ( PrMode == 0 -> Node = node(Label,Paths)  % fprobf
    ; PrMode == 4 ->                           % fprobfio
        $pp_get_gnode_probs(PrMode,Gid,Value),
        Node = node(Label,Paths,Value),
        Value = [_,Vo]
    ; $pp_get_gnode_probs(PrMode,Gid,Value), % fprobfi,fprobfo
      Node  = node(Label,Paths,Value),
      Value = Vo % ??
    ),
    $pp_decode_paths_feature(Paths0,Paths,Decode,PrMode,Vo),
    Expls = [Node|Expls1],!,
    $pp_build_expls_feature(I,Decode,PrMode,_,Expls1).

$pp_decode_paths_feature([],[],_Decode,_PrMode,_Vo).
$pp_decode_paths_feature([Pair|Pairs],[Path|Paths],Decode,PrMode,Vo) :-
    Pair = [Gids,Sids],
    $pp_decode_gnodes_feature(Gids,GNodes,Decode,PrMode,Vg),
    $pp_decode_snodes_feature(Sids,SNodes,Decode,PrMode,Vs),
    get_prism_flag(log_scale,LogScale),
    ( PrMode == 0 ->
        Path = path(GNodes,SNodes)
    ; PrMode == 1 -> ( LogScale == on -> Vi is Vg + Vs ; Vi is Vg * Vs),
        Path = path(GNodes,SNodes,Vi)
    ; PrMode == 2 ->
        Path = path(GNodes,SNodes,Vo)
    ; PrMode == 4 -> ( LogScale == on -> Vi is Vg + Vs ; Vi is Vg * Vs),
        Path = path(GNodes,SNodes,[Vi,Vo])
    ),!,
    $pp_decode_paths_feature(Pairs,Paths,Decode,PrMode,Vo).

$pp_decode_gnodes_feature(Gids,GNodes,Decode,PrMode,V) :-
    get_prism_flag(log_scale,LogScale),
    ( LogScale == on -> V0 = 0.0 ; V0 = 1.0 ),
    $pp_decode_gnodes_feature(Gids,GNodes,Decode,PrMode,LogScale,V0,V).

$pp_decode_gnodes_feature([],[],_Decode,_PrMode,_LogScale,V,V) :- !.
$pp_decode_gnodes_feature([Gid|Gids],[GNode|GNodes],Decode,PrMode,LogScale,V0,V) :-
    ( Decode == 0 -> Gid = Label
    ; $pc_prism_goal_term(Gid,Label)
    ),
    ( PrMode == 0 -> GNode = Label
    ; $pp_get_gnode_probs(PrMode,Gid,Value),
      GNode = gnode(Label,Value),
      ( LogScale == on ->
        V1 is Value + V0
      ; V1 is Value * V0
      )
    ),!,
    $pp_decode_gnodes_feature(Gids,GNodes,Decode,PrMode,LogScale,V1,V).

$pp_decode_snodes_feature(Sids,SNodes,Decode,PrMode,V) :-
    get_prism_flag(log_scale,LogScale),
    ( LogScale == on -> V0 = 0.0 ; V0 = 1.0 ),
    $pp_decode_snodes_feature(Sids,SNodes,Decode,PrMode,LogScale,V0,V).

$pp_decode_snodes_feature([],[],_Decode,_PrMode,_LogScale,V,V) :- !.
$pp_decode_snodes_feature([Sid|Sids],[SNode|SNodes],Decode,PrMode,LogScale,V0,V) :-
    ( Decode == 0 -> Sid = Label
    ; $pc_prism_sw_ins_term(Sid,Label)
    ),
    ( PrMode == 0 -> SNode = Label
    ; PrMode == 1 ->
        $pp_get_snode_feature(PrMode,Sid,[Pi,F]),
        SNode = snode(Label,[Pi,F]),
        ( LogScale == on ->
          V1 is Pi * F + V0
        ; V1 is exp(Pi * F) * V0
        )
    ; PrMode == 4 ->
        $pp_get_snode_feature(PrMode,Sid,[PiF,E]),
        SNode = snode(Label,[PiF1,E]),
        ( LogScale = on ->
          V1 is PiF + V0,PiF1 = PiF
        ; V1 is exp(PiF) * V0,PiF1 is exp(PiF)
        )
    ; $pp_get_snode_feature(PrMode,Sid,Value),
        SNode = snode(Label,Value),
        ( LogScale == on ->
          V1 is Value + V0
        ; V1 is Value * V0
        )
    ),!,
    $pp_decode_snodes_feature(Sids,SNodes,Decode,PrMode,LogScale,V1,V).

$pp_get_snode_feature(1,Sid,[Pi,F]) :-
    $pc_get_snode_feature(Sid,F,Pi),!.
$pp_get_snode_feature(2,Sid,E) :-
    $pc_get_snode_expectation(Sid,E),!.
$pp_get_snode_feature(4,Sid,[PiF,E]) :-
    $pc_get_snode_feature(Sid,F,Pi),
    PiF is Pi * F,
    $pc_get_snode_expectation(Sid,E),!.
