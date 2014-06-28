%%%% CRF-Viterbi wrappers

crfviterbi(G) :-
    $pp_crfviterbi_wrapper(crfviterbi(G)).
crfviterbi(G,P) :-
    $pp_crfviterbi_wrapper(crfviterbi(G,P)).
crfviterbif(G) :-
    $pp_crfviterbi_wrapper(crfviterbif(G)).
crfviterbif(G,P,V) :-
    $pp_crfviterbi_wrapper(crfviterbif(G,P,V)).
crfviterbit(G) :-
    $pp_crfviterbi_wrapper(crfviterbit(G)).
crfviterbit(G,P,T) :-
    $pp_crfviterbi_wrapper(crfviterbit(G,P,T)).
n_crfviterbi(N,G) :-
    $pp_crfviterbi_wrapper(n_crfviterbi(N,G)).
n_crfviterbi(N,G,P) :-
    $pp_crfviterbi_wrapper(n_crfviterbi(N,G,P)).
n_crfviterbif(N,G) :-
    $pp_crfviterbi_wrapper(n_crfviterbif(N,G)).
n_crfviterbif(N,G,V) :-
    $pp_crfviterbi_wrapper(n_crfviterbif(N,G,V)).
n_crfviterbit(N,G) :-
    $pp_crfviterbi_wrapper(n_crfviterbit(N,G)).
n_crfviterbit(N,G,T) :-
    $pp_crfviterbi_wrapper(n_crfviterbit(N,G,T)).
crfviterbig(G) :-
    $pp_crfviterbi_wrapper(crfviterbig(G)).
crfviterbig(G,P) :-
    $pp_crfviterbi_wrapper(crfviterbig(G,P)).
crfviterbig(G,P,V) :-
    $pp_crfviterbi_wrapper(crfviterbig(G,P,V)).
n_crfviterbig(N,G) :-
    $pp_crfviterbi_wrapper(n_crfviterbig(N,G)).
n_crfviterbig(N,G,P) :-
    $pp_crfviterbi_wrapper(n_crfviterbig(N,G,P)).
n_crfviterbig(N,G,P,V) :-
    $pp_crfviterbi_wrapper(n_crfviterbig(N,G,P,V)).

$pp_crfviterbi_wrapper(Pred0) :-
    Suffix = '_p',
    Pred0 =.. [Name0|Args],
    atom_concat(Name0,Suffix,Name1),
    Pred1 =.. [Name1|Args],!,
    call(Pred1).  % do not add cut here (n_viterbig is non-deterministic)

%%%% Viterbi routine with C interface
%%
%% viterbi_p(G) :- print the Viterbi prob
%% viterbi_p(G,P) :- output the Viterbi prob
%% viterbif_p(G) :- print the Viterbi path and the Viterbi prob
%% viterbif_p(G,P,VPath) :- output the Viterbi path and the Viterbi prob
%%
%% VPath is a list of node(G,Paths), where Paths is a list of
%% path(Gs,Sws), where Gs are subgoals of G and Sws are switches.
%%
%% Usually in VPath, node(msw(Sw,V),[]) is omitted, but optionally
%% it can be included in VPath.

% Main routine:

% viterbi family:

crfviterbi_p(Goal) :-
    crfviterbif_p(Goal,Pmax,_),
    $pp_print_crfviterbi_prob(Pmax).

crfviterbi_p(Goal,Pmax) :-
    crfviterbif_p(Goal,Pmax,_).

% viterbif family:

crfviterbif_p(Goal) :-
    crfviterbif_p(Goal,Pmax,VNodeL),
    format("~n",[]),
    print_graph(VNodeL,[lr('<=')]),
    $pp_print_crfviterbi_prob(Pmax).

crfviterbif_p(Goal,Pmax,VNodeL) :-
    $pp_require_tabled_probabilistic_atom(Goal,$msg(0006),viterbif_p/3),
    ( Goal = msw(I,_) ->
        $pp_require_ground(I,$msg(0101),viterbif_p/3),
        $pp_require_switch_outcomes(I,$msg(0102),viterbif_p/3)
    ; true
    ),
    $pp_crfviterbif_p(Goal,Pmax,VNodeL).

$pp_crfviterbif_p(Goal,Pmax,VNodeL) :-
    $pp_clean_infer_stats,
    cputime(T0),
    $pp_crfviterbi_core(Goal,Pmax,VNodeL),
    cputime(T1),
    $pp_garbage_collect,
    InfTime is T1 - T0,
    $pp_assert_viterbi_stats1(InfTime),!.

% viterbit family:

crfviterbit_p(Goal) :-
    crfviterbit_p(Goal,Pmax,VTreeL),
    format("~n",[]),
    print_tree(VTreeL),
    $pp_print_crfviterbi_prob(Pmax).

crfviterbit_p(Goal,Pmax,VTreeL) :-
    $pp_require_tabled_probabilistic_atom(Goal,$msg(0006),viterbit_p/3),
    $pp_crfviterbif_p(Goal,Pmax,VNodeL),
    viterbi_tree(VNodeL,VTreeL).

% viterbig family:

crfviterbig_p(Goal) :-
    ( ground(Goal) -> crfviterbi_p(Goal)
    ; crfviterbig_p(Goal,_,_)
    ).

crfviterbig_p(Goal,Pmax) :-
    ( ground(Goal) -> crfviterbi_p(Goal,Pmax)
    ; crfviterbig_p(Goal,Pmax,_)
    ).

crfviterbig_p(Goal,Pmax,VNodeL) :-
    $pp_require_tabled_probabilistic_atom(Goal,$msg(0006),viterbig_p/3),
    ( Goal = msw(I,_) ->
        $pp_require_ground(I,$msg(0101),viterbif_p/3),
        $pp_require_switch_outcomes(I,$msg(0102),viterbig_p/3)
    ; true
    ),
    $pp_crfviterbig_p(Goal,Pmax,VNodeL).

$pp_crfviterbig_p(Goal,Pmax,VNodeL) :-
    $pp_clean_infer_stats,
    cputime(T0),
    $pp_crfviterbi_core(Goal,Pmax,VNodeL),
    ( ground(Goal) -> true
    ; VNodeL = [node(_,[path([Goal1],[])])|_] -> Goal = Goal1
    ; VNodeL = [node(_,[path([],[SwIns])])|_] -> Goal = SwIns
    ),
    cputime(T1),
    InfTime is T1 - T0,
    $pp_garbage_collect,
    $pp_assert_viterbi_stats1(InfTime),!.

%% Common routine:

$pp_print_crfviterbi_prob(Pmax) :-
    format("~nCRF-Viterbi_P = ~15f~n",[Pmax]).

$pp_crfviterbi_core(Goal,Pmax,VNodeL) :-
    Goal = msw(I,V),!,
    $pp_require_ground(I,$msg(0101),$pp_viterbi_core/3),
    $pp_require_switch_outcomes(I,$msg(0102),$pp_viterbi_core/3),
    $pp_init_tables_aux,
    $pp_clean_graph_stats,
    $pp_init_tables_if_necessary,!,
    ( ground(V) -> V = VCp ; copy_term(V,VCp) ),
    $pp_create_dummy_goal(DummyGoal),
    DummyBody = ($prism_expl_msw(I,VCp,Sid),
                 $pc_prism_goal_id_register(DummyGoal,Hid),
                 $prism_eg_path(Hid,[],[Sid])),
    Prog = [pred(DummyGoal,0,_,_,tabled(_,_,_,_),[(DummyGoal:-DummyBody)])],
    $pp_consult_preds_cond([],Prog),!,
    cputime(T1),
    $pp_find_explanations(DummyGoal),
    cputime(T2),
    $pp_compute_crfviterbi_p(DummyGoal,Pmax,[node(DummyGoal,Paths)|VNodeL0]),!,
    cputime(T3),
    VNodeL = [node(msw(I,V),Paths)|VNodeL0],
    $pc_import_graph_stats(NSubgraphs,NGoalNodes,NSwNodes,AvgShared),
    $pp_assert_graph_stats(NSubgraphs,NGoalNodes,NSwNodes,AvgShared),
    SearchTime  is T2 - T1,
    NumCompTime is T3 - T2,
    $pp_assert_viterbi_stats2(SearchTime,NumCompTime),!.

$pp_crfviterbi_core(Goal,Pmax,VNodeL) :-
    ground(Goal),!,
    $pp_init_tables_aux,
    $pp_clean_graph_stats,
    $pp_init_tables_if_necessary,!,
    cputime(T1),
    $pp_find_explanations(Goal),
    cputime(T2),
    $pp_compute_crfviterbi_p(Goal,Pmax,VNodeL),!,
    cputime(T3),
    $pc_import_graph_stats(NSubgraphs,NGoalNodes,NSwNodes,AvgShared),
    $pp_assert_graph_stats(NSubgraphs,NGoalNodes,NSwNodes,AvgShared),
    SearchTime  is T2 - T1,
    NumCompTime is T3 - T2,
    $pp_assert_viterbi_stats2(SearchTime,NumCompTime),!.

$pp_crfviterbi_core(Goal,Pmax,VNodeL) :-
    $pp_init_tables_aux,
    $pp_clean_graph_stats,
    $pp_init_tables_if_necessary,!,
    copy_term(Goal,GoalCp),
    ( $pp_trans_one_goal(GoalCp,CompGoal) -> BodyGoal = CompGoal
    ; BodyGoal = (savecp(CP),Depth=0,
                  $pp_expl_interp_goal(GoalCp,Depth,CP,[],_,[],_,[],_,[],_))
    ),
    $pp_create_dummy_goal(DummyGoal),
    DummyBody = (BodyGoal,
                 $pc_prism_goal_id_register(GoalCp,GId),
                 $pc_prism_goal_id_register(DummyGoal,HId),
                 $prism_eg_path(HId,[GId],[])),
    Prog = [pred(DummyGoal,0,_Mode,_Delay,tabled(_,_,_,_),
                 [(DummyGoal:-DummyBody)])],
    $pp_consult_preds_cond([],Prog),!,
    cputime(T1),
    $pp_find_explanations(DummyGoal),
    cputime(T2),
    $pp_compute_crfviterbi_p(DummyGoal,Pmax,[node(DummyGoal,Paths)|VNodeL0]),!,
    cputime(T3),
    VNodeL = [node(Goal,Paths)|VNodeL0],
    $pc_import_graph_stats(NSubgraphs,NGoalNodes,NSwNodes,AvgShared),
    $pp_assert_graph_stats(NSubgraphs,NGoalNodes,NSwNodes,AvgShared),
    SearchTime  is T2 - T1,
    NumCompTime is T3 - T2,
    $pp_assert_viterbi_stats2(SearchTime,NumCompTime),!.

% Sws = [sw(Id,Instances,Probs,PseudoCs,Fixed,FixedH),...]
$pp_compute_crfviterbi_p(Goal,Pmax,VNodeL) :-
    $pp_export_sw_info,
    $pc_prism_goal_id_get(Goal,Gid),
    $pc_compute_crfviterbi(Gid,EGs,EGPaths,ESwPaths,Pmax),
    $pp_decode_viterbi_path(EGs,EGPaths,ESwPaths,VNodeL),!.

%%%%
%%%%  Top-N Viterbi
%%%%
%%%% n_viterbi_p(N,G) :- print the top-N Viterbi probs
%%%% n_viterbi_p(N,G,Ps) :- output the top-N Viterbi probs
%%%% n_viterbif_p(N,G) :- print the top-N Viterbi paths and the corresponding
%%%%                     Viterbi probs
%%%% n_viterbif_p(N,G,VPathL) :- output the list of top-N Viterbi paths and
%%%%                            the corresponding Viterbi probs
%%%%

% n_viterbi family

n_crfviterbi_p(N,Goal) :-
    n_crfviterbif_p(N,Goal,VPathL),
    ( member(v_expl(J,Pmax,_),VPathL),
      $pp_print_n_crfviterbi(J,Pmax),
      fail
    ; true
    ).

n_crfviterbi_p(N,Goal,Ps) :-
    n_crfviterbif_p(N,Goal,VPathL),!,
    findall(Pmax,member(v_expl(_,Pmax,_),VPathL),Ps).

% n_viterbif family

n_crfviterbif_p(N,Goal) :-
    n_crfviterbif_p(N,Goal,VPathL),!,
    $pp_print_n_crfviterbif(VPathL).

n_crfviterbif_p(N,Goal,VPathL) :-
    $pp_require_positive_integer(N,$msg(1400),n_viterbif_p/3),
    $pp_require_tabled_probabilistic_atom(Goal,$msg(0006),n_viterbif_p/3),
    $pp_n_crfviterbif_p(N,Goal,VPathL).

$pp_n_crfviterbif_p(N,Goal,VPathL) :-
    $pp_clean_infer_stats,
    cputime(T0),
    $pp_n_crfviterbi_p_core(N,Goal,VPathL),
    cputime(T1),
    InfTime is T1 - T0,
    $pp_garbage_collect,
    $pp_assert_viterbi_stats1(InfTime),!.

% n_viterbit family

n_crfviterbit_p(N,Goal) :-
    n_crfviterbif_p(N,Goal,VPathL),!,
    $pp_print_n_crfviterbit(VPathL).

n_crfviterbit_p(N,Goal,VPathL) :-
    n_crfviterbif_p(N,Goal,VPathL0),!,
    $pp_build_n_viterbit(VPathL0,VPathL).

%%%% 
%%%% $pp_n_viterbig_p(N,Goal) :- the same as $pp_n_viterbig_p(N,Goal,_,_)
%%%% $pp_n_viterbig_p(N,Goal,Pmax) :- the same as $pp_n_viterbig_p(N,Goal,Pmax,_)
%%%% $pp_n_viterbig_p(N,Goal,Pmax,VNodeL) :-
%%%%      if Goal is not ground, unify Goal with the first element in the K-th
%%%%      Viterbi path VNodeL (K=0,1,2,...,(N-1) on backtracking). Pmax is the
%%%%      probability of VNodeL.
%%%%

n_crfviterbig_p(N,Goal) :-
    ( ground(Goal) -> n_crfviterbi_p(N,Goal)
    ; n_crfviterbig_p(N,Goal,_,_)
    ).

n_crfviterbig_p(N,Goal,Pmax) :-
    ( ground(Goal) -> n_crfviterbi_p(N,Goal,Ps),!,member(Pmax,Ps)
    ; n_crfviterbig_p(N,Goal,Pmax,_)
    ).

n_crfviterbig_p(N,Goal,Pmax,VNodeL) :-
    $pp_require_positive_integer(N,$msg(1400),n_viterbi_p/3),
    $pp_require_tabled_probabilistic_atom(Goal,$msg(0006),n_viterbi_p/3),
    $pp_n_crfviterbig_p(N,Goal,Pmax,VNodeL).

$pp_n_crfviterbig_p(N,Goal,Pmax,VNodeL) :-
    $pp_clean_infer_stats,
    cputime(T0),
    $pp_n_crfviterbi_p_core(N,Goal,VPathL),!,
    cputime(T1),
    InfTime is T1 - T0,
    $pp_garbage_collect,
    $pp_assert_viterbi_stats1(InfTime),!,
    ( ground(Goal) -> member(v_expl(J,Pmax,VNodeL),VPathL)
    ; Goal = msw(_,_) ->
        member(v_expl(J,Pmax,VNodeL),VPathL),
        VNodeL = [node(_,[path([],[SwIns])])|_],
        Goal = SwIns
    ; % else
        member(v_expl(J,Pmax,VNodeL),VPathL),
        VNodeL = [node(_,[path([Goal1],[])])|_],
        Goal = Goal1
    ).

%% Common routines:

$pp_print_n_crfviterbi(J,Pmax) :-
      format("#~w: CRF-Viterbi_P = ~15f~n",[J,Pmax]).

$pp_print_n_crfviterbif([]).
$pp_print_n_crfviterbif([v_expl(J,Pmax,VNodeL)|VPathL]) :-
    format("~n#~w~n",[J]),
    print_graph(VNodeL,[lr('<=')]),
    format("~nCRF-Viterbi_P = ~15f~n",[Pmax]),!,
    $pp_print_n_crfviterbif(VPathL).

$pp_print_n_crfviterbit([]).
$pp_print_n_crfviterbit([v_expl(J,Pmax,VNodeL)|VPathL]) :-
    format("~n#~w~n",[J]),
    viterbi_tree(VNodeL,VTreeL),
    print_tree(VTreeL),
    $pp_print_crfviterbi_prob(Pmax),!,
    $pp_print_n_crfviterbit(VPathL).

$pp_n_crfviterbi_p_core(N,Goal,VPathL) :-
    Goal = msw(I,V),!,
    $pp_require_ground(I,$msg(0101),$pp_n_viterbi_p_core/3),
    $pp_require_switch_outcomes(I,$msg(0102),$pp_n_viterbi_p_core/3),
    $pp_init_tables_aux,
    $pp_clean_graph_stats,
    $pp_init_tables_if_necessary,!,
    ( ground(V) -> V = VCp ; copy_term(V,VCp) ),
    $pp_create_dummy_goal(DummyGoal),
    DummyBody = ($prism_expl_msw(I,VCp,Sid),
                 $pc_prism_goal_id_register(DummyGoal,Hid),
                 $prism_eg_path(Hid,[],[Sid])),
    Prog = [pred(DummyGoal,0,_Mode,_Delay,tabled(_,_,_,_),
                 [(DummyGoal:-DummyBody)])],
    $pp_consult_preds_cond([],Prog),!,
    cputime(T1),
    $pp_find_explanations(DummyGoal),
    cputime(T2),
    $pp_compute_n_crfviterbi_p(N,DummyGoal,VPathL0),!,
    cputime(T3),
    $pp_replace_dummy_goal(Goal,DummyGoal,VPathL0,VPathL),
    $pc_import_graph_stats(NSubgraphs,NGoalNodes,NSwNodes,AvgShared),
    $pp_assert_graph_stats(NSubgraphs,NGoalNodes,NSwNodes,AvgShared),
    SearchTime  is T2 - T1,
    NumCompTime is T3 - T2,
    $pp_assert_viterbi_stats2(SearchTime,NumCompTime),!.

$pp_n_crfviterbi_p_core(N,Goal,VPathL) :-
    ground(Goal),!,
    $pp_init_tables_aux,
    $pp_clean_graph_stats,
    $pp_init_tables_if_necessary,!,
    cputime(T1),
    $pp_find_explanations(Goal),
    cputime(T2),
    $pp_compute_n_crfviterbi_p(N,Goal,VPathL),!,
    cputime(T3),
    $pc_import_graph_stats(NSubgraphs,NGoalNodes,NSwNodes,AvgShared),
    $pp_assert_graph_stats(NSubgraphs,NGoalNodes,NSwNodes,AvgShared),
    SearchTime  is T2 - T1,
    NumCompTime is T3 - T2,
    $pp_assert_viterbi_stats2(SearchTime,NumCompTime),!.

$pp_n_crfviterbi_p_core(N,Goal,VPathL) :-
    $pp_init_tables_aux,
    $pp_clean_graph_stats,
    $pp_init_tables_if_necessary,!,
    copy_term(Goal,GoalCp),
    ( $pp_trans_one_goal(GoalCp,CompGoal) -> BodyGoal = CompGoal
    ; BodyGoal = (savecp(CP),Depth=0,
                  $pp_expl_interp_goal(GoalCp,Depth,CP,[],_,[],_,[],_,[],_))
    ),
    $pp_create_dummy_goal(DummyGoal),
    DummyBody = (BodyGoal,
                 $pc_prism_goal_id_register(GoalCp,GId),
                 $pc_prism_goal_id_register(DummyGoal,HId),
                 $prism_eg_path(HId,[GId],[])),
    Prog = [pred(DummyGoal,0,_Mode,_Delay,tabled(_,_,_,_),
                 [(DummyGoal:-DummyBody)])],
    $pp_consult_preds_cond([],Prog),!,
    cputime(T1),
    $pp_find_explanations(DummyGoal),
    cputime(T2),
    $pp_compute_n_crfviterbi_p(N,DummyGoal,VPathL0),!,
    cputime(T3),
    $pp_replace_dummy_goal(Goal,DummyGoal,VPathL0,VPathL),
    $pc_import_graph_stats(NSubgraphs,NGoalNodes,NSwNodes,AvgShared),
    $pp_assert_graph_stats(NSubgraphs,NGoalNodes,NSwNodes,AvgShared),
    SearchTime  is T2 - T1,
    NumCompTime is T3 - T2,
    $pp_assert_viterbi_stats2(SearchTime,NumCompTime),!.

$pp_compute_n_crfviterbi_p(N,Goal,VPathL) :-
    $pp_export_sw_info,!,
    $pc_prism_goal_id_get(Goal,Gid),
    $pc_compute_n_crfviterbi(N,Gid,VPathL0),
    $pp_build_n_viterbi_path(VPathL0,VPathL),!.
