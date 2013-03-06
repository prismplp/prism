%-------------[ Task : Metropolis-Hastings sampler ]----------------
% Usage: "mcmc"
%        "mcmc(Gs)"
%        "mcmc(Gs,Option)"

:- dynamic $pd_mcmc_free_energy/1.
:- dynamic $pd_viterbi_ranking/1.   % only supported informally

mcmc(Gs) :- mcmc(Gs,[]).

mcmc(Gs,Opts) :-                    % sample P(E_all|Gs) Max times
    $pp_mcmc_check_goals(Gs,mcmc/2),
    $pp_proc_mcmc_opts(Opts,[EndStep,BurnIn,Skip,Trace0,PostParam0],mcmc/2),
    $pp_clean_mcmc_sample_stats,
    $pp_mcmc_message(_,_,MsgC,MsgT,MsgM),
    $pc_set_mcmc_message(MsgC),
    $pp_format_if(MsgM,"~n<<Running VBEM for the initial state>>~n"),
    $pp_mcmc_vb_learn(both,Gs), % theta* affects init state & sampled value
    $pc_import_graph_stats(NSubgraphs,NGoalNodes,NSwNodes,AvgShared),
    $pp_mcmc_replace_dummy_goals(Gs,DummyGoals),
    length(Gs,NumG),
    $pp_conv_trace_opts(Trace0,Trace,mcmc/2),
    $pp_conv_postparam_opts(PostParam0,PostParam,mcmc/2),
    $pp_format_if(MsgM,"~n<<Running MCMC sampling>>~n"),
    cputime(Start),
    $pc_mcmc_sample(DummyGoals,NumG,EndStep,BurnIn,Skip,Trace,PostParam),
    cputime(End),
    $pp_assert_graph_stats(NSubgraphs,NGoalNodes,NSwNodes,AvgShared),
    $pp_assert_mcmc_sample_stats(Start,End,1000),
    $pp_print_mcmc_sample_stats_message(MsgT),!.

%% VB learning for obtaining the initial state
$pp_mcmc_vb_learn(Mode,Goals) :-
    $pp_learn_check_goals(Goals),
    $pp_mcmc_message(MsgS,MsgE,_,MsgT,_),
    $pp_clean_learn_stats,
    $pc_set_em_message(MsgE),
    cputime(Start),
    $pp_clean_dummy_goal_table,
    $pp_clean_graph_stats,
    $pp_init_tables_aux,
    $pp_init_tables_if_necessary,!,
    $pp_trans_goals(Goals,GoalCountPairs,AllGoals),!,
    global_set($pg_observed_facts,GoalCountPairs),
    cputime(StartExpl),
    $pp_print_search_progress(MsgS),
    $pp_find_explanations(AllGoals),!,
    $pp_print_num_goals(MsgS),
    cputime(EndExpl),
    statistics(table,[TableSpace,_]),
    $pp_export_sw_info,
    $pp_observed_facts(GoalCountPairs,GidCountPairs,
                       0,Len,0,NGoals,-1,FailRootIndex),
    $pp_check_failure_in_mcmc(Goals,FailRootIndex,mcmc/2),
    $pc_mcmc_prepare(GidCountPairs,Len,NGoals),
    cputime(StartEM),
    $pp_em(Mode,Output),
    cputime(EndEM),
    $pc_import_switch_stats(NSwitches,NSwVals),
    $pc_import_graph_stats(NSubgraphs,NGoalNodes,NSwNodes,AvgShared),
    $pp_assert_mcmc_free_energy(Output),
    cputime(End),
    $pp_assert_graph_stats(NSubgraphs,NGoalNodes,NSwNodes,AvgShared),
    $pp_assert_learn_stats(Mode,Output,NSwitches,NSwVals,TableSpace,
                           Start,End,StartExpl,EndExpl,StartEM,EndEM,1000),
    $pp_print_learn_stats_message(MsgT),!.

$pp_proc_mcmc_opts(Opts,[End,BurnIn,Skip,Trace,PostParam],Source) :-
    % default values
    get_prism_flag(mcmc_e,End0),
    get_prism_flag(mcmc_b,BurnIn0),
    get_prism_flag(mcmc_s,Skip0),
    Trace0     = none,
    PostParam0 = none,
    % parse the user options
    $pp_proc_opts(Opts,$pp_mcmc_option,
                  [End1,BurnIn1,Skip1,Trace1,PostParam1],
                  [End0,BurnIn0,Skip0,Trace0,PostParam0],
                  Source),
    % error check & additional processing:
    ( BurnIn1 > End1 ->
          $pp_raise_domain_error($msg(1501),[BurnIn1],[burn_in,BurnIn1],Source)
    ; true
    ),
    ( Skip1 > (End1 - BurnIn1) ->
          $pp_raise_domain_error($msg(1502),[Skip1],[skip,Skip1],Source)
    ; true
    ),
    End    = End1,
    BurnIn = BurnIn1,
    Skip   = Skip1,
    ( Trace1 = [TraceSw,TraceStep0] ->
        ( get_values0(TraceSw,_) -> 
            ( TraceStep0 = burn_in -> TraceStep = BurnIn
            ; TraceStep0 > End ->
                $pp_raise_domain_error($msg(1503),[TraceStep0],[trace_step,Skip1],Source)
            ; TraceStep = TraceStep0
            )
        ; $pp_raise_runtime_error($msg(1505),[TraceSw],switch_unavailable,Source)
        ),
      Trace = [TraceSw,TraceStep]
    ; Trace = none
    ),
    ( PostParam1 = [PParamSw,PParamStep0] ->
        ( get_values0(TraceSw,_) -> 
            ( PParamStep0 = burn_in -> PParamStep = BurnIn
            ; PParamStep0 > End ->
                $pp_raise_domain_error($msg(1503),[PParamStep0],[post_param_step,Skip1],Source)
            ; PParamStep = PParamStep0
            )
        ; $pp_raise_runtime_error($msg(1505),[PParamSw],switch_unavailable,Source)
        ),
      PostParam = [PParamSw,PParamStep]
    ; PostParam = none
    ),!.
   
% this should be called after VB learning
$pp_conv_trace_opts(Trace0,Trace,Source) :-
    ( Trace0 = [TraceSw,TraceStep] ->
        ( $pc_prism_sw_id_get(TraceSw,TraceSwID)
        ; $pp_raise_runtime_error($msg(1505),[TraceSw],switch_unavailable,Source)
        )
    ; TraceSwID = -1, TraceStep = -1
    ),
    Trace = [TraceSwID,TraceStep],!.

% this should be called after VB learning
$pp_conv_postparam_opts(PostParam0,PostParam,Source) :-
    ( PostParam0 = [PParamSw,PParamStep] ->
        ( $pc_prism_sw_id_get(PParamSw,PParamSwID)
        ; $pp_raise_runtime_error($msg(1505),[PParamSw],switch_unavailable,Source)
        )
    ; PParamSwID = -1, PParamStep = -1
    ),
    PostParam = [PParamSwID,PParamStep],!.

$pp_mcmc_option(end(N),1,N)                    :- integer(N),N>=0.
$pp_mcmc_option(burn_in(N),2,N)                :- integer(N),N>=0.
$pp_mcmc_option(skip(N),3,N)                   :- integer(N),N>=1.
$pp_mcmc_option(trace(Sw),4,[Sw,burn_in])      :- not(integer(Sw)).
$pp_mcmc_option(trace(Sw,N),4,[Sw,N])          :- integer(N),N>=1,not(integer(Sw)).
$pp_mcmc_option(post_param(Sw),5,[Sw,burn_in]) :- not(integer(Sw)).
$pp_mcmc_option(post_param(Sw,N),5,[Sw,N])     :- integer(N),N>=1,not(integer(Sw)).

$pp_assert_mcmc_free_energy(Output) :-
    Output = [_,FE],
    retractall($pd_mcmc_free_energy(_)),
    assert($pd_mcmc_free_energy(FE)),!.


%-------------[ Task : Estimate log-marginal-likelihood ]----------------
% Usage: "marg_mcmc"
%        "marg_mcmc([VFE,EML])"
%        "marg_mcmc_full(Gs)"
%        "marg_mcmc_full(Gs,Opts)"
%        "marg_mcmc_full(Gs,Opts,[VFE,EML])"

marg_mcmc:-
    marg_mcmc([VFE,EML]),
    format("~nFree energy = ~12f~n",[VFE]),
    format("Estimated log-marginal-likelihood = ~12f~n~n",[EML]).

marg_mcmc([VFE,EML]) :-
    $pp_mcmc_check_free_energy(VFE,marg_mcmc/1),
    $pp_clean_mcmc_marg_stats,
    cputime(Start),
    $pc_mcmc_marginal(EML),
    cputime(End),
    $pp_assert_mcmc_marg_stats(Start,End,1000),!.

marg_mcmc_full(Gs) :- marg_mcmc_full(Gs,[]).

marg_mcmc_full(Gs,Opts) :-
    marg_mcmc_full(Gs,Opts,[VFE,EML]),
    format("~nFree energy = ~12f~n",[VFE]),
    format("Estimated log-marginal-likelihood = ~12f~n~n",[EML]).

marg_mcmc_full(Gs,Opts,[VFE,EML]) :-
    $pp_proc_mcmc_opts(Opts,_,marg_mcmc_full/3), % just for checking the format
    $pp_mcmc_check_goals(Gs,marg_mcmc_full/3),
    $pp_clean_mcmc_marg_stats,
    $pp_disable_message(Msg),
    mcmc(Gs,Opts),
    $pp_enable_message(Msg),
    cputime(Start),
    $pp_mcmc_check_free_energy(VFE),
    $pc_mcmc_marginal(EML),
    cputime(End),
    $pp_assert_mcmc_marg_stats(Start,End,1000),!.

$pp_mcmc_check_free_energy(FE) :- $pd_mcmc_free_energy(FE).
$pp_mcmc_check_free_energy(FE,Source) :-
    ( $pd_mcmc_free_energy(FE) -> true
    ; $pp_raise_runtime_error($msg(1506),mcmc_unfinished,Source)
    ).


%-------------[ Task : Average log-marginal-likelihood ]----------------
% Usage: "ave_marg_mcmc(Iterate,Gs)"
%        "ave_marg_mcmc(Iterate,Gs,Opts)"
%        "ave_marg_mcmc(Iterate,Gs,Opts,[AvgEML,StdEML])"
%        "ave_marg_mcmc(Iterate,Gs,Opts,[AvgVFE,StdVFE],[AvgEML,StdEML])"

ave_marg_mcmc(Iterate,Gs) :-
    ave_marg_mcmc(Iterate,Gs,[]).

ave_marg_mcmc(Iterate,Gs,Opts) :-
    ave_marg_mcmc(Iterate,Gs,Opts,[AvgVFE,StdVFE],[AvgEML,StdEML]),
    format("~nIteration: ~d~n",[Iterate]),
    format("VFE: ave = ~12f, std = ~12f~n",[AvgVFE,StdVFE]),
    format("Estimated ML: ave = ~12f, std = ~12f~n~n",[AvgEML,StdEML]).    

ave_marg_mcmc(Iterate,Gs,Opts,[AvgEML,StdEML]) :-
    ave_marg_mcmc(Iterate,Gs,Opts,_,[AvgEML,StdEML]).

% some execution flags are forcedly modified while $pp_ave_marg_mcmc_aux/5 is called
ave_marg_mcmc(Iterate,Gs,Opts,[AvgVFE,StdVFE],[AvgEML,StdEML]) :-
    $pp_require_positive_integer(Iterate,$msg(1507),ave_marg_mcmc/5),
    $pp_proc_mcmc_opts(Opts,_,ave_marg_mcmc/5), % just for checking the format
    $pp_mcmc_check_goals(Gs,ave_marg_mcmc/5),
    $pp_clean_mcmc_marg_stats,
    numlist(1,Iterate,Ks),
    get_prism_flag(clean_table,CleanTable),
    $pp_ave_marg_mcmc_aux(Ks,Gs,Opts,VFEs,EMLs),
    set_prism_flag(clean_table,CleanTable),
    ( Iterate == 1 -> VFEs = [AvgVFE], StdVFE = 0, EMLs = [AvgEML], StdEML = 0
    ; avglist(VFEs,AvgVFE),
      stdlist(VFEs,StdVFE),
      avglist(EMLs,AvgEML),
      stdlist(EMLs,StdEML)
    ),!.

$pp_ave_marg_mcmc_aux([],_,_,[],[]).
$pp_ave_marg_mcmc_aux([K|Ks],Gs,Opts,[VFE|VFEs],[EML|EMLs]) :-
    marg_mcmc_full(Gs,Opts,[VFE,EML]),
    set_prism_flag(clean_table,off),
    format("[~d] VFE = ~12f, EML = ~12f~n",[K,VFE,EML]),!,
    $pp_ave_marg_mcmc_aux(Ks,Gs,Opts,VFEs,EMLs).


%----------------[ Task : Prediction by M-H sampling ]----------------
% Usage: "predict_mcmc(PGs,AnsL)" where Ans = [ViterbiG,ViterbiEG,LogP]
%        "predict_mcmc(M,PGs,AnsL)"
%        "predict_mcmc_full(OGs,PGs,AnsL)"
%        "predict_mcmc_full(OGs,Opts,PGs,AnsL)"
%        "predict_mcmc_full(OGs,Opts,M,PGs,AnsL)"
%
% use "print_predict_ans(AnsL)" to print tree of AnsL

predict_mcmc_full(OGs,PGs,AnsL) :-
    predict_mcmc_full(OGs,[],PGs,AnsL).

predict_mcmc_full(OGs,Opts,PGs,AnsL) :-
    get_prism_flag(rerank,M),
    predict_mcmc_full(OGs,Opts,M,PGs,AnsL).

predict_mcmc_full(OGs,Opts,M,PGs,AnsL) :-
    $pp_disable_message(Msg),
    mcmc(OGs,Opts),
    $pp_enable_message(Msg),
    predict_mcmc(M,PGs,AnsL).

predict_mcmc(PGs,AnsL) :-
    get_prism_flag(rerank,M),
    predict_mcmc(M,PGs,AnsL).

predict_mcmc(M,PGs,AnsL) :-
    $pp_require_positive_integer(M,$msg(1400),predict_mcmc/3),
    $pp_mcmc_check_goals(PGs,predict_mcmc/3),
    $pp_clean_mcmc_pred_stats,
    get_prism_flag(clean_table,CleanTable),
    set_prism_flag(clean_table,off),
    get_prism_flag(viterbi_mode,ViterbiMode),
    set_prism_flag(viterbi_mode,ml), % use ml, do not consider counts
    $pp_disable_message(Msg),
    cputime(Start),
    $pp_predict_mcmc_core(M,PGs,AnsL),
    cputime(End),
    set_prism_flag(clean_table,CleanTable),
    set_prism_flag(viterbi_mode,ViterbiMode),
    $pp_enable_message(Msg),
    $pp_assert_mcmc_pred_stats(Start,End,1000),
    garbage_collect,!.

$pp_predict_mcmc_core(M,Goals,AnsL) :-
    $pp_clean_dummy_goal_table,
    $pp_init_tables_aux,
    $pp_init_tables_if_necessary,!,
    $pp_trans_goals(Goals,GoalCountPairs,AllGoals),!,
    $pp_find_explanations(AllGoals),!,
    $pp_export_sw_info,
    $pp_observed_facts(GoalCountPairs,GidCountPairs,
                       0,Len,0,NGoals,-1,FailRootIndex),
    $pp_check_failure_in_mcmc(Goals,FailRootIndex,predict_mcmc/3),
    $pc_mcmc_prepare(GidCountPairs,Len,NGoals),
    $pp_mcmc_replace_dummy_goals(Goals,DummyGoals),
    $pc_mcmc_predict(DummyGoals,NGoals,M,AnsL0),
    $pp_mcmc_decode_ans(Goals,DummyGoals,AnsL0,AnsL,VRankLs),
    $pp_assert_mcmc_viterbi_ranking(VRankLs),!.

%% FIXME: isn't it better to remove?
$pp_assert_mcmc_viterbi_ranking(VRankLs) :-
    retractall($pd_viterbi_ranking(_)),
    assert($pd_viterbi_ranking(VRankLs)).

$pp_mcmc_replace_dummy_goals([],[]).
$pp_mcmc_replace_dummy_goals([Goal|Goals],[DummyGoal|DummyGoals]) :-
    ( Goal = msw(I,V) ->
        $pd_dummy_goal_table(DummyGoal,msw(I,V))
    ; ground(Goal) ->
        DummyGoal = Goal
    ; % else
        $pd_dummy_goal_table(DummyGoal,Goal)
    ),!,
    $pp_mcmc_replace_dummy_goals(Goals,DummyGoals).

$pp_mcmc_check_goals([],_).
$pp_mcmc_check_goals([Goal|Goals],Source) :-
    $pp_require_tabled_probabilistic_atom(Goal,$msg(0006),Source),!,
    $pp_mcmc_check_goals(Goals,Source).

$pp_mcmc_decode_ans([],[],[],[],[]).
$pp_mcmc_decode_ans([Goal|Goals],
                    [DummyGoal|DummyGoals],
                    [ans(Ans,LogP,VRankL)|AnsL0],
                    [[VG,VNodeL,LogP]|AnsL],[VRankL|VRankLs]) :-
    $pp_build_n_viterbi_path(Ans,VPathL0),
    $pp_replace_dummy_goal(Goal,DummyGoal,VPathL0,VPathL),
    ( ground(Goal) -> member(v_expl(J,Pmax,VNodeL),VPathL),VNodeL = [node(VG,_)|_]
    ; Goal = msw(_,_) ->
        member(v_expl(J,Pmax,VNodeL),VPathL),
        VNodeL = [node(VG,[path([],[SwIns])])|_],
        Goal = SwIns
    ; % else
        member(v_expl(J,Pmax,VNodeL),VPathL),
        VNodeL = [node(VG,[path([Goal1],[])])|_],
        Goal = Goal1
    ),!,
    $pp_mcmc_decode_ans(Goals,DummyGoals,AnsL0,AnsL,VRankLs).


%----------------[ Task : Compute exaxt log marginal likelihood ]----------------
% Usage: "marg_exact(Gs,LML)"
%        "marg_exact(Gs)"

marg_exact(Gs) :-
    marg_exact(Gs,LML),
    format("Exact log-marginal-likelihood = ~f~n",[LML]).

marg_exact(Gs,LML) :-
    $pp_mcmc_check_goals(Gs,marg_exact/2),
    $pp_clean_mcmc_exact_stats,
    cputime(Start),
    $pp_clean_dummy_goal_table,
    $pp_init_tables_aux,
    $pp_init_tables_if_necessary,!,
    $pp_trans_goals(Gs,GoalCountPairs,AllGoals),!,
    $pp_disable_message(Msg),
    $pp_find_explanations(AllGoals),!,
    $pp_export_sw_info,
    $pp_observed_facts(GoalCountPairs,GidCountPairs,
                       0,Len,0,NGoals,-1,FailRootIndex),
    $pp_check_failure_in_mcmc(Gs,FailRootIndex,marg_exact/2),
    $pc_mcmc_prepare(GidCountPairs,Len,NGoals),
    $pp_mcmc_replace_dummy_goals(Gs,DummyGoals),
    $pc_exact_marginal(DummyGoals,NGoals,LML),
    cputime(End),
    $pp_enable_message(Msg),
    $pp_assert_mcmc_exact_stats(Start,End,1000),!.


%-----------------------------------------------------------------------------------
% Miscellaneous routines

$pp_check_failure_in_mcmc(Gs,FailRootIndex,Source) :-
    ( FailRootIndex >= 0 ->
        $pp_raise_runtime_error($msg(1500),[Gs],failure_in_mcmc,Source)
    ; true
    ).

%-----------------------------------------------------------------------------------
% Statistics-related routines

$pp_clean_mcmc_sample_stats :- retractall($ps_mcmc_sample_time(_)).
$pp_clean_mcmc_marg_stats   :- retractall($ps_mcmc_marg_time(_)).
$pp_clean_mcmc_pred_stats   :- retractall($ps_mcmc_pred_time(_)).
$pp_clean_mcmc_exact_stats  :- retractall($ps_mcmc_exact_time(_)).


$pp_assert_mcmc_sample_stats(Start,End,UnitsPerSec) :-
    Time is (End - Start) / UnitsPerSec,
    assertz($ps_mcmc_sample_time(Time)).

$pp_assert_mcmc_marg_stats(Start,End,UnitsPerSec) :-
    Time is (End - Start) / UnitsPerSec,
    assertz($ps_mcmc_marg_time(Time)).

$pp_assert_mcmc_pred_stats(Start,End,UnitsPerSec) :-
    Time is (End - Start) / UnitsPerSec,
    assertz($ps_mcmc_pred_time(Time)).

$pp_assert_mcmc_exact_stats(Start,End,UnitsPerSec) :-
    Time is (End - Start) / UnitsPerSec,
    assertz($ps_mcmc_exact_time(Time)).


$pp_print_mcmc_sample_stats_message(Msg) :-
    ( Msg == 0 -> true
    ; format("Statistics on MCMC sampling:~n",[]),
      ( $pp_print_mcmc_sample_stats_message_aux,fail ; true ),nl
    ).
$pp_print_mcmc_sample_stats_message_aux :-
    $ps_mcmc_sample_time(L), format("~tMCMC sampling time: ~3f~n",[L]).

$pp_print_mcmc_marg_stats_message(Msg) :-
    ( Msg == 0 -> true
    ; format("Statistics on estimated log-marginal-likelihood:~n",[]),
      ( $pp_print_mcmc_marg_stats_message_aux,fail ; true ),nl
    ).
$pp_print_mcmc_marg_stats_message_aux :-
    $ps_mcmc_marg_time(L), format("~tTime for estimated log-marginal-likelihood: ~3f~n",[L]).

$pp_print_mcmc_pred_stats_message(Msg) :-
    ( Msg == 0 -> true
    ; format("Statistics on MCMC prediction:~n",[]),
      ( $pp_print_mcmc_pred_stats_message_aux,fail ; true ),nl
    ).
$pp_print_mcmc_pred_stats_message_aux :-
    $ps_mcmc_pred_time(L), format("~tTime for MCMC prediction: ~3f~n",[L]).

$pp_print_mcmc_exact_stats_message(Msg) :-
    ( Msg == 0 -> true
    ; format("Statistics on exact log-marginal-likelihood:~n",[]),
      ( $pp_print_mcmc_exact_stats_message_aux,fail ; true ),nl
    ).
$pp_print_mcmc_exact_stats_message_aux :-
    $ps_mcmc_exact_time(L), format("~tTime for exact log-marginal-likelihood: ~3f~n",[L]).

%-----------------------------------------------------------------------------------
% Message control

$pp_print_search_progress(MsgS) :-
    ( MsgS == 0 -> global_set($pg_num_goals,-1)
    ; global_set($pg_num_goals,0)
    ).

$pp_disable_message(Msg) :-
    $pp_print_search_progress(0),
    get_prism_flag(mcmc_message,Msg),
    set_prism_flag(mcmc_message,none).

$pp_enable_message(Msg) :-
    set_prism_flag(mcmc_message,Msg).
