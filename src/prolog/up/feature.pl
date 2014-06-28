%%--------------------------

$pp_require_crf_params(X,MsgID,Source) :-
    ( $pp_test_crf_params(X) -> true
    ; $pp_raise_on_require([X],MsgID,Source,$pp_error_distribution)
    ).

$pp_test_crf_params(X) :-
    ( $pp_test_fixed_size_crf_params(X)
    ; $pp_test_variable_size_distribution(X)
    ).

$pp_test_fixed_size_crf_params(X) :-
    ground(X),
    ( $pp_test_numbers(X)
    ; $pp_test_crf_params_plus(X)
    ; $pp_test_ratio(X)
    ).

$pp_test_crf_params_plus(X) :-
    $pp_expr_to_list('+',X,Ps),
    length(Ps,L),
    L > 1,!,
    $pp_test_probabilities(Ps).

%%--------------------------

expand_fprobs(Dist,N,Probs) :-
    $pp_expand_fprobs(Dist,N,Probs,expand_fprobs/3).

$pp_expand_fprobs(Dist,N,Probs,Source) :-
    $pp_require_crf_params(Dist,$msg(0200),Source),
    $pp_require_positive_integer(N,$msg(0204),Source),
    $pp_spec_to_ratio(Dist,N,Ratio,Source),
    $pp_check_expanded_prob_size(Ratio,N,Source),
    $pp_normalize_ratio_crf(Dist,Ratio,Probs).

$pp_normalize_ratio_crf(Dist,Ratio,Probs) :-
    ( Dist = [_|_] -> Probs = Ratio
    ; Dist = (_+_) -> Probs = Ratio
    ; $pp_normalize_ratio(Ratio,Probs)
    ).

%%-------------------------

set_sw_w(Sw) :- set_sw_a(Sw).
set_sw_w(Sw,Spec) :- set_sw_a(Sw,Spec).

set_sw_all_w :- set_sw_all_a.
set_sw_all_w(Sw) :- set_sw_all_a(Sw).

set_sw_w_all          :- set_sw_all_w.
set_sw_w_all(Sw)      :- set_sw_all_w(Sw).
set_sw_w_all(Sw,Spec) :- set_sw_all_w(Sw,Spec).

show_sw_w :- show_sw_a.
show_sw_w(Sw) :- show_sw_a(Sw).
