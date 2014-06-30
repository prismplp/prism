
nonlin_prob_check(Goal) :-
  in_prob_m(1,Goal,P),
  format("Probability is ~w\n",P).
nonlin_prob_check(Goal,P) :-
  in_prob_m(1,Goal,P).

nonlin_prob(Goal) :-
  in_prob_m(0,Goal,P),
  format("Probability is ~w\n",P).
nonlin_prob(Goal,P) :-
  in_prob_m(0,Goal,P).

nonlin_prob_m(Mode,Goal) :-
  in_prob_m(Mode,Goal,P),
  format("Probability is ~w\n",P).
nonlin_prob_m(Mode,Goal,P) :-
  probefi(Goal,_),
  $pc_infix(Mode,P).



