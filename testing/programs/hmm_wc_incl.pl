%%------------------------------------
%%  Utility part:

hmm_learn(N):-
   set_params,!,                % Set parameters manually
   get_samples(N,hmm(_),Gs),!,  % Get N samples
   learn(Gs).                   % Learn with the samples

set_params:-
   set_sw(init,   [0.9,0.1]),
   set_sw(tr(s0), [0.2,0.8]),
   set_sw(tr(s1), [0.8,0.2]),
   set_sw(out(s0),[0.5,0.5]),
   set_sw(out(s1),[0.6,0.4]).

%%  prism_main/1 is a special predicate for batch execution.
%%  The following command conducts learning from 50 randomly
%%  generated samples:
%%      > upprism hmm 50

prism_main([]):-hmm_learn(10).
prism_main([Arg]):-
   format("*****"),
   parse_atom(Arg,N),           % Convert an atom ('50') to a number (50)
   hmm_learn(N).                % Learn with N samples

%%  viterbi_states(Os,Ss) returns the most probable sequence Ss
%%  of state transitions for an output sequence Os.
%%
%%  | ?- viterbi_states([a,a,a,a,a,b,b,b,b,b],States).
%%
%%  States = [s0,s1,s0,s1,s0,s1,s0,s1,s0,s1,s0] ?

viterbi_states(Outputs,States):-
   viterbif(hmm(Outputs),_,E),
   viterbi_subgoals(E,E1),
   maplist(hmm(_,_,S,_),S,true,E1,States).
