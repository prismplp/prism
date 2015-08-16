%%%%
%%%%  Grammar-dependent part to be included in prefix_pcfg.psm --- rule.pl
%%%%
%%%%  Copyright (C) 2014-2015
%%%%    Sato Laboratory, Dept. of Computer Science,
%%%%    Tokyo Institute of Technology
%%%%

%% The clauses in this file specify a probabilistic context-free grammar:
%%
%%   s --> s s (prob. 0.4)
%%   s --> a (0.3)
%%   s --> b (0.3)
%%
%% In particular, its probabilistic behavior is specified by a multi-valued
%% switch declaration (values/3).

start_symbol(s).

nonterminal(X) :- not terminal(X).

terminal(a).
terminal(b).

values(s,[[s,s],[a],[b]],[0.4,0.3,0.3]).
