nonterminal(X):-not terminal(X).

terminal(a).
terminal(b).

%values(s,[[s,s],[a]]).
values(s,[[s,s],[a],[b]],[0.4,0.3,0.3]).
