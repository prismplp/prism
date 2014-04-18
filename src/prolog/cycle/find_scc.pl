% Utility functions for manipulating the stack
$pp_scc_pop_until_less([Id|P],CTarget,PreTable,[Id|P]) :-
  bigarray_get(PreTable,Id,C),
  C =< CTarget.
$pp_scc_pop_until_less([_|P],CTarget,PreTable,P1) :-
  $pp_scc_pop_until_less(P,CTarget,PreTable,P1).

$pp_scc_pop_until_found(S,NTarget,S1,Popped) :-
  $pp_scc_pop_until_found(S,NTarget,S1,Popped,[]).
$pp_scc_pop_until_found([N|S],N,S,[N|Popped],Popped).
$pp_scc_pop_until_found([N|S],NTarget,S1,Popped,PoppedAcc) :-
  $pp_scc_pop_until_found(S,NTarget,S1,Popped,[N|PoppedAcc]).

% Wrapper for finding strongly connected components
$pp_find_scc(G,Components,CompTable) :-
  % Initialize arrays
  bigarray_length(G,Size),
  new_bigarray(PreTable,Size),
  new_bigarray(CompTable,Size),
  % Call Gabow algorithm on each node
  foreach(I in 1..Size,
    [ac(S,[]),ac(P,[]),ac(C,1),ac(D,1),ac(Comp,[])],
    $pp_find_scc(G,I,S^0,P^0,C^0,D^0,Comp^0,S^1,P^1,C^1,D^1,Comp^1,PreTable,CompTable)
  ),
  % Have components in topological ordering
  reverse(Comp,Components).

% Find strongly connected components using Gabow's algorithm 
% G: Graph (array of outgoing edges for each node id)
% N: Current node id
% S: Stack of nodes that have not yet been assigned to a scc
% P: Stack of nodes which have not yet been determined to belong to different scc
% C: Counter for preorder number
% D: Counter for component number
% Comp: List of components found so far
% PreTable: Table containing the preorder number for each node
% CompTable: Table containing the component number for each node and the
%   index number within that component
$pp_find_scc(G,N,S0,P0,C0,D0,Comp0,S3,P3,C3,D3,Comp3,PreTable,CompTable) :-
  ((bigarray_get(PreTable,N,C),nonvar(C)) ->
    % Node already has a preorder number
    ((bigarray_get(CompTable,N,Comp),nonvar(Comp)) ->
      % Node has been assigned to a component, do nothing
      P3 = P0
    ; % Node has not been assigned to a component
      % pop from P to collapse nodes until top element 
      % has same or lower preorder number
      $pp_scc_pop_until_less(P0,C,PreTable,P3)
    ),
    S3 = S0, C3 is C0, D3 is D0, Comp3 = Comp0
  ; % Node has no preorder number
    % Assigning new preorder number, pushing N on both stacks
    bigarray_put(PreTable,N,C0),
    S1 = [N|S0],
    P1 = [N|P0],
    C1 is C0 + 1,
    % Traverse all children
    bigarray_get(G,N,Edges),
    foreach(edge(_,Children) in Edges, Child in Children,
      [ac(C2,C1),ac(D2,D0),ac(S2,S1),ac(P2,P1),ac(Comp2,Comp0)],
        $pp_find_scc(G,Child,S2^0,P2^0,C2^0,D2^0,Comp2^0,S2^1,P2^1,C2^1,D2^1,Comp2^1,PreTable,CompTable)
    ),
    ( P2 = [N|P3] ->
      % N is top element of P, make new component
      $pp_scc_pop_until_found(S2,N,S3,NewComp),
      foreach(NodeId in NewComp, [ac(SubId,1)],
        ( bigarray_put(CompTable,NodeId,(D2,SubId^0)), SubId^1 is SubId^0 + 1 )
      ),
      Comp3 = [NewComp|Comp2],
      D3 is D2 + 1
    ; % N is part of another component, simply continue
      P3 = P2, S3 = S2, D3 is D2, Comp3 = Comp2
    ), C3 = C2
  ).

