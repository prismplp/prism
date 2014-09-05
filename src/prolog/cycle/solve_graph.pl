% log scale mode use scaling for matrix calculous
$pp_scaling_param(Scale):-Scale is 10000000.
% solve system given by graph G, dividided into components
% by creating linear equation systems and solving bottom up
$pp_solve_graph(G,Components,CompTable,ProbTable) :-
  bigarray_length(G,L),
  new_bigarray(ProbTable,L),
  % Solve each component
  foreach(Comp in Components,
    $pp_solve_component(G,Comp,CompTable,ProbTable)).

% Solve component by creating a linear system Ax = b
$pp_solve_component(G,CompNodes,CompTable,ProbTable) :-
  length(CompNodes,Length),
  SizeA is Length * Length,
  % Create matrix A and vector b
  new_bigarray(A,SizeA),
  new_bigarray(B,Length),
  % Fill diagonal of A with 1 and rest with 0
  foreach(I in 1..Length,
    ( foreach(J in 1..Length, [IJ],
      ( IJ is (I - 1) * Length + J,
        ( I == J -> bigarray_put(A,IJ,-1)
        ; bigarray_put(A,IJ,0)
        )
      )),
      bigarray_put(B,I,0)
    )
  ),
  % Iterate over nodes to fill A and b
  foreach(Node in CompNodes,
    $pp_update_linear_system(G,Node,CompTable,ProbTable,A,B,Length)
  ),
  % Solve linear system to x
  ( Length == 1 -> % single value, solve directly
    new_bigarray(X,1),
    bigarray_get(A,1,AVal),
    bigarray_get(B,1,BVal),
    ( AVal =:= 0.0 -> % singular system, not solvable
      $pp_raise_evaluation_error($msg(0200),['system not solvable'],non_solvable,$pp_solve_component/4)
    ; ($pp_in_log_scale -> ($pp_scaling_param(Scale),Prob is -1* BVal / AVal);Prob is -1*BVal/AVal),
      bigarray_put(X,1,Prob)
    )
  ; % Transform to list and call c interface for solving
    bigarray_to_list(A,AList),
    bigarray_to_list(B,BList),
    ( $pc_solve_linear_system(Length, AList, BList, XList,0) ->
      list_to_bigarray(XList, X)
    ; $pp_raise_evaluation_error($msg(0200),['system not solvable'],non_solvable,$pp_solve_component/4)
    )
  ),
  % Write probabilites
  foreach(Node in CompNodes, [CompSubId,NodeProb,CompId,NP,Scale],
    ( bigarray_get(CompTable,Node,(CompId,CompSubId)),
      bigarray_get(X,CompSubId,NodeProb),
      ($pp_in_log_scale->($pp_scaling_param(Scale),NP is log(NodeProb/Scale));NP is NodeProb),
      bigarray_put(ProbTable,Node,NP) )
  ).

% Update linear system for a certain node by traversing all edges
$pp_update_linear_system(G,Node,CompTable,ProbTable,A,B,Length) :-
  bigarray_get(G,Node,Edges),
  bigarray_get(CompTable,Node,(CompId,CompSubId)),
  foreach(edge(EdgeProb,Children) in Edges,
    [ProdProb,Dependants,DependantId,SumProb,TmpProb,Index,Scale],
    ( foreach(Child in Children, [ac(ProdProb,EdgeProb),ac(Dependants,[])],
        [ChildCompId,ChildProb,ChildCompSubId],
        ( bigarray_get(CompTable,Child,(ChildCompId,ChildCompSubId)),
          ( ChildCompId == CompId ->
            ProdProb^1 is ProdProb^0,
            Dependants^1 = [ChildCompSubId|Dependants^0]
          ; ( ChildCompId < CompId ->
              % Child is in lower component, then prob already known
              bigarray_get(ProbTable,Child,ChildProb),
              ($pp_in_log_scale->
                ProdProb^1 is ProdProb^0 + ChildProb
                ;ProdProb^1 is ProdProb^0 * ChildProb),
              Dependants^1 = Dependants^0
            ; % Edge to higher component, should not happen
              $pp_raise_internal_error($msg(9802), invalid_component_dependence,$pp_update_lin_system/7)
            )
          )
        )
      ), % Check for number of dependants
      ( Dependants = [] -> % Put into B
        bigarray_get(B,CompSubId,TmpProb),
        ($pp_in_log_scale->
                ($pp_scaling_param(Scale),SumProb is TmpProb + exp(ProdProb)*Scale)
          ;SumProb is TmpProb + ProdProb),
        bigarray_put(B,CompSubId,SumProb)
      ; ( Dependants = [DependantId] -> % Put into A
          Index is (CompSubId - 1) * Length + DependantId,
          bigarray_get(A,Index,TmpProb),
          ($pp_in_log_scale->
                  ($pp_scaling_param(Scale),SumProb is (TmpProb + exp(ProdProb)))
            ;(SumProb is TmpProb + ProdProb)),
          bigarray_put(A,Index,SumProb)
        ; % Non-linear relation, can not solve it
          $pp_raise_evaluation_error($msg(1402),['non-linear dependence'],non_linear_dependence,$pp_update_lin_system/7)
        )
      )
    )
  ).


