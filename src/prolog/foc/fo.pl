%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%                                                   %%%%%%
%%%%%%               First Order Compiler                %%%%%%
%%%%%%                                                   %%%%%%
%%%%%%                Revised by T.Sato                  %%%%%%
%%%%%%                   Feb. 2004                       %%%%%%
%%%%%%                                                   %%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% file = fo.pl

%%% PRISM用の追加・変更 => For PRISM
%%% HPSG用の追加・変更 => For HPSG

%%% trace 制御 ： fo_trace/0, fo_notrace/0

%%% 変数の使い方：
%%% 節のボディ内の束縛変数はすべて異なるように書かなければならない。

%%% FOC の能力拡張：
%%% 以下のプログラムでは、現在のFOCが all([X],cont(X,Y)) を扱えない
%%% 為の失敗が起きている。しかし、Xがall で縛られているという情報を
%%% cont()を通じて保持し、cont()のbodyの計算の中で
%%% all([Z],A(X,Z)->B) が現れた時 all([X,Z],A(X,Z)->B) を
%%% コンパイルすれば良い。
%%% 
%%%
%%% p(X,Y):- all([W],(q(W,W),Y=gg(W) -> 1=1)). % コンパイル失敗
%%% p(X,Y):- all([W],(Y=gg(W),q(W,W) -> 1=1)). % OK
%%%
%%% q(Y,Y).

%%% dynamic 宣言：
:- dynamic use_built_in/1.
:- dynamic fo_sort/2.
:- dynamic fo_sort/1.
:- dynamic pi/2.
:- dynamic distribute_disjunction/1.
:- dynamic delay_ununifiablity_check/1.
:- dynamic built_in_mode/1.
:- dynamic user_pred/1.
:- dynamic fo_trace/1.


%%% 現在、以下の宣言が有効
% use_built_in(yes/no).     % defaultは yes, msw を使う場合は必ず yes
% fo_sort(list,[[],[_|_]])  % list sort = nil or cons pattern
% delay_ununifiablity_check(yes/no). % default は no
% distribute_disjunction(yes/no).    % default は no
% built_in_mode(q(+,-)).             % ユーザ定義述語の宣言
% fo_sort(mysort,[0,s(_)]).          % ユーザ定義ソートの宣言
% fo_sort(q(mysort,list)).           % ソート情報

%%% 使用法：
% how to compile -> see file "aab"
% > prolog
% ?- compile(fo).
% ?- fo(aab).  <= run the FOC on file "aab"
% ?- compile(compiled_aab).  <= compiled program
% ?- s([a,X,Y,a,a,Z]).

%%% ユーザ定義のソート宣言：
%%% 
%%%   fo_sort(mysort,[a,b(_,_)]).
%%%   fo_sort(q(mysort,_,list)).
%%% 
%%% と定義すると、q(X,Y,Z) のコンパイルの際 'X'の位置が
%%% all で縛られていなければ、\+X=a の簡約化に使われる
%%% （X=b(_,_)になる）.
%%% ソート宣言に使える変数は無名変数 '_'に限る.
%%% リストソート
%%% 
%%%   fo_sort(list,[[],[_|_]])
%%% 
%%% はdefaultで宣言されている. モードパタンは
%%% 
%%%   get_mode_pattern(All,Left,RIGHT,Mode)
%%% 
%%% により作られ、入力モード'+'はソート宣言がある場合
%%% ソート情報付きの'+' モードになる（他のモード'-','*'
%%% は無関係. 例えば Mode = append([+,list],+,-) になる
%%% '*' はコンパイル中に導入され、all([Y],(q(X,Y) -> r(W))
%%%  のYのように、先件でall で縛られているが、後件に出現
%%%  しない変数が引数（位置）を示す.

%%% ユーザ定義の組み込み述語：
%%%
%%% ソースプログラムに以下のような宣言があると
%%% 
%%%    built_in_mode(q(+,-,-)).
%%% 
%%% FOCは q(X,Y,Z)について、fail するか、または
%%% XからY及びZが高々一つ決まると解釈し、
%%% 
%%%    all([Y,Z],(q(X,Y,Z) -> r(Y,W))
%%% 
%%% を
%%% 
%%%    (\+q(X,Y,Z) ; q(X,Y,Z)&r(Y,W))
%%% 
%%% に翻訳し、q/3についてそれ以上のコンパイルをしない.
%%% 実際はq(X,Y,Z)が２度実行される事を防ぐため、
%%%
%%%    user_q(Ans,X,Y,Z) :- ( q(X,Y,Z),Ans=true ; Ans=false ),!.
%%%
%%% をコンパイル結果に追加し、もとのゴールを
%%%
%%%    user_q(Ans,X,Y,Z),(Ans==false ; (Ans==true,r(Y,W)))
%%%
%%% に翻訳する.
%%%
%%% 注1）built_in_mode は各述語につき複数あっても良い.
%%% 例： fail_hpsg より
%%% built_in_mode(add_type(+,+)).
%%% built_in_mode(add_type(+,-)).
%%% built_in_mode(add_type(-,+)).
%%% built_in_mode(rule(+,-)).
%%% built_in_mode(unify(+,+,-)).
%%% ...

%%% ユーザ定義の確率組み込み述語：
%%%
%%% ソースプログラムに以下のような宣言があると
%%% 
%%%    built_in_p_mode(q(+,-,-)).
%%% 
%%% FOCは確率述語 q(X,Y,Z)について、msw をサンプリングで決めた時、
%%% fail するか、または XからY及びZが高々一つ決まると解釈し、
%%% 
%%%    all([Y,Z],(q(X,Y,Z) -> r(Y,W))
%%% 
%%% を
%%% 
%%%    (all([Y,Z],q(X,Y,Z)->fail) ; q(X,Y,Z)&r(Y,W))
%%% 
%%% とし、all([Y,Z],q(X,Y,Z)->fail を更にコンパイルする.
%%% 注1）現在の2引数のmsw の実装では、サンプリング実行プログラム
%%% としては正しくないコンパイルであるが、全解探索に関する限り正しい
%%% コンパイルである.
%%% 注2）built_in_mode のように節を追加する事はない.
%%% 注3）built_in_p_mode は各述語につき複数あっても良い.

%%% ディスユニフィケーションの遅延:
%%% delay_ununifiablity_check(no) の場合：
%%%    ある節Cのコンパイル途中結果 (s1=t1,..,sk=tk -> G) は
%%%    DisEq = ((\+s1=t1) ; ... ; (s1=t1k,\+s2=t2,..\sk=tk))、
%%%    EqG = (s1=t1,..,sk=tk,G) とした時
%%%    ( EqG ; DisEq )にコンパイルされる.
%%% 
%%% delay_ununifiablity_check(yes) の場合：
%%%    Cの前（上）にある節のコンパイル結果を XX とすると、
%%%    DisEqの実行を遅らせるため、((XX,DisEq) ; (XX,EqG))に
%%%    にコンパイルされる => 部分式のコピーが 2^n回起こる 

%-----------first_order_compiler----------

% 初期化し、step1,step2,step3を順番に呼び、後かたづけする。

% fo(Source)
%   First Order Compilerを起動する
%   入力ファイルは Source、
%   出力ファイルは compiled_Source となる
fo(Source):-
   first_order_compiler(Source).

% fo(Source,Compiled)
%    First Order Compilerを起動する
%    入力ファイルは Source、
%    出力ファイルは Compiled となる
fo(Source,Compiled):-
   first_order_compiler(Source,Compiled).

% first_order_compiler(Source)
%  First Order Compilerを起動する
%  入力ファイルはSource、
%  出力ファイルはcompiled_Sourceとなる

first_order_compiler(Source):-
  first_order_compiler_sub(Source), % step1とstep2を実行する
  step3_1(Source,N),                % 節を減らす
  write_clause(step3),              % step3が終った事を報告
  initialize,!,                     % 後片ずけする
  N=0.                              % ゴールコンパイルを失敗した場合、
                                    % コンパイルを失敗する

% first_order_compiler(Source,Compiled)
%    First Order Compilerを起動する
%    入力ファイルはSource、
%    出力ファイルはCompiledとなる
first_order_compiler(Source,Compiled):-
  first_order_compiler_sub(Source), % step1とstep2を実行する
  step3_2(Compiled,N),              % 節を減らす
  write_clause(step3),              % step3が終った事を報告 
  initialize,!,                     % 後片ずけする
  N=0.                              % ゴールコンパイルを失敗した場合、
                                    % コンパイルを失敗する

% first_order_compiler_sub(Source)
%    step1とstep2を実行する
first_order_compiler_sub(Source):-
   step0(Source,SOURCE),
        % 初期化する. delay などdefault を set する
   open(temp1,write,TEMP1),
        % temp1を開く. temp1は、節を減らす前のコンパイル結果を書く 
   open(temp2,write,TEMP2),
        % temp2を開く. temp2には、(->)を変形した Source を書く
        % このファイルで体に(->)を含む節を使った展開を可能にする
   step1(SOURCE,TEMP1,TEMP2),       % step1を実行する
   close(SOURCE),                   % 使用済のファイルを閉じる
   close(TEMP2),
   write_clause(step1),             % step1が終った事を報告

   retract(delay_ununifiablity_check(A)),
   ( A==yes,
       step2a(temp2,TEMP1)          % step2を実行する
   ; A\==yes,
       step2b(temp2,TEMP1) ),
   close(TEMP1),                    % 使用済のファイルを閉じる
   write_clause(step2).             % step2が終った事を報告


%--------------step0------------------------
step0(Source,SOURCE):-
   initialize,                      % 初期化する
   assertz(stack_number(0)),        % stack番号をゼロにする
   assertz(goal_compile_fail(0)),    
   ( clause(fo_trace(_),true)       % 始めは trace なし, trace は
   ; assertz(fo_trace(no)) ),!,     % fo_trace/0, fo_notrace/0 で制御
                                           % default setting は
   assertz(use_built_in(yes)),             % built_in あり
   assertz(delay_ununifiablity_check(no)), % delay なし
   assertz(distribute_disjunction(no)),    % distribute せず
   assertz(fo_sort(list,[[],[_|_]])),      % list sort あり

   open(Source,read,SO),
   step0_a(SO),                    % step0_a で宣言文の処理をする
   close(SO),
   open(Source,read,SOURCE).       % ソースファイルを読めるようにする 

step0_a(SO):-                      
   read(SO,Clause),
   ( Clause == end_of_file,!       % ファイルを最後まで読む
   ; step0_b(Clause) ).            % step0_b は常に失敗する

step0_a(SO):-
   step0_a(SO).

step0_b(Clause):-
   compound(Clause),                      % Clauseが宣言かを調べる。
   functor(Clause,(:-),1),!,
   arg(1,Clause,Declaration),
   ( compound(Declaration),
     ( functor(Declaration,op,3),         % オペレータ宣言の場合
         call(Declaration)
     ; Declaration =.. [dynamic,Dynamic], % dynamicの場合
         Dynamic =.. [(/),DyName,DyArity],
         assertz(for_dynamic(DyName,DyArity))
     )
   ; simple(Declaration) 
   ),!,     % :- XXX 宣言はすべてstep1_a で出力用ファイルに書かれる
   fail.    % 必ず失敗する

step0_b(Clause):-           % FOC に対する指示をassert する
   ( Clause = fo_sort(Fo_SortName,_), % fo_sort(list,[[],[_|_]])
       ( retract(fo_sort(Fo_SortName,_)) ; true )
   ; Clause = fo_sort(SortDecl),      %  fo_sort(app(list,_,list)).
       ( retract(fo_sort(SortDecl)) ; true )
   ; Clause = use_built_in(_),
       ( retract(use_built_in(_)) ; true )
   ; Clause = delay_ununifiablity_check(_),
       ( retract(delay_ununifiablity_check(_)) ; true )
   ; Clause = distribute_disjunction(_),
      ( retract(distribute_disjunction(_)) ; true )
   ; Clause = built_in_mode(_)
   ; Clause = built_in_p_mode(_)
   ),
   assertz(Clause),!, % 後で使うので、assertしておく
   fail.              % 必ず失敗する


%---------------------------------------------
% 初期化

% initialize
% 初期化する。
initialize:- 
   clean_up_file([temp1,temp2,temp3]),!,  % temp1とtemp2の中身を消去    
   clean_up_clause([                  % コンパイル過程でassertする
     pi(_,_),                         % 節を消去
     closure(_,_,_),                 
     transformed_formula(_),
     stack_number(_),
     clause_for_closure(_),
     for_dynamic(_,_),
     goal_compile_fail(_),
     fo_sort(_,_),
     fo_sort(_),
     cont_for_step3(_),
     closure_for_step3(_),
     user_pred(_),
     delay_ununifiablity_check(_),
     use_built_in(_),
     built_in_mode(_),
     built_in_p_mode(_),
     distribute_disjunction(_)
   ]).

% clean_up_file(FileList)
%  指定したファイルの中身を消去する
clean_up_file([]):-!.                 
clean_up_file([File|FileList]):-
   open(File,write,FILE),             % ファイルをopenすれば、
   close(FILE),!,                     % ファイルの中身が消える
   clean_up_file(FileList).

% clean_up_clause(ClauseList)
%  リスト中の節を全て削除する。
clean_up_clause([Clause|ClauseList]):-
   clean_up_clause_sub(Clause),!,
   clean_up_clause(ClauseList).
clean_up_clause([]).

% clean_up_clause_sub(Clause)
%  Clauseとユニファイ可能な節を全て削除する。
clean_up_clause_sub(Clause):-
   retract(Clause),
   fail.
clean_up_clause_sub(_).

%-------------step1----------------------------------------------

% step1
% SOURCEから節を読む。
% 節が definite clause なら TEMP1 に書く。
% 節が extended clause なら、
% extended clause を definite clause にして TEMP1 に書き、
% closure clause と continuation clauseを作る。
%
% closure clause をキューに入れる(assertzする)
% continuation clause に適当な操作(transform_cont_clause)を施し
% TEMP1(出力用一時ファイル)に書く。
%
% また、Sourceの not や all(All,(A->B)) を
% all(All,imply(LEFT,RIGHT)) にした節を
% TEMP2(入力用一時ファイル、展開する時に使う)に書く。

% step1(SOURCE,TEMP1,TEMP2)
step1(SOURCE,TEMP1,TEMP2):-
   read(SOURCE,Clause),               % ソースファイルから節を読む。 
   ( Clause = end_of_file,!           % EOFなら終了
   ; step1_a(TEMP1,TEMP2,Clause) ).
       % 節が definite clause か extended clauseかを判断し
       % 節に適当な操作を施す。必ず失敗する

step1(SOURCE,TEMP1,TEMP2):-          % step1をループにする
   step1(SOURCE,TEMP1,TEMP2).         

% step1_a(TEMP1,TEMP2,Clause)
%  Clauseがdefinite clauseか
%  extended clauseかを判断し、
%  節に適当な操作を施す。               
%  必ず失敗する
step1_a(TEMP1,_,Clause):-          % Clauseが宣言の時
   compound(Clause),               % Clauseが宣言かを調べる。
   functor(Clause,(:-),1),
   write_clause(TEMP1,Clause),!,   % 宣言を出力用ファイルに書く
   fail.                           % 必ず失敗する

step1_a(_,_,Clause):-
   ( Clause = fo_sort(_,_)         % これらは出力用ファイルに書かない
   ; Clause = fo_sort(_)
   ; Clause = delay_ununifiablity_check(_)
   ; Clause = use_built_in(_)
   ; Clause = distribute_disjunction(_)
   ; Clause = built_in_mode(_)
   ; Clause = built_in_p_mode(_)
   ),!,
   fail.

step1_a(TEMP1,TEMP2,Clause):-
   is_extended_clause(TEMP1,Clause,  % Clauseが extended clause なら成功する。
        NewClause1,NewClause2),!,    % closure clauseをキューに入れ、
                                     % continuation clauseを変形して
                                     % TEMP1に書く。
   remove_fo_sort(NewClause1,NewClause3),
   write_clause(TEMP1,NewClause3),   % extended clauseから作った
                                     % definite clauseをTEMP1に書く。
   write_clause(TEMP2,NewClause2),!, % (->)を変形した節をTEMP2に書く。
   fail.

step1_a(TEMP1,TEMP2,Clause):-         % Clause が definite clause の時
   remove_fo_sort(Clause,Clause1),
   write_clause(TEMP1,Clause1),       % Clause を TEMP1とTEMP2に書く。
   write_clause(TEMP2,Clause),!,
   fail.

%----------------step2--------------------------------------------

% step2
% step2(Temp2,TEMP1)
% closure clauseをキューから一つ取ってくる。
% 展開、畳み込みなどをする
% TEMP1(出力用一時ファイル)に出力する。

% delay_ununifiablity_check(yes) の時、ここで処理する
step2a(Temp2,TEMP1):-
     % キューからclosure clauseを一つ取ってくる
     % closure clauseは、
     % closure(closure_clause,mode pattern,closure_number)
     % の形でキューに入っている。
   retract(closure(Closure,Mode,_)),
     % closure clauseを実行可能にし、TEMP1に出力する
     % 必ず失敗する
   step2a_a(TEMP1,Temp2,Closure,Mode).

step2a(Temp2,TEMP1):-            % ループを作る。
   clause(closure(_,_,_),true),  % これが無いと動かない
   step2a(Temp2,TEMP1).           
step2a(_,_).                     % 最後に成功する。        

% step2a_a(TEMP1,Temp2,Closure,Mode)
%  Closure clauseを実行可能にする
%  必ず失敗する
step2a_a(TEMP1,Temp2,Closure,Mode):-
      % Temp2 から節を持ってくるために open する
   open(Temp2,read,TEMP2),

      % Closure clauseから Head と imply の左辺(Left)を取り出す
   Closure =.. [(:-),ClosureHead,ClosureBody],
   ClosureBody = all(_,imply(left(Left),right(_))),

      % implyの左辺とユニファイ可能な節を
      % ファイルから持ってきてリストにする
   step2_b(TEMP2,Left,ClauseList),

      % 展開、畳み込みをした変換結果をリストにしたものがReList
      % ReListは、[[[\+(s1=t1)],[s1=t1,\+(s2=t2)],[s1=t1,s2=t2,G]],...]
      % の形をしている
      % ReListの各要素は \+(s=t) or s=t,G を表す
      % ReListの各要素をandしたものが節のBodyとなる
   step2_c(TEMP1,ClauseList,ClosureHead,Mode,ReList),

      % 展開、畳み込みの結果(ReList)を節にして出力用ファイルに書く。
   remove_fo_sort(ClosureHead,ClosureHead1),

   step2a_d(TEMP1,ReList,ClosureHead1),
   close(TEMP2),!,  % 使用済のファイルを閉じる
   fail.

% delay_ununifiablity_check(no) の時、ここで処理する
step2b(Temp2,TEMP1):-
      % キューからclosure clauseを一つ取ってくる
      % closure clauseは、
      % closure(closure_clause,mode pattern,closure_number)
      % の形でキューに入っている。
   retract(closure(Closure,Mode,_)),
      % closure clauseを実行可能にし、TEMP1に出力する
      % closure clause がなくなったら失敗する
      % step2b_a は必ず失敗する
   step2b_a(TEMP1,Temp2,Closure,Mode).

step2b(Temp2,TEMP1):-            % ループを作る。
   clause(closure(_,_,_),true),  % これが無いと動かない
   step2b(Temp2,TEMP1).           
step2b(_,_).                     % 最後に成功する。        

% step2b_a(TEMP1,Temp2,Closure,Mode)
%  Closure clauseを実行可能にする
%  必ず失敗する
step2b_a(TEMP1,Temp2,Closure,Mode):-
      % Temp2 から節を持ってくるために open する
   open(Temp2,read,TEMP2),
      % Closure clauseから Headとimplyの左辺(Left)を取り出す
   Closure =.. [(:-),ClosureHead,ClosureBody],
   ClosureBody = all(_,imply(left(Left),right(_))),

      % implyの左辺とユニファイ可能な節を
      % ファイルから持ってきてリストにする
   step2_b(TEMP2,Left,ClauseList),

      % 展開、畳み込みをする. 変換結果をリストにしたものが ReList
      % ReListは、[[[\+(s1=t1)],[s1=t1,\+(s2=t2)],[s1=t1,s2=t2,G]],...]
      % の形をしている
      % ReListの各要素は \+(s=t) or s=t,G を表す
      % ReListの各要素をandしたものが節のBodyとなる
   step2_c(TEMP1,ClauseList,ClosureHead,Mode,ReList),

      % 展開、畳み込みの結果(ReList)を節して、出力用ファイルに書く。
      % ソート情報 fo_sort(xx,Var) を除く
   remove_fo_sort(ClosureHead,ClosureHead1),

   step2b_d(TEMP1,ReList,ClosureHead1),
   close(TEMP2),!,  % 使用済のファイルを閉じる
   fail.

% step2_b(TEMP2,Left,ClauseList)
%  implyの左辺(Left)とユニファイ可能な節を
%  ファイルから持ってきてリストにする
step2_b(TEMP2,Left,ClauseList):-
   step2_b1(TEMP2,Left),  % TEMP2から節を一つ持ってくる。
                          % それがLeftとユニファイ可能かを調べる。
   step2_b3(ClauseList).  % Leftとユニファイ可能な節をリストにする

% step2_b1(TEMP2,Left)
%  TEMP2から節を一つ持ってくる。
%  それがLeftとユニファイ可能かを調べる。
step2_b1(TEMP2,Left):-
   read(TEMP2,Clause),        % TEMP2から節を読む。
   ( Clause == end_of_file,!  % 全部読んだら終了。
   ; ( compound(Clause),
         functor(Clause,(:-),1),!,
         fail
     ; step2_b2(Left,Clause) )  % 節がLeftとユニファイ可能かを調べる。
   ).                           % 必ず失敗する

step2_b1(TEMP2,Left):-   % ループを作る
   step2_b1(TEMP2,Left).

%step2_b2(Left,Clause)
%  Clauseの頭がLeftとユニファイ可能なら、assertz しておく
step2_b2(Left,Clause):-
   get_head_body(Clause,Head,_),!, % 節から Head を持ってくる
   unifiable(Left,Head),!,         % 節が Left とユニファイ可能かを調べる。
   assertz(clause_for_closure(Clause)),!, % ユニファイ可能な節を保存しておく
   fail.

% step2_b3(ClauseList)
% Leftとユニファイ可能な節をリストにする
step2_b3(ClauseList):-
   step2_b3_dl(ClauseList-[]). % 差分リストを使う。

step2_b3_dl(ClauseList1-ClauseList3):-
   step2_b3_dl_sub(ClauseList1-ClauseList2),!,
   step2_b3_dl(ClauseList2-ClauseList3).
step2_b3_dl(ClauseList-ClauseList).

step2_b3_dl_sub([Clause|ClauseList]-ClauseList):-
   retract(clause_for_closure(Clause)).

% step2_c(TEMP1,ClauseList,ClosureHead,Mode,ReList)
%  展開、畳み込みをする
%  変換結果をリストにしたものがReList

% Leftとユニファイ可能な節がない時
step2_c(_,[],_,_,[]).

step2_c(TEMP1,ClauseList,ClosureHead,Mode,ReList):-
      % closure clauseのHeadから
      % モードパターン+に対応する項のリスト(X)と
      % continuation用変数(C)を取り出す
   ClosureHead =.. [_|ClosureHeadArgList],

      % モードパターン+に対応する項のリスト(X)と
      % continuation用変数(C)を取り出す
   get_last(ClosureHeadArgList,C,X1),
   check_x(X1,X),!,
      % 展開、畳み込みなどをする. 結果がReList
   step2_c1(TEMP1,ClauseList,X,C,Mode,ClosureHead,ReList).


% step2_c1(TEMP1,ClauseList,X,C,Mode,ClosureHead,ReList)
%  展開、畳み込みなどをする
%  結果がReList

% 節が無くなったら終了
step2_c1(_,[],_,_,_,_,[]).

step2_c1(TEMP1,[Clause|ClauseList],X,C,Mode,ClosureHead,ReList1):-
       % Clauseからall([w],imply(left(E),right(cont(t,C))))
       % を作り、ゴールコンパイルする. 結果は Re1となる
       % Re1は、[[\+(s1=t1)],[s1=t1,\+(s2=t2)],[s1=t1,s2=t2,G]]
       % の形をしている
       % ソート情報があれば\+(s1=t1)の変形に使う
   transform_formula(TEMP1,Clause,X,Mode,C,ClosureHead,Re1),
       % 結果をリストにする
   ReList1 = [Re1|ReList2],!,
       % 次の節について同じ操作をする
   step2_c1(TEMP1,ClauseList,X,C,Mode,ClosureHead,ReList2).

%--- delay_ununifiablitiy_check(yes)の時ここで処理する---
% step2a_d(TEMP1,ReList,Head)
%  展開、畳み込みの結果(ReList)を
%  節にして、出力用ファイルに書く。

% Leftとユニファイ可能な節が無いとき
% closure(x,C). をファイルに出力する
step2a_d(TEMP1,[],Head):-!,
   write_clause(TEMP1,Head).

step2a_d(TEMP1,ReLists,Head):-
       % ReListを節の体用のリストに変形する
   step2a_d1(ReLists,Body),
       % 頭と体から節を作って、ファイルに出力する。

   step2a_d2(TEMP1,Head,Body).

step2a_d1([ReList],Body):-!,
       % [[aa,bb],[cc,dd,ee]] = ReList
       %   => [(aa,bb),(cc,(dd,ee))] = BodyList
       %      元の節1つ分に対応
   step2a_d1_dl(ReList,BodyList-[]),
   list_to_disjunction(BodyList,Body).

% 返り値Bodyは実行可能式、
step2a_d1(ReList,Body):-
   fo_reverse(ReList,ReRevList),
      % ReList = 述語定義節のコンパイル結果をリストに
      % したもの. conjunctive に繋げて実行可能Bodyにする
   step2a_d1a(ReRevList,[],Body),!.

step2a_d1_dl([Re|ReList],BodyList1-BodyList3):-
   list_to_conjunction(Re,Re2),
   BodyList1=[Re2|BodyList2],!,
   step2a_d1_dl(ReList,BodyList2-BodyList3).
step2a_d1_dl([],BodyList-BodyList).

step2a_d1a([Re|ReRevList],BodyTemp1,Body):-
   step2a_d1a_sub(Re,BodyTemp1,BodyTemp2),!,
   step2a_d1a(ReRevList,BodyTemp2,Body).
step2a_d1a([],Body,Body).

% 返り値BodyTemp2 は実行可能式
% step2a_d1a(ReRevList,[],Body) から最初に呼ばれる
step2a_d1a_sub(Re1,[],BodyTemp2):-!,
   get_last(Re1,ReLastList,ReRestListList),
   list_to_conjunction(ReLastList,ReLast),
   ( ReRestListList = [],!,
       BodyTemp2 = ReLast
   ; step2a_d1_dl(ReRestListList,ReRestList-[]),
       list_to_disjunction(ReRestList,ReRest),
       list_to_disjunction([ReRest,ReLast],BodyTemp2)
   ).

% step2a_d1a_sub(Re1,BTemp1,BTemp2)
%  Re1 はソースプログラムの１つの節C1のコンパイル結果
%  Re1 =[[\+s1=t1],[s1=t1,\+s2=t2],.., [s1=t1,..,sk=tk,G]]
%       |<-----  ReRestListList ---->|      ReLastList
%  BTemp1はソースファイルでC1の上にあった節達のコンパイル
%  結果であり、実行可能な式になっている
%
%  ReRest = ((\+s1=t1);...;(s1=t1k,\+s2=t2,..\sk=tk)) と
%  ReLast =(s1=t1,..,sk=tk,G) を作る.
%  本来 ((ReRest;ReLast),BTemp1) にコンパイルする所を
%  disunificationの部分 ReRest の計算をdelay させる為
%  BTemp2 = ((ReLast,BTemp1);(BTemp1,ReRest))
%                    EqG          DisEq
%  にコンパイルする => 部分式のコピーが 2^n回 起こる

step2a_d1a_sub(Re1,BTemp1,BTemp2):-
   get_last(Re1,ReLastList,ReRestListList),
   list_to_conjunction(ReLastList,ReLast),
   ( ReRestListList = [],!,
       generate_term((','),[ReLast,BTemp1],BTemp2)
   ; true,
       generate_term((','),[ReLast,BTemp1],OrLeft),
       step2a_d1_dl(ReRestListList,ReRestList-[]),
       list_to_disjunction(ReRestList,ReRest),
       generate_term((','),[BTemp1,ReRest],OrRight),
       list_to_disjunction([OrLeft,OrRight],BTemp2)
   ).

step2a_d2(TEMP1,Head,Body):-
   get_fo_sort(Head,Fo_SortList),
   \+(Fo_SortList = []),
   remove_fo_sort(Head,Head1),
   generate_clause(Head1,[Body],Clause),
   use_fo_sort(TEMP1,Fo_SortList,Clause).

step2a_d2(TEMP1,Head,Body):-
   generate_clause(Head,[Body],Clause),
   write_clause(TEMP1,Clause).

%--- 以上で delay_ununifiablitiy_check(yes)の処理修了---

%--- delay_ununifiablitiy_check(no)の時ここで処理する---
% step2b_d(TEMP1,ReList,Head)
%  展開、畳み込みの結果(ReList)を
%  節にして、出力用ファイルに書く。

% Leftとユニファイ可能な節が無いとき
% closure(x,C). をファイルに出力する
step2b_d(TEMP1,[],Head):-
   write_clause(TEMP1,Head).

step2b_d(TEMP1,ReList,Head):-
      % ReListを節の体用のリストに変形する
   step2b_d1(ReList,BodyList),
      % 頭と体から節を作って、ファイルに出力する。
   step2b_d2(TEMP1,Head,BodyList).

step2b_d1(ReList,BodyList):-
   step2b_d1_dl(ReList,BodyList-[]).

step2b_d1_dl([Re|ReList],BodyList1-BodyList3):-
   step2b_d1_dl_sub(Re,BodyList1-BodyList2),!,
   step2b_d1_dl(ReList,BodyList2-BodyList3).
step2b_d1_dl([],BodyList-BodyList).

step2b_d1_dl_sub(Re1,BodyList1-BodyList2):-
   step2b_d1_dl_sub1(Re1,Re2-[]),
   ( Re2 =[],
       BodyList1 = BodyList2
   ; list_to_disjunction(Re2,Re3),
       BodyList1 = [Re3|BodyList2]
   ).
 
step2b_d1_dl_sub1([R1|Re1],Re2-Re4):-
   step2b_d1_dl_sub2(R1,Re2-Re3),!,
   step2b_d1_dl_sub1(Re1,Re3-Re4).
step2b_d1_dl_sub1([],Re2-Re2).

step2b_d1_dl_sub2(R2,Re2-Re3):-
   ( R2 = [],
       Re2 = Re3
   ; list_to_conjunction(R2,R3),
       Re2 = [R3|Re3]
   ).

step2b_d2(_,_,[]):-!.

%%% この節は恐らく無駄 %%%%
step2b_d2(TEMP1,Head,BodyList):-
   get_fo_sort([Head,BodyList],Fo_SortList),
   \+(Fo_SortList = []),!,
   remove_fo_sort(Head,H1),
   remove_fo_sort(BodyList,B1),
   list_to_conjunction(B1,Body1),
   generate_clause(H1,[Body1],Clause),
     % Clause の Var をソートの値の一つ L で
     % 置き換えたものをTEMP1 に書き込む.
   use_fo_sort(TEMP1,Fo_SortList,Clause).

step2b_d2(TEMP1,Head,BodyList):-
   list_to_conjunction(BodyList,Body),
   generate_clause(Head,[Body],Clause),
   remove_fo_sort(Clause,Clause1),
   write_clause(TEMP1,Clause1).

%---- 以上で delay_ununifiablitiy_check(no)の処理修了 ---

%---------------step3-------------------------

% step3_1
% ゴールを減らす。
% goal_compile が失敗していたときは N=1 となる。

%step3_1(Source,N)
step3_1(Source,N):-
   retract(goal_compile_fail(N)),
   ( N=1,!
           % 出力用ファイルのファイル名を作る
   ; append_name('compiled_',Source,Compiled),
       open(Compiled,write,COMPILED),  % ファイルを開く
       open(temp1,read,TEMP1),
       step3_a(COMPILED,TEMP1),        % 節を整理し、記録する
       step3_closure(COMPILED),        % closure clauseのゴールを減らす
       step3_user_pred(COMPILED),      % ユーザ定義述語の節を書き出す
       close(TEMP1),                   % ファイルを閉じる
       close(COMPILED)
    ).


% step3_2(Compiled,N)
% 出力用ファイルを指定した場合。
step3_2(Compiled,N):-
   retract(goal_compile_fail(N)),
   ( N=1,!
   ; open(temp1,read,TEMP1),          % ファイルを開く
       open(Compiled,write,COMPILED),
       step3_a(COMPILED,TEMP1),!,     % 節を整理し、assert で記録する
       step3_closure(COMPILED),       % closure clauseのゴールを減らす
       step3_user_pred(COMPILED),
       close(TEMP1),                  % ファイルを閉じる
       close(COMPILED)
    ).

step3_user_pred(COMPILED):-
   ( retract(user_pred(UserBuilt_In)),
        copy_term(UserBuilt_In,CL),
        numbervars(CL,0,_),
        write_clause(COMPILED,CL),
        fail
   ; true ).


% step3_a(COMPILED,TEMP)
%  TEMPから節を読む
%  節を整理する
%  節を COMPILED に書く
step3_a(COMPILED,TEMP):-
   read(TEMP,Clause), % 節を読む

   ( Clause == end_of_file,!      % 節を全部読んだら終了
   ; step3_b(COMPILED,Clause) ).  % 節を整理して、COMPILEDに出力する
                                  % 必ず失敗する

step3_a(COMPILED,TEMP):-          % ループを作る
   step3_a(COMPILED,TEMP).

%step3_b(COMPILED,Clause)
% 節を整理して、COMPILEDに出力する
% 必ず失敗する

% Clauseが宣言(:- op(...)等)の場合
step3_b(COMPILED,Clause):-
   compound(Clause),
   functor(Clause,(:-),1),
   write_clause(COMPILED,Clause),!,
   fail.

step3_b(COMPILED,Clause):-
   get_head_body(Clause,Head,BodyList),
   ( BodyList = [],
       write_clause2(COMPILED,Head),!,
       fail
%%% >>>>>>>> 修正 by T.Sato, Sept. 2003 >>>>>>>>>>>>>>>>>>>>>>>>>
    ; true,
%%% <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
%%% original codes
%%%   ; compound(Head),

       functor(Head,HeadName,_),
       ( HeadName = cont,  % 変形後のcontinuation clauseの場合
            assertz(cont_for_step3(Clause)),
            Head =.. HeadList,
            step3_cont(COMPILED,HeadList,BodyList),!,
            fail
       ; ( name(HeadName,HeadNameList),  % 変形後のclosure clauseの場合
              fo_append("closure",_,HeadNameList),
              assertz(closure_for_step3(Clause)),!,
              fail
         ; write_clause2(COMPILED,Clause),!, % その他の場合
              fail
         )
       )
   ).
 

% step3_cont(COMPILED,HeadList,BodyList1)
% 変形後の continuation clauseを
% 論理和標準形に変形した後ファイルに書く

%-----------------------------------------
% distribute_disjunction(yes) の場合
step3_cont(COMPILED,HeadL,BodyL):-
   clause(distribute_disjunction(yes),true),!,
   ( step3_closure1_sub(COMPILED,HeadL,BodyL), fail
   ; true ).

step3_cont(_,_,[]).
step3_cont(COMPILED,HeadList,[Body1]):-
   ( compound(Body1),
       functor(Body1,(;),2),
       disjunction_to_list(Body1,BodyList2),!,
       step3_cont2(COMPILED,HeadList,BodyList2)
   ; Head =.. HeadList,
       generate_clause(Head,[Body1],Cont),
       write_clause2(COMPILED,Cont)
   ).

% s(X)=0 など常に失敗するゴールを持つBody2は無視し、ファイルに書かない
step3_cont2(COMPILED,Head1List,[Body2|BodyList2]):-
   conjunction_to_list(Body2,Body2List),
   ( reduction_of_goal1(Head1List,Body2List,Head2List,Body3List),
       Head2 =.. Head2List,
       generate_clause(Head2,Body3List,Cont),
       write_clause2(COMPILED,Cont)
   ; true ),!,
   step3_cont2(COMPILED,Head1List,BodyList2).
step3_cont2(_,_,[]).

% step3_closure(TEMP3)
%  変形後のclosure clauseのゴールを減らす。
step3_closure(TEMP3):-
   clause(distribute_disjunction(yes),true),
   step3_closure1(TEMP3).

step3_closure(TEMP3):-
   clause(distribute_disjunction(no),true),
   step3_closure2(TEMP3).

step3_closure(_):-
   \+(clause(closure_for_step3(_),true)).

%-----------------------------------------
% distribute_disjunction(yes) の場合

step3_closure1(TEMP3):-
   retract(closure_for_step3(Closure1)),
      % このゴールのfail によるretractの繰り返し
   step3_closure1a(TEMP3,Closure1).

step3_closure1a(TEMP3,Closure1):-
      % Body1List はconjunction を表す
   get_head_body(Closure1,Head1,Body1List),
   Head1 =.. Head1List,
   step3_closure1_sub(TEMP3,Head1List,Body1List),!,
   fail.

step3_clousre1_sub(TEMP3,Head1List,[]):-!,
   Head =.. Head1List,
   write_clause2(TEMP3,Head).

%%%%>>>>>>> 入れ換え T.Sato Oct.2003 >>>>>>
step3_closure1_sub(TEMP3,HeadL,BodyL):-
   step3_and_to_or(BodyL,AndL-[]),
   step3_and_to_or2(TEMP3,HeadL,AndL), fail.

step3_and_to_or2(TEMP3,HeadL,AndL):-
      % AndL がfail する節はファイルに書かない
   reduction_of_goal1(HeadL,AndL,HeadL2,AndL2),
   Head =.. HeadL2,
   generate_clause(Head,AndL2,Clause),
   write_clause2(TEMP3,Clause),!.

% step3_and_to_or(BodyL,X-Z)
%  BodyL (conjunction) を論理和標準形にした
%  時の disjunct をX-Z に返す
step3_and_to_or([],X-X).
step3_and_to_or([B|BodyL],X-W):-
   ( B = (B1 , B2),
        step3_and_to_or([B1],X-Y),
        step3_and_to_or([B2],Y-Z),
        step3_and_to_or(BodyL,Z-W)
   ; B = (B1 ; B2),
        ( step3_and_to_or([B1|BodyL],X-W)
        ; step3_and_to_or([B2|BodyL],X-W) )
   ; \+(B=(B1,B2)),\+(B=(B1;B2)),
        X=[B|Y],step3_and_to_or(BodyL,Y-W)
   ).
%%%%<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

%-----------------------------------------
%  distribute_disjunction(no) の場合
ste3_closure2(_):-
   \+(clause(closure_for_step3(_)),true),!.

% 最後に必ず失敗
step3_closure2(TEMP3):-
   retract(closure_for_step3(Closure)),
       % fail による繰り返し
   step3_closure2a(TEMP3,Closure).

% 最後に必ず失敗
step3_closure2a(TEMP3,Closure):-
   get_head_body(Closure,Head,BodyList),
   Head =.. HeadList,
   step3_closure2_sub(TEMP3,HeadList,BodyList),!,
   fail.

%%%>>>>>> 修正、追加 T.Sato OCt.2003 >>>>>>>>>>>>>>
step3_closure2_sub(TEMP3,HeadL,BodyL):-
   ( reduction_of_goal1(HeadL,BodyL,HeadL2,BodyL1),!,

      list_to_conjunction(BodyL1,XX),
      elim_eqs(XX,YY),
      conjunction_to_list(YY,BodyL2),!,
      ( is_variant(BodyL2,BodyL),
          Head =.. HeadL2,
          generate_clause(Head,BodyL,Clause),
          write_clause2(TEMP3,Clause)
      ; step3_closure2_sub(TEMP3,HeadL2,BodyL2) )
   ; true ).

% A=B, \+A=Bの簡単化をする
elim_eqs((A,B),F2):-
   elim_eqs(A,A1),
   elim_eqs(B,B1),
   ( A1 == true, F2=B1
   ; B1 == true, F2=A1
   ; (A1 == fail ; B1 == fail), F2=fail
   ; F2 = (A1,B1) ),!.
elim_eqs((A;B),F2):-
   elim_eqs(A,A1),
   elim_eqs(B,B1),
   ( A1 == fail, F2=B1
   ; B1 == fail, F2=A1
   ; (A1 == true ; B1 ==true ), F2=true
   ; F2 = (A1;B1) ),!.
elim_eqs((A=B),F2):-
   ( A==B, F2=true
   ; \+(A=B),F2=fail
   ; F2 = (A=B) ),!.
elim_eqs(\+(A=B),F2):-
   ( A==B, F2=fail
   ; \+(A=B), F2=true
   ; F2 = \+(A=B) ),!.
elim_eqs(F,F).
%%%<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

%---------------------------------------------------------------
% Extend clause を definite clause に変換する。
%
% is_extended_clause(TEMP1,Clause,NewClause1,NewClause2)
%   Clauseが extended clause なら成功する。
%   closurec clauseをキューに入れ、
%   continuation clauseを変形して TEMP1 に書く。
is_extended_clause(TEMP1,Clause,NewClause1,NewClause2):-

      % 節からHeadとBodyのリストを取り出す
   get_head_body(Clause,Head,BodyList),!,

      % Bodyの中にextended term(all([V],(A->B)))があるかを調べる
      % extended termがあったら、closure clause と
      % continuation clause を作り、適当な操作をする
   check_body(TEMP1,BodyList,NewBodyList1,NewBodyList2),!,

      % extended termを closure(x,f) に置き換えた節が NewClause1
   generate_clause(Head,NewBodyList1,NewClause1),

      %(->)を imply(LEFT,RIGHT) に置き換えた節が NewClause2
   generate_clause(Head,NewBodyList2,NewClause2). 

% check_body(TEMP1,BodyList,NewBodyList1,NewBodyList2)
%  BodyList 内に extend term がある場合、
%  closure clause などを生成して、definite clause を作るような項を返す。
%  BodyList内にextend termが無い場合、失敗する。
check_body(TEMP1,BodyList,NewBodyList1,NewBodyList2):-
   check_body(TEMP1,0,BodyList,NewBodyList1,NewBodyList2).

check_body(_,1,[],[],[]).
      % extended termがあった場合は、N=1となり、成功する

check_body(TEMP1,_,[Term|BodyList],NewBodyList1,NewBodyList2):-
   check_exterm(Term,ALL1),!,
      % Termがextended termかどうかを調べる
      % extended termの場合は、(->)をimply(LEFT,RIGHT)に変形する
   goal_compile(TEMP1,ALL1,DefTerm),
      % goal_compileする, goal_compileの結果がDefTerm
   NewBodyList1 = [DefTerm|NewBodyList3], 

    ALL1=ALL2,

   NewBodyList2 = [ALL2|NewBodyList4],!,
   check_body(TEMP1,1,BodyList,NewBodyList3,NewBodyList4).

check_body(TEMP1,N,[Term|BodyList],[Term|NewBodyList1],[Term|NewBodyList2]):-
   check_body(TEMP1,N,BodyList,NewBodyList1,NewBodyList2).

% check_body(TEMP1,BodyList1,BodyList2)
%   BodyList内にextend termがある場合、
%   closure clauseなどを生成して、definite clauseを作るような項を返す。
%   常に成功する
check_body(_,[],[]).
     % extended termがあった場合は、N=1となり、成功する

check_body(TEMP1,[Term|BodyList1],BodyList2):-
   check_exterm(Term,ALL),!,
     % Termがextended termかどうかを調べる
     % extended termの場合は、(->)をimply(LEFT,RIGHT)に変形する
   goal_compile(TEMP1,ALL,DefTerm),
     % goal_compileする,goal_compileの結果がDefTerm
   BodyList2 = [DefTerm|BodyList3],!, 
   check_body(TEMP1,BodyList1,BodyList3).

check_body(TEMP1,[Term|BodyList1],[Term|BodyList2]):-
   check_body(TEMP1,BodyList1,BodyList2).

% check_exterm(Term,ALL)
%   Termが extended term(all([],(->)))の場合、適当に変形する。
%                                  でない場合、失敗する。
check_exterm(Term,ALL):-
   compound(Term),
      % Term が not(A) かを調べ、
   functor(Term,not,_),!,   
   not_to_all(Term,Exterm), 
      % Term が not(A) だったら、all([],imply(left(A),right(fail)))に変形する。
      % allの中にextermがあったら、変形する。
   check_all(Exterm,ALL).

%%% >>>>>>>> 追加 by T.Sato, Sept. 2003 >>>>>>>>>>>>>>
check_exterm(Term,ALL):-
   compound(Term),
   Term = all(Qv,not(F)),
   Term2 = all(Qv,(F -> fail)),
   check_exterm(Term2,ALL).
%%% <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

check_exterm(Term,ALL):-
   compound(Term),
      % Term が (A->B)かを調べ、
   functor(Term,(->),_),!,    
   imply_to_all(Term,Exterm),
      % (A->B)なら、all([],imply(left(A),right(B)))に変形する。
      % allの中にextermがあったら変形する。           
   check_all(Exterm,ALL).

check_exterm(Term,ALL2):-
   compound(Term),
      % Termが(all([X],(A->B)))かを調べ、
   Term =.. [all,All,IMPLY],
   IMPLY =.. [(->)|_],!,
   all_to_all(Term,Exterm),
      % (all([X],(A->B))だったら、
      % all([X],imply(left(A),right(B)))に変形する。
      % allの中にextermがあったら変形する。            
      % 変数名を換えておく
   check_all(Exterm,ALL1),
   ALL1 = all(All,_),
   ALL1 =.. ALL1List,
   some_variables_to_new_variable(All,ALL1List,ALL2List),
   ALL2 =.. ALL2List.

% (A,B) となっている場合
check_exterm(Term,ALL):-
   compound(Term),
   functor(Term,(','),_),!,
   conjunction_to_list(Term,TermList),!,
%%% 修正 by T.Sato,  '04 Nov. <<<<<<<<<<<<<<<<<<<<<<<<
%%%  check_exterm(TermList,ALLList),
   check_exterm_list(TermList,ALLList),
%%%>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
   list_to_conjunction(ALLList,ALL).

% (A;B)となっている場合
check_exterm(Term,ALL):-
   compound(Term),
   functor(Term,(;),_),!,
   disjunction_to_list(Term,TermList),!,
%%% 修正 by T.Sato,  '04 Nov. <<<<<<<<<<<<<<<<<<<<<<<<
%%%   check_exterm(TermList,ALLList),
   check_exterm_list(TermList,ALLList),
%%%>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
   list_to_disjunction(ALLList,ALL).

% 既に、all(All,imply(LEFT,RIGHT))の形になってる時
% goal_compileなどで使う
check_exterm(Term,ALL2):-
   Term = all(All,imply(LEFT,RIGHT)), % Termの形をチェックする
   fo_list(All),
   compound(LEFT),
   compound(RIGHT),
   LEFT =.. [left|_],
   RIGHT =..[right|_],!,
   check_all(Term,ALL1),  % allの中にextermがあったら変形する。
   ALL1 = all(All,_),     % 変数名を換えておく
   ALL1 =.. ALL1List,
   some_variables_to_new_variable(All,ALL1List,ALL2List),
   ALL2 =.. ALL2List.

check_exterm(Term,ALL2):-
   Term = all(All,imply(left,Right)), % 左辺が空の時
   fo_list(All),
   compound(Right),
   Right =.. [right|_],!,
   check_all(Term,ALL1),  % allの中にextermがあったら、変形する。
   ALL1=all(All,_),       % 変数名を換えておく
   ALL1=..ALL1List,
   some_variables_to_new_variable(All,ALL1List,ALL2List),
   ALL2=.. ALL2List.

% exist([Var],A)の場合
check_exterm(Term1,Term3):-
       % Term が exist([Var],A) の場合
   compound(Term1),
   functor(Term1,exist,2),!, 
   check_all(Term1,Term2),     % existの中身を確認
   Term2 =.. ExistList2,       % 変数名を換えておく
   ExistList2 = [exist|[ExistVar2|_]],
   some_variables_to_new_variable(ExistVar2,ExistList2,ExistList3),
   Term3 =.. ExistList3.

% check_exterm_list(List1,List2)
% check_extermのリスト版
check_exterm_list([],[]).

check_exterm_list([Term1|List1],List2):-
   check_exterm(Term1,Term2),!, % Termがallやexistの場合
   List2 = [Term2|List3],!,
   check_exterm_list(List1,List3).

check_exterm_list([Term|List1],[Term|List2]):-
   check_exterm_list(List1,List2).

% not_to_all(NOT,ALL)
% not(A)をall([],imply(left(A),right(fail)))に変形する。
not_to_all(NOT,ALL):-
   compound(NOT),
   NOT =.. [not|NotList],
   generate_left(NotList,LEFT), % left(A)を作る
      % all([],imply(left(A),right(fail)))を作る
   generate_all([],LEFT,right(fail),ALL).

% imply_to_all(IMPLY,ALL)
%  (A->B)を all([],imply(left(A),right(B))) に変形する。
imply_to_all(IMPLY,ALL):-
     % (A->B)をimply(left(A),right(B))に変形する。
  transform_imply(IMPLY,[],ALL).

% transform_imply(IMPLY,All,ALL)
%   (A->B)をimply(left(A),right(B))に変形する。
transform_imply(IMPLY,All,ALL):-
   IMPLY =..[( ->)|[LefT|[RighT|[]]]],
   conjunction_to_list(LefT,Left),   % 連言をリストにする
   conjunction_to_list(RighT,Right),
   generate_left(Left,LEFT),         % 左辺を作る
   generate_right(Right,RIGHT),      % 右辺を作る
   generate_all(All,LEFT,RIGHT,ALL). % ALLを作る

% all_to_all(ALL1,ALL2)
%  all([X],(A->B))をall([X],imply(left(A),right(B)))に変形する。
all_to_all(ALL1,ALL2):-
   ALL1 =.. [all|[All|[IMPLY|[]]]], 
     % (A->B)をimply(left(A),right(B))に変形する
   transform_imply(IMPLY,All,ALL2). 

% check_all(ALL1,ALL2)
%  ALLの中のextermを変形する
check_all(ALL1,ALL2):-
     % 入力がALLの形をしていることを確認
   ALL1 = all(All,imply(LEFT1,RIGHT1)),
   compound(LEFT1),
   compound(RIGHT1),
   LEFT1 =..[left|Left1],
   RIGHT1 =..[right|Right1],!,
   check_exterm_list(Left1,Left2),   % 左辺変形
   check_exterm_list(Right1,Right2), % 右辺変形

   generate_left(Left2,LEFT2),       % 合成
   generate_right(Right2,RIGHT2),
   generate_all(All,LEFT2,RIGHT2,ALL2).

% 左辺が空
check_all(ALL1,ALL2):-
     % 入力がALLの形をしていることを確認
   ALL1 = all(All,imply(LEFT,RIGHT1)),
   LEFT = left,
   compound(RIGHT1),
   RIGHT1 =..[right|Right1],!,
   check_exterm_list(Right1,Right2),  % 右辺変形
   generate_right(Right2,RIGHT2),     % 合成
   generate_all(All,LEFT,RIGHT2,ALL2).

% existの場合
check_all(Exist1,Exist2):-
     % 入力がexist([ExistVar],F)であることを確認
   Exist1 =.. [exist|[ExistVar|[ExistArg1List|[]]]],!,
   conjunction_to_list(ExistArg1List,ExistArg1),
   check_exterm_list(ExistArg1,ExistArg2List),   % 中身変形
   list_to_conjunction(ExistArg2List,ExistArg2),
   Exist2 =.. [exist|[ExistVar|[ExistArg2|[]]]]. % 合成

% and_to_imply(ALL1,ALL2)
%  imply の左辺が連言の時、左辺のandを imply に変える。
%  goal_compile の<ケース 3>で使う。
and_to_imply(ALL1,ALL2):-
       % 入力がALLの形をしていることを確認
   ALL1 = all(All1,imply(LEFT1,RIGHT1)),
   LEFT1 =.. [left|[Left1l|Left1r]],
   \+(Left1r = []),!,         % 左辺が連言である。
       % 左辺のなかでも、最も左にある項が使ってる
       % 自由変数をリストにする。
   get_free_variable(Left1l,Left1lFVar),     
       % Allと、左辺の中で最左の項の自由変数の
       % 積集合を作る
   variable_join(All1,Left1lFVar,All2l),
       % 重複した変数を取り除く
   variable_unique(All2l,All2lu),
       % 元のAll から All2l を取り除いたものが All2r
       % ただし、all(y,(A1,A2 -> B))を
       % all(y1,(A1 -> all(y2,(A2 -> B)))) に変形すると、
       % Allはy、All2l は y1, All2r は y2 に相当する
   variable_diff(All1,All2lu,All2r),
   generate_left(Left1r,LEFT2R),
       % 項を作る。

   generate_all(All2r,LEFT2R,RIGHT1,ALL2R),
   generate_left([Left1l],LEFT2L),
   generate_right([ALL2R],RIGHT2L),
   generate_all(All2lu,LEFT2L,RIGHT2L,ALL2).

%----------------------------------------
% goal_compile(TEMP,Term,Compiled)
% ゴールコンパイルする。
% 入力は Term、出力は Compiled

goal_compile(_,Term,_):-
  clause(fo_trace(yes),true),
  \+(\+((numbervars(Term,0,_),
      format("  => CALL goal_compile/3:~n       ~w~n",[Term])))),
  fail.

% 入力Termが all(All,IMPLY) の場合
goal_compile(TEMP,Term,Compiled):-
   compound(Term),
   Term = all(All,imply(LEFT,RIGHT)),
   clause(use_built_in(A),true),!,
   ( A==yes,!,
        goal_compile(TEMP,All,LEFT,RIGHT,Compiled)
   ; A\==yes,
        goal_compile2(TEMP,All,LEFT,RIGHT,Compiled) ).

% 入力Termが exist(ExistVar,ExistArg)の場合
goal_compile(TEMP,Term,Compiled):-
   compound(Term),
   Term =.. [exist, ExistVar, ExistArg1],!,
   conjunction_to_list(ExistArg1,ExistArg1List),
   check_body(TEMP,ExistArg1List,ExistArg2List),
   some_variables_to_new_variable(ExistVar,ExistArg2List,ExistArg3List),
   list_to_conjunction(ExistArg3List,Compiled).

% 入力Termが連言の場合
goal_compile(TEMP,Term,Compiled):-
   compound(Term),
   functor(Term,(','),_),!,
   conjunction_to_list(Term,TermList),
   check_body(TEMP,TermList,CompiledList),
   list_to_conjunction(CompiledList,Compiled).

% 入力Termが選言の場合
goal_compile(TEMP,Term,Compiled):-
   compound(Term),
   functor(Term,(;),_),!,
   disjunction_to_list(Term,TermList),
   check_body(TEMP,TermList,CompiledList),
   list_to_disjunction(CompiledList,Compiled).

% 入力Termが以上のパターンに当てはまらない場合
goal_compile(_,X,X).

%---------------------------------------------------
% goal_compile/5
%   all(All,left(Left),right(Right)) をコンパイルする

% <ケース 1> 左辺Leftが空の場合
goal_compile(TEMP,All,left,right(Right),Compiled):-
   ( All == []
   ; get_free_variable(Right,RightVar),
       variable_join(All,RightVar,[]) ),!,
   goal_compile(TEMP,Right,Compiled).

goal_compile(_,All,left,right(Right),_):-
   F = all(All, (left -> right(Right))),
   numbervars(F,0,_),
   format("~nFailed goal_compile/5 at < Case 1 >: ~w~n~n",[F]),
   retract(goal_compile_fail(_)),
   assertz(goal_compile_fail(1)),!,
   fail.

%--------- use_built_in(yes) ----------
% <ケース 2> 左辺Leftが一つのアトムの場合
%   all(_,left(Left)->...)の形をコンパイル

%%%%%%%%%% For compile Trace %%%%%%%%%%%%%%%
%goal_compile(_,_,F,_,_):-
%  F = left(Goals),
%  numbervars(Goals,0,_),
%  format(" compiling : ~w~n",[Goals]),
%  fail.

% <ケース 2.1> 左辺が常にfalseの場合
% 左辺Leftが fail の場合
goal_compile(_,_,left(fail),_,true):-!.

% <ケース 2.2.0>
% 左辺Leftが cut の場合
% cut をそのままにしておく
goal_compile(TEMP,All,left(Left),RIGHT1,Compiled):-
   Left == '!',
   compound(RIGHT1),
   RIGHT1 =.. [right|Right1],!,
   goal_compile_cut(TEMP,All,Right1,Compiled).

% 左辺Leftが true の場合
% true を取り除く
goal_compile(TEMP,All,left(Left),RIGHT1,Compiled):-
   Left == true ,
   compound(RIGHT1),
   RIGHT1 =.. [right|Right1],!,
   goal_compile_true(TEMP,All,Right1,Compiled).

% 左辺Leftが必ず成功し、副作用のある built_in/0 の場合
goal_compile(TEMP,All,left(Left),RIGHT1,Compiled):-
   ( Left == nl
   ; Left == trace
   ; get_variable(Left,LeftV),
        variable_join(LeftV,All,[]), % 'All' must not quantify vars in Left
        ( Left = read(_)
        ; Left = write(_)
        ; Left = set_sw(_)
        )
   ),  % add others!
   compound(RIGHT1),
   RIGHT1 =.. [right|Right1],!,
   goal_compile_built_in_0(TEMP,All,Left,Right1,Compiled).

% <ケース 2.2.1>
% 左辺が<ケース 2.2.0>以外のユーザ定義のbuilt-in述語の場合
% The predicate has functional dependency
% (+ : input, - : output) and only output is universally quantified
% Left = built_in atom e.g. add_type(A,B) that has corresponding ModeDecls
% ModeDecl = e.g. built_in_mode(add_type('-','+')).

%%%%<<<<<<<< ユーザ定義組み込み述語 by T.Sato Feb. 2004 <<<<<<<<
% 左辺がユーザ定義述語の場合
% Left = q(.,.)はソースプログラムで
%   built_in_mode(q(+,-)) の形の宣言を持っていなければならない
goal_compile(TEMP,All,left(Left),RIGHT1,Compiled):-
   compound(Left),
   functor(Left,FuncName,Arity),
   functor(MGA,FuncName,Arity),          % MGA = most general atom
   clause(built_in_mode(MGA),true),      % q(.,.) has a model decl.

   clause(built_in_mode(ModeDecl),true), % let's find a matching mode
   functor(ModeDecl,FuncName,Arity),     % by backtracking that is
   get_x('+',Left,ModeDecl,InputL),      % compilable
   get_variable(InputL,InputV),
   variable_join(InputV,All,[]), % 'All' must not quantify input var.
   !,                            % now a compilable mode found
   compound(RIGHT1),
   RIGHT1 =.. [right|Right1],
   ( clause(fo_trace(yes),true)
   -> format("  => Built_in_mode used: ~w~n",[ModeDecl])
   ; true ),
   goal_compile_user_built_in(TEMP,All,Left,ModeDecl,Right1,Compiled).

%%%%>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

%%%%<<<<<<<< ユーザ定義確率組み込み述語 by T.Sato Mar. 2004 <<<<<<<<
% 左辺がユーザ定義確率述語の場合
% Left = q(.,.)はソースプログラムで
%   built_in_p_mode(q(+,-)) の形の宣言を持っていなければならない
goal_compile(TEMP,All,left(Left),RIGHT1,Compiled):-
   compound(Left),
   \+RIGHT1=right(fail),         % <= to avoid infifite looping
   functor(Left,FuncName,Arity),
   functor(MGA,FuncName,Arity),            % MGA = most general atom
   clause(built_in_p_mode(MGA),true),!,    % q(.,.) has a p_model decl.

   clause(built_in_p_mode(ModeDecl),true), % let's find a matching mode
   functor(ModeDecl,FuncName,Arity),       % by backtracking that is
   get_x('+',Left,ModeDecl,InputL),        % compilable
   get_variable(InputL,InputV),
   variable_join(InputV,All,[]), % 'All' must not quantify input var.
   !,                            % now a compilable mode found
   compound(RIGHT1),
   RIGHT1 =.. [right|Right1],
   ( clause(fo_trace(yes),true)
   -> format("  => Built_in_p_mode used: ~w~n",[ModeDecl])
   ; true ),
   goal_compile_user_p_built_in(TEMP,All,Left,ModeDecl,Right1,Compiled).

%%%%>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

% Input and Output of Terms
% 左辺がread(Term)の場合
goal_compile(TEMP,All,left(Left),RIGHT1,Compiled):-
   compound(Left),
   Left = read(Term),
   compound(RIGHT1),
   RIGHT1 =.. [right|Right1],!, 
   goal_compile_read(TEMP,All,Term,Right1,Compiled).

% read(Stream,Term)の場合
goal_compile(TEMP,All,left(Left),RIGHT1,Compiled):-
   compound(Left),
   Left = read(Stream,Term),
   compound(RIGHT1),
   RIGHT1 =.. [right|Right1],!, 
   goal_compile_read(TEMP,All,Stream,Term,Right1,Compiled).

% read_term(Term,Option)の場合
goal_compile(TEMP,All,left(Left),RIGHT1,Compiled):-
   compound(Left),
   Left = read_term(Stream,Term),
   compound(RIGHT1),
   RIGHT1 =.. [right|Right1],!, 
   goal_compile_read_term(TEMP,All,Stream,Term,Right1,Compiled).

% read_term(Stream,Term,Option)の場合
goal_compile(TEMP,All,left(Left),RIGHT1,Compiled):-
   compound(Left),
   Left = read_term(Stream,Term,Option),
   compound(RIGHT1),
   RIGHT1 =.. [right|Right1],!, 
   goal_compile_read_term(TEMP,All,Stream,Term,Option,Right1,Compiled).

% write(Term)の場合
goal_compile(TEMP,All,left(Left),RIGHT1,Compiled):-
   compound(Left),
   Left = write(Term),
   compound(RIGHT1),
   RIGHT1 =.. [right|Right1],!, 
   goal_compile_write(TEMP,All,Term,Right1,Compiled).

% write(Stream,Term)の場合
goal_compile(TEMP,All,left(Left),RIGHT1,Compiled):-
   compound(Left),
   Left = write(Stream,Term),
   compound(RIGHT1),
   RIGHT1 =.. [right|Right1],!, 
   goal_compile_write(TEMP,All,Stream,Term,Right1,Compiled).

% write_canonical(Term)の場合
goal_compile(TEMP,All,left(Left),RIGHT1,Compiled):-
   compound(Left),
   Left = write_canonical(A),
   compound(RIGHT1),
   RIGHT1 =.. [right|Right1],!, 
   goal_compile_write_canonical(TEMP,All,A,Right1,Compiled).

% write_canonical(Stream,Term)の場合
goal_compile(TEMP,All,left(Left),RIGHT1,Compiled):-
   compound(Left),
   Left = write_canonical(Stream,Term),
   compound(RIGHT1),
   RIGHT1 =.. [right|Right1],!, 
   goal_compile_write_canonical(TEMP,All,Stream,Term,Right1,Compiled).

% writeq(Term)の場合
goal_compile(TEMP,All,left(Left),RIGHT1,Compiled):-
   compound(Left),
   Left = writeq(A),
   compound(RIGHT1),
   RIGHT1 =.. [right|Right1],!, 
   goal_compile_writeq(TEMP,All,A,Right1,Compiled).

% writeq(Stream,Term)の場合
goal_compile(TEMP,All,left(Left),RIGHT1,Compiled):-
   compound(Left),
   Left = writeq(Stream,Term),
   compound(RIGHT1),
   RIGHT1 =.. [right|Right1],!, 
   goal_compile_writeq(TEMP,All,Stream,Term,Right1,Compiled).

% write_term(Term,Option)の場合
goal_compile(TEMP,All,left(Left),RIGHT1,Compiled):-
   compound(Left),
   Left = write(Term,Option),
   compound(RIGHT1),
   RIGHT1 =.. [right|Right1],!, 
   goal_compile_write_term(TEMP,All,Term,Option,Right1,Compiled).

% write_term(Stream,Term,Option)の場合
goal_compile(TEMP,All,left(Left),RIGHT1,Compiled):-
   compound(Left),
   Left = write_term(Stream,Term,Option),
   compound(RIGHT1),
   RIGHT1 =.. [right|Right1],!, 
   goal_compile_write_term(TEMP,All,Stream,Term,Option,Right1,Compiled).

% Character Input Output
%  左辺がnlの場合
goal_compile(TEMP,All,left(Left),RIGHT1,Compiled):-
   Left = nl,
   compound(RIGHT1),
   RIGHT1 =.. [right|Right1],!,
   goal_compile_nl(TEMP,All,Right1,Compiled).

% 左辺がnl(Stream)の場合
goal_compile(TEMP,All,left(Left),RIGHT1,Compiled):-
   Left = nl(Stream),
   compound(RIGHT1),
   RIGHT1 =.. [right|Right1],!,
   goal_compile_nl(TEMP,All,Stream,Right1,Compiled).

% 左辺がget0(N)の場合
goal_compile(TEMP,All,left(Left),RIGHT1,Compiled):-
   compound(Left),
   Left = get0(N),
   compound(RIGHT1),
   RIGHT1 =.. [right|Right1],!, 
   goal_compile_get0(TEMP,All,N,Right1,Compiled).

% 左辺がget0(Stream,Term)の場合
goal_compile(TEMP,All,left(Left),RIGHT1,Compiled):-
   compound(Left),
   Left = get0(Stream,Term),
   compound(RIGHT1),
   RIGHT1 =.. [right|Right1],!, 
   goal_compile_get0(TEMP,All,Stream,Term,Right1,Compiled).

% 左辺がpeek_char(Term)の場合
goal_compile(TEMP,All,left(Left),RIGHT1,Compiled):-
   compound(Left),
   Left = peek_char(Term),
   compound(RIGHT1),
   RIGHT1 =.. [right|Right1],!, 
   goal_compile_peek_char(TEMP,All,Term,Right1,Compiled).

% 左辺がpeek_char(Stream,Term)の場合
goal_compile(TEMP,All,left(Left),RIGHT1,Compiled):-
   compound(Left),
   Left = peek_char(Stream,Term),
   compound(RIGHT1),
   RIGHT1 =.. [right|Right1],!, 
   goal_compile_peek_char(TEMP,All,Stream,Term,Right1,Compiled).

% 左辺がput(Term)の場合
goal_compile(TEMP,All,left(Left),RIGHT1,Compiled):-
   compound(Left),
   Left = put(Term),
   compound(RIGHT1),
   RIGHT1 =.. [right|Right1],!, 
   goal_compile_put(TEMP,All,Term,Right1,Compiled).

% 左辺がput(Stream,Term)の場合
goal_compile(TEMP,All,left(Left),RIGHT1,Compiled):-
   compound(Left),
   Left = put(Stream,Term),
   compound(RIGHT1),
   RIGHT1 =.. [right|Right1],!, 
   goal_compile_put(TEMP,All,Stream,Term,Right1,Compiled).

% Stream I/O
% 左辺が open(File,Mode,Stream)の場合
goal_compile(TEMP,All,left(Left),RIGHT1,Compiled):-
   compound(Left),
   Left = open(File,Mode,Stream),
   compound(RIGHT1),
   RIGHT1 =.. [right|Right1],!, 
   goal_compile_open(TEMP,All,File,Mode,Stream,Right1,Compiled).

% 左辺がopen(File,Mode,Stream,Option)の場合
goal_compile(TEMP,All,left(Left),RIGHT1,Compiled):-
   compound(Left),
   Left = open(File,Mode,Stream,Option),
   compound(RIGHT1),
   RIGHT1 =.. [right|Right1],!, 
   goal_compile_open(TEMP,All,File,Mode,Stream,Option,Right1,Compiled).

% 左辺がclose(X)の場合
goal_compile(TEMP,All,left(Left),RIGHT1,Compiled):-
 compound(Left),
 Left = close(X),
 compound(RIGHT1),
 RIGHT1 =.. [right|Right1],!, 
 goal_compile_close(TEMP,All,X,Right1,Compiled).

% 左辺がcurrent_input(Stream)の場合
goal_compile(TEMP,All,left(Left),RIGHT1,Compiled):-
   compound(Left),
   Left = current_input(Stream),
   compound(RIGHT1),
   RIGHT1 =.. [right|Right1],!, 
   goal_compile_current_input(TEMP,All,Stream,Right1,Compiled).

% 左辺がcurrent_output(Stream)の場合
goal_compile(TEMP,All,left(Left),RIGHT1,Compiled):-
   compound(Left),
   Left = current_output(Stream),
   compound(RIGHT1),
   RIGHT1 =.. [right|Right1],!, 
   goal_compile_current_output(TEMP,All,Stream,Right1,Compiled).

% 左辺がset_input(Stream)の場合
goal_compile(TEMP,All,left(Left),RIGHT1,Compiled):-
   compound(Left),
   Left = set_input(Stream),
   compound(RIGHT1),
   RIGHT1 =.. [right|Right1],!, 
   goal_compile_set_input(TEMP,All,Stream,Right1,Compiled).

% 左辺がset_output(Stream)の場合
goal_compile(TEMP,All,left(Left),RIGHT1,Compiled):-
   compound(Left),
   Left = set_output(Stream),
   compound(RIGHT1),
   RIGHT1 =.. [right|Right1],!, 
   goal_compile_set_output(TEMP,All,Stream,Right1,Compiled).

% 左辺がflush_outputの場合
goal_compile(TEMP,All,left(Left),RIGHT1,Compiled):-
   Left = flush_output ,
   compound(RIGHT1),
   RIGHT1 =.. [right|Right1],!,
   goal_compile_flush_output(TEMP,All,Right1,Compiled).

% flush_output(Stream)の場合
goal_compile(TEMP,All,left(Left),RIGHT1,Compiled):-
   compound(Left),
   Left = flush_output(Stream),
   compound(RIGHT1),
   RIGHT1 =.. [right|Right1],!, 
   goal_compile_flush_output(TEMP,All,Stream,Right1,Compiled).

% 左辺がset_stream_position(Stream,Position)の場合
goal_compile(TEMP,All,left(Left),RIGHT1,Compiled):-
   compound(Left),
   Left = set_stream_position(Stream,Position),
   compound(RIGHT1),
   RIGHT1 =.. [right|Right1],!, 
   goal_compile_set_stream_position(TEMP,All,Stream,Position,Right1,Compiled).

% 左辺がformat(A,B)の場合
goal_compile(TEMP,All,left(Left),RIGHT1,Compiled):-
   compound(Left),
   Left = format(A,B),
   compound(RIGHT1),
   RIGHT1 =.. [right|Right1],!, 
   goal_compile_format(TEMP,All,A,B,Right1,Compiled).

%------- Arithmetic ---------
% 左辺が Z is Xの場合
goal_compile(TEMP,All,left(Left),RIGHT1,Compiled):-
   compound(Left),
   Left = (Z is X),
   compound(RIGHT1),
   RIGHT1 =.. [right|Right1],!, 
   goal_compile_is(TEMP,All,Z,X,Right1,Compiled).

% 左辺がX =:= Yの場合
goal_compile(TEMP,All,left(Left),RIGHT1,Compiled):-
   compound(Left),
   Left = ( X =:= Y),
   compound(RIGHT1),
   RIGHT1 =.. [right|Right1],!, 
   goal_compile_eq_col_eq(TEMP,All,X,Y,Right1,Compiled).

% 左辺がX =\= Yの場合
goal_compile(TEMP,All,left(Left),RIGHT1,Compiled):-
   compound(Left),
   Left = ( X =\= Y),
   compound(RIGHT1),
   RIGHT1 =.. [right|Right1],!, 
   goal_compile_eq_sla_eq(TEMP,All,X,Y,Right1,Compiled).

% 左辺がX < Yの場合
goal_compile(TEMP,All,left(Left),RIGHT1,Compiled):-
   compound(Left),
   Left = (X < Y),
   compound(RIGHT1),
   RIGHT1 =.. [right|Right1],!, 
   goal_compile_le(TEMP,All,X,Y,Right1,Compiled).

% 左辺がX > Yの場合
goal_compile(TEMP,All,left(Left),RIGHT1,Compiled):-
   compound(Left),
   Left = (X > Y),
   compound(RIGHT1),
   RIGHT1 =.. [right|Right1],!, 
   goal_compile_ge(TEMP,All,X,Y,Right1,Compiled).

% 左辺がX =< Yの場合
goal_compile(TEMP,All,left(Left),RIGHT1,Compiled):-
 compound(Left),
 Left = ( X =< Y),
 compound(RIGHT1),
 RIGHT1 =.. [right|Right1],!, 
 goal_compile_leq(TEMP,All,X,Y,Right1,Compiled).

% 左辺がX >= Yの場合
goal_compile(TEMP,All,left(Left),RIGHT1,Compiled):-
   compound(Left),
   Left = ( X >= Y),
   compound(RIGHT1),
   RIGHT1 =.. [right|Right1],!, 
   goal_compile_geq(TEMP,All,X,Y,Right1,Compiled).

% Comparison of Terms
% 左辺がX == Yの場合
goal_compile(TEMP,All,left(Left),RIGHT1,Compiled):-
   compound(Left),
   Left = (X == Y),
   compound(RIGHT1),
   RIGHT1 =.. [right|Right1],!, 
   goal_compile_eq_eq(TEMP,All,X,Y,Right1,Compiled).

% 左辺がX \== Yの場合
goal_compile(TEMP,All,left(Left),RIGHT1,Compiled):-
 compound(Left),
 Left = (X \== Y),
 compound(RIGHT1),
 RIGHT1 =.. [right|Right1],!, 
 goal_compile_not_eq_eq(TEMP,All,X,Y,Right1,Compiled).

%----- Meta Logic ---------
% 左辺がvar(X)の場合
goal_compile(TEMP,All,left(Left),RIGHT1,Compiled):-
   compound(Left),
   Left = (var(X)),
   compound(RIGHT1),
   RIGHT1 =.. [right|Right1],!, 
   goal_compile_var(TEMP,All,X,Right1,Compiled).

% 左辺がnonvar(X)の場合
goal_compile(TEMP,All,left(Left),RIGHT1,Compiled):-
   compound(Left),
   Left = (nonvar(X)),
   compound(RIGHT1),
   RIGHT1 =.. [right|Right1],!, 
   goal_compile_nonvar(TEMP,All,X,Right1,Compiled).

% 左辺がatom(X)の場合
goal_compile(TEMP,All,left(Left),RIGHT1,Compiled):-
   compound(Left),
   Left = (atom(X)),
   compound(RIGHT1),
   RIGHT1 =.. [right|Right1],!, 
   goal_compile_atom(TEMP,All,X,Right1,Compiled).

% 左辺がfloat(X)の場合
goal_compile(TEMP,All,left(Left),RIGHT1,Compiled):-
   compound(Left),
   Left = (float(X)),
   compound(RIGHT1),
   RIGHT1 =.. [right|Right1],!, 
   goal_compile_float(TEMP,All,X,Right1,Compiled).

% 左辺がinteger(X)の場合
goal_compile(TEMP,All,left(Left),RIGHT1,Compiled):-
   compound(Left),
   Left = (integer(X)),
   compound(RIGHT1),
   RIGHT1 =.. [right|Right1],!, 
   goal_compile_integer(TEMP,All,X,Right1,Compiled).

% 左辺がnumber(X)の場合
goal_compile(TEMP,All,left(Left),RIGHT1,Compiled):-
   compound(Left),
   Left = (number(X)),
   compound(RIGHT1),
   RIGHT1 =.. [right|Right1],!, 
   goal_compile_number(TEMP,All,X,Right1,Compiled).

% 左辺がatomic(X)の場合
goal_compile(TEMP,All,left(Left),RIGHT1,Compiled):-
   compound(Left),
   Left = (atomic(X)),
   compound(RIGHT1),
   RIGHT1 =.. [right|Right1],!, 
   goal_compile_atomic(TEMP,All,X,Right1,Compiled).

% 左辺がsimple(X)の場合
goal_compile(TEMP,All,left(Left),RIGHT1,Compiled):-
   compound(Left),
   Left = (simple(X)),
   compound(RIGHT1),
   RIGHT1 =.. [right|Right1],!, 
   goal_compile_simple(TEMP,All,X,Right1,Compiled).

% 左辺がcompound(X)の場合
goal_compile(TEMP,All,left(Left),RIGHT1,Compiled):-
   compound(Left),
   Left = (compound(X)),
   compound(RIGHT1),
   RIGHT1 =.. [right|Right1],!, 
   goal_compile_compound(TEMP,All,X,Right1,Compiled).

% 左辺がfunctor(Term,Name,Arity)の場合
goal_compile(TEMP,All,left(Left),RIGHT1,Compiled):-
   compound(Left),
   Left = (functor(Term,Name,Arity)),
   compound(RIGHT1),
   RIGHT1 =.. [right|Right1],!, 
   goal_compile_functor(TEMP,All,Term,Name,Arity,Right1,Compiled).

% 左辺がarg(ArgNo,Name,Arg)の場合
goal_compile(TEMP,All,left(Left),RIGHT1,Compiled):-
   compound(Left),
   Left = (arg(ArgNo,Name,Arg)),
   compound(RIGHT1),
   RIGHT1 =.. [right|Right1],!, 
   goal_compile_arg(TEMP,All,ArgNo,Name,Arg,Right1,Compiled).

% 左辺がTerm =.. Listの場合
goal_compile(TEMP,All,left(Left),RIGHT1,Compiled):-
   compound(Left),
   Left = (Term =.. List),
   compound(RIGHT1),
   RIGHT1 =.. [right|Right1],!, 
   goal_compile_eq_dot_dot(TEMP,All,Term,List,Right1,Compiled).

% 左辺がname(Const,CharList)の場合
goal_compile(TEMP,All,left(Left),RIGHT1,Compiled):-
   compound(Left),
   Left = name(Const,CharList),
   compound(RIGHT1),
   RIGHT1 =.. [right|Right1],!, 
   goal_compile_name(TEMP,All,Const,CharList,Right1,Compiled).

% 左辺がatom_chars(Const,CharList)の場合
goal_compile(TEMP,All,left(Left),RIGHT1,Compiled):-
   compound(Left),
   Left = atom_chars(Const,CharList),
   compound(RIGHT1),
   RIGHT1 =.. [right|Right1],!, 
   goal_compile_atom_chars(TEMP,All,Const,CharList,Right1,Compiled).

% 左辺がnumber_chars(Const,CharList)の場合
goal_compile(TEMP,All,left(Left),RIGHT1,Compiled):-
   compound(Left),
   Left = number_chars(Const,CharList),
   compound(RIGHT1),
   RIGHT1 =.. [right|Right1],!, 
   goal_compile_number_chars(TEMP,All,Const,CharList,Right1,Compiled).

% 左辺がasserta(Clause)の場合
goal_compile(TEMP,All,left(Left),RIGHT1,Compiled):-
   compound(Left),
   Left = asserta(Clause),
   compound(RIGHT1),
   RIGHT1 =.. [right|Right1],!, 
   goal_compile_asserta(TEMP,All,Clause,Right1,Compiled).

% 左辺がassertz(Clause)の場合
goal_compile(TEMP,All,left(Left),RIGHT1,Compiled):-
   compound(Left),
   Left = assertz(Clause),
   compound(RIGHT1),
   RIGHT1 =.. [right|Right1],!, 
   goal_compile_assertz(TEMP,All,Clause,Right1,Compiled).

% 左辺がassert(Clause)の場合
goal_compile(TEMP,All,left(Left),RIGHT1,Compiled):-
   compound(Left),
   Left = assert(Clause),
   compound(RIGHT1),
   RIGHT1 =.. [right|Right1],!, 
   goal_compile_assert(TEMP,All,Clause,Right1,Compiled).

% 左辺がretract(Clause)の場合
goal_compile(TEMP,All,left(Left),RIGHT1,Compiled):-
   compound(Left),
   Left = retract(Clause),
   compound(RIGHT1),
   RIGHT1 =.. [right|Right1],!, 
   goal_compile_retract(TEMP,All,Clause,Right1,Compiled).

% 左辺が findall(X,Y,Z)の場合
goal_compile(TEMP,All,left(Left),RIGHT1,Compiled):-
   compound(Left),
   ( Left = findall(_,_,_) ; Left = setof(_,_,_) ; Left = bagof(_,_,_) ),
   compound(RIGHT1),
   RIGHT1 =.. [right|Right1],!, 
   goal_compile_findall(TEMP,All,Left,Right1,Compiled).


%------ Miscellaneous -------
% 左辺が(A=B)の場合
goal_compile(TEMP,All,left(Left),RIGHT1,Compiled):-
   compound(Left),
   Left = (A=B),
   compound(RIGHT1),
   RIGHT1 =.. [right|Right1],!,
   goal_compile_equal(TEMP,All,A,B,Right1,Compiled).

% 左辺がlength(List,Length)の場合
goal_compile(TEMP,All,left(Left),RIGHT1,Compiled):-
   compound(Left),
   Left = length(List,Length),
   compound(RIGHT1),
   RIGHT1 =.. [right|Right1],!, 
   goal_compile_length(TEMP,All,List,Length,Right1,Compiled).

% 左辺がsort(In,Out)の場合
goal_compile(TEMP,All,left(Left),RIGHT1,Compiled):-
   compound(Left),
   Left = sort(In,Out),
   compound(RIGHT1),
   RIGHT1 =.. [right|Right1],!, 
   goal_compile_sort(TEMP,All,In,Out,Right1,Compiled).

%%%%%%%% For PRISM %%%%%%%%%%
% 左辺がmsw(I,V)の場合
goal_compile(TEMP,All,left(Left),RIGHT1,Compiled):-
  compound(Left),
  Left = msw(I,V),
  compound(RIGHT1),
  RIGHT1 =.. [right|Right1],!, 
  goal_compile_msw(TEMP,All,msw(I,V),Right1,Compiled).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 左辺が\+G の場合
% 
goal_compile(TEMP,All,left(Left),RIGHT1,Compiled):-
   Left = (\+ _),
   compound(RIGHT1),
   RIGHT1 =.. [right|Right1],!, 
   goal_compile_naf(TEMP,All,Left,Right1,Compiled).

%------------- (A;B)------------------
% 左辺が(A;B) の場合
goal_compile(TEMP,All,left(Left),RIGHT1,Compiled):-
   compound(Left),
   Left = (A ; B),!,
   compound(RIGHT1),

   conjunction_to_list(A,AList),
   generate_left(AList,ALEFT),
   get_free_variable(AList,AListVar),
   get_free_variable(B,BVar),
   variable_union(AListVar,BVar,ABVar),
   variable_join(All,ABVar,AllAndABVar),
   same_variable_list(All,AllAndABVar),!,
   variable_join(All,AListVar,AAll),
   generate_all(AAll,ALEFT,RIGHT1,AALL),!,
   goal_compile(TEMP,AALL,ACompiled),

   generate_left([B],BLEFT),
   variable_join(All,BVar,BAll),
   generate_all(BAll,BLEFT,RIGHT1,BALL),!,
   goal_compile(TEMP,BALL,BCompiled),

   list_to_conjunction([ACompiled,BCompiled],Compiled).

% 左辺が(A,B) の場合
goal_compile(TEMP,All,left(Left),RIGHT1,Compiled):-
   compound(Left),
   functor(Left,(','),2),!,
   conjunction_to_list(Left,LeftList),
     % Left=(A,B,C) => LEFT2=left(A,B,C)
   generate_left(LeftList,LEFT2),
   goal_compile(TEMP,All,LEFT2,RIGHT1,Compiled).

% 左辺がexist(ExistVar,Exist)の場合
goal_compile(TEMP,All,left(Left),RIGHT1,Compiled):-
   compound(Left),
   Left =.. [exist, ExistVar, Exist],!,
   variable_union(All,ExistVar,All2),
   conjunction_to_list(Exist,ExistList),
   generate_left(ExistList,LEFT2),
   goal_compile(TEMP,All2,LEFT2,RIGHT1,Compiled).

% 左辺がdynamicで指定されたものの場合。
% 実行時にgroundなら動く事にする
goal_compile(TEMP,[],left(Left),RIGHT1,Compiled):-
    compound(Left),
    functor(Left,LeftName,LeftArity),
    clause(for_dynamic(LeftName,LeftArity),true),!,
    RIGHT1 =.. [right|Right1],
    check_body(TEMP,Right1,Right2),
    list_to_conjunction([ground(Left)|[Left|Right2]],OrLeft),
    generate_term((;),[ OrLeft, (ground(Left),\+(Left)) ],Compiled).

% <ケース 2.3> user defined atom
% 左辺がユーザ定義アトムの場合： all([Y],(q(X,Y) -> ...)) の形
%                                         -----左辺
%
% この上のclausesで、left(Left)->...の形（left/1）：
%   all([Y],((Prolog_built_in(X,Y)) -> ...)), 
%   all([Y],((User_built_in(X,Y)) -> ...)),
%   all([X],((A & B) -> ...)) all([X],((A ; B) -> ...))
%   all([X],((\+G) -> ...)), all([X],( msw(I,V) -> ...)),
% のパターンを処理済み

goal_compile(TEMP,All,left(Left),RIGHT,Compiled):-
      % 左辺が一つのアトムで、all(All,IMPLY)ではない事を確認
      % Name==all の場合は <ケース 4> で処理
   functor(Left,Name,_),
   Name \== all,!,
      % モードパターンを作る
   get_mode_pattern(All,Left,RIGHT,Mode),
      % closure clause と continuation clause を作る
   generate_closure_continuation_clause(TEMP,All,Left,RIGHT,Mode,Compiled).
  
% <ケース 3>
% 左辺Leftが連言の場合(left/2,left/3...の場合）
%  all(_,left(A,B,..) -> ...)の形のコンパイル
goal_compile(TEMP,All1,LEFT1,RIGHT1,Compiled):-
      % 左辺が連言である事を確認
      % e.g. LEFT1 =left(A,B,C) for conjunctive goals (A,B,C)
   compound(LEFT1),
   functor(LEFT1,left,LE1Arity),
   LE1Arity > 1,
   compound(RIGHT1),!,
   generate_all(All1,LEFT1,RIGHT1,ALL1),
     % 連言を変形
   and_to_imply(ALL1,ALL2),!,
   goal_compile(TEMP,ALL2,Compiled).

%%%%%>>>>>>>>>> 修正 T.Sato Oct.2003 >>>>>>>>>
% <ケース 4>
% 左辺Leftが all(All,IMPLY) の場合
goal_compile(TEMP,[],left(ALL),RIGHT,Compiled):-
     % ２重否定の場合
     % 入力がall([], all(V,(A -> fail)) -> fail) である事を確認
   ALL = all(_,imply(left(A),right(fail))),
   RIGHT =right(fail),
   check_body(TEMP,[A],A2),
   list_to_conjunction(A2,Compiled),!.

goal_compile(TEMP,[],left(ALL2),RIGHT1R,Compiled):-
     % 入力がall([], all(V,(A -> B)) -> C) である事を確認
     % A,B,Cは、全てatoms
     % (exist(V,(A & not(B))) or (all(V,(A->B)) & C) を作る
   ALL2 = all(V,imply(AA,BB)), 
     % all(V,imply(AA,BB)) を実行可能にする
   goal_compile(TEMP,V,AA,BB,C_ALL2),
   AA =.. [left | A],
   BB =.. [right| B],  % B \==[fail]
   RIGHT1R =.. [right| C],!,
     % B -> failを作り、実行可能にする
   generate_left(B,LEFT2),
   generate_all([],LEFT2,right(fail),NotBB),!,
     % BB = right(fail)の場合 AndRight = true になる
   goal_compile(TEMP,NotBB,AndRight), 
     % Aを実行可能にする
   check_body(TEMP,A,AndLeftList),
     % Cを実行可能にする
   check_body(TEMP,C,C2),
   OrRightList = [C_ALL2|C2],
     % (exist(V,(A & not(B))) or (all(V,(A->B)) & C) を作る
   goal_compile_case4_sub(AndLeftList,AndRight,
                        OrRightList,V,Compiled),!.
%%%%%<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


%%%%%%%% For HPSG  %%%%%%%%%%
goal_compile(TEMP,All,left(ALL1L),RIGHT1R,Compiled):-
     % 入力がall(All,all([],(unify(U1,U2,U3) -> fail)) -> C1) である事を確認
     % exist([..,U3],not(unify(U1,U2,U3))) = true である事を利用する
   ALL1L = all([],imply(left(unify(U1,U2,U3)),right(fail))),
   get_free_variable(RIGHT1R,RV),
   variable_join(All,RV,[]),
   variable_join(All,[U1,U2],[]),
   variable_member(U3,All),
   RIGHT1R =.. [right|C1],!,
   check_body(TEMP,C1,C2),      % C1を実行可能にする
   list_to_conjunction(C2,Compiled).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% その他
goal_compile(_,ALL,LEFT,RIGHT,_):-
   numbervars(all(ALL,(LEFT ->RIGHT)),0,_),
   format("~nFailed goal_compile/5 at < Case 4 >: ~w~n",
           [all(ALL,(LEFT ->RIGHT))]),
   retract(goal_compile_fail(_)),
   assertz(goal_compile_fail(1)),!,
   fail.

goal_compile_case4_sub(_,_,[true],_,true):-!.

goal_compile_case4_sub(AndLeftList1,true,[fail],V,Compiled):-!,
   some_variables_to_new_variable(V,AndLeftList1,AndLeftList2),
   list_to_conjunction(AndLeftList2,Compiled).

goal_compile_case4_sub(AndLeftList,AndRight,[fail],All1l,Compiled):-
   fo_append(AndLeftList,[AndRight],AndList1),
   some_variables_to_new_variable(All1l,AndList1,AndList2),
   list_to_conjunction(AndList2,Compiled).

goal_compile_case4_sub(AndLeftList1,true,OrRightList,All1l,Compiled):-
   some_variables_to_new_variable(All1l,AndLeftList1,AndLeftList2),
   list_to_conjunction(AndLeftList2,OrLeft),
   list_to_conjunction(OrRightList,OrRight),
   generate_term((;),[OrLeft,OrRight],Compiled).

goal_compile_case4_sub(AndLeftList,AndRight,OrRightList,All1l,Compiled):-
   fo_append(AndLeftList,[AndRight],OrLeftList1),
   some_variables_to_new_variable(All1l,OrLeftList1,OrLeftList2),
   list_to_conjunction(OrLeftList2,OrLeft),
   list_to_conjunction(OrRightList,OrRight),
   generate_term((;),[OrLeft,OrRight],Compiled).

%-----------------------------------------
% 左辺が built-in 述語で
%  失敗したときのメッセージ出力
goal_compile_built_in_error:-
   format("   Failed at built-in predicate.",[]),
   retract(goal_compile_fail(_)),
   assertz(goal_compile_fail(1)),!,
   fail.

goal_compile_built_in_error(Term):-
   numbervars(Term,0,_),
   format("   Failed at built-in predicate ""~w"".~n",[Term]),
   retract(goal_compile_fail(_)),
   assertz(goal_compile_fail(1)),!,
   fail.

%-------------------------------------------
% 以下個別 built_in 述語のコンパイル

% 左辺が true の場合
goal_compile_true(TEMP,[],Right1,Compiled):-
   check_body(TEMP,Right1,Right2),
   list_to_conjunction(Right2,Compiled).
goal_compile_true(_,_,_,_):-
   goal_compile_built_in_error(true).

% 左辺が cut の場合
goal_compile_cut(TEMP,[],Right1,Compiled):-
   check_body(TEMP,Right1,Right2),
   list_to_conjunction(['!'|Right2],Compiled).

goal_compile_cut(_,_,_,_):-
   goal_compile_built_in_error('!').

% 左辺が必ず成功し、副作用のある built_in/0 の場合
goal_compile_built_in_0(TEMP,[],Left,Right1,Compiled):-
   check_body(TEMP,Right1,Right2),
   list_to_conjunction([Left|Right2],Compiled).

goal_compile_built_in_0(_,All,Left,Right1,_):-
   goal_compile_built_in_error(all(All,(Left -> Right1))).

% Input and Output of Terms
% 左辺がread(Term)の場合
%goal_compile_read(TEMP,All,Term,Right1,Compiled)
goal_compile_read(TEMP,All,Term,Right1,Compiled):-
   same_variable_list(All,[Term]),!,
   check_body(TEMP,Right1,Right2),
   list_to_conjunction([(read(Term))|Right2],Compiled).

goal_compile_read(_,_,_,_,_):-
   goal_compile_built_in_error.

% 左辺がread(Stream,Term)の場合
% goal_compile_read(TEMP,All,Stream,Term,Right1,Compiled)
goal_compile_read(TEMP,[],Stream,Term,Right1,Compiled):-
   check_body(TEMP,Right1,Right2),
   list_to_conjunction([(read(Stream,Term))|Right2],Compiled).

goal_compile_read(TEMP,All,Stream,Term,Right1,Compiled):-
   same_variable_list(All,[Term]),!,
   check_body(TEMP,Right1,Right2),
   list_to_conjunction([(read(Stream,Term))|Right2],Compiled).

goal_compile_read(_,_,_,_,_,_):-
   goal_compile_built_in_error.

% 左辺がread_term(Term,Option)の場合
% goal_compile_read_term(TEMP,All,Term,Option,Right1,Compiled)
goal_compile_read_term(TEMP,All,Term,Option,Right1,Compiled):-
   same_variable_list(All,[Term]),!,
   check_body(TEMP,Right1,Right2),
   list_to_conjunction([(read_term(Term,Option))|Right2],Compiled).

goal_compile_read_term(_,_,_,_,_,_):-
   goal_compile_built_in_error.

% 左辺がread_term(Stream,Term,Option)の場合
% goal_compile_read_term(TEMP,All,Stream,Term,Option,Right1,Compiled)
goal_compile_read_term(TEMP,All,Stream,Term,Option,Right1,Compiled):-
   same_variable_list(All,[Term]),!,
   check_body(TEMP,Right1,Right2),
   list_to_conjunction([(read_term(Stream,Term,Option))|Right2],Compiled).

goal_compile_read_term(_,_,_,_,_,_,_):-
  goal_compile_built_in_error.

% 左辺がwrite(Term)の場合
% goal_compile_write(TEMP,[],Term,Right1,Compiled)
goal_compile_write(TEMP,[],Term,Right1,Compiled):-
   check_body(TEMP,Right1,Right2),
   list_to_conjunction([(write(Term))|Right2],Compiled).

goal_compile_write(_,_,_,_,_):-
   goal_compile_built_in_error.

%左辺がwrite(Stream,Term)の場合
%goal_compile_write(TEMP,[],Stream,Term,Right1,Compiled)
goal_compile_write(TEMP,[],Stream,Term,Right1,Compiled):-
   check_body(TEMP,Right1,Right2),
   list_to_conjunction([(write(Stream,Term))|Right2],Compiled).

goal_compile_write(_,_,_,_,_,_):-
 goal_compile_built_in_error.

%左辺がwrite_canonical(Term)の場合
%goal_compile_write_canonical(TEMP,[],Term,Right1,Compiled)
goal_compile_write_canonical(TEMP,[],Term,Right1,Compiled):-
   check_body(TEMP,Right1,Right2),
   list_to_conjunction([(write_canonical(Term))|Right2],Compiled).

goal_compile_write_canonical(_,_,_,_,_):-
   goal_compile_built_in_error.

%左辺がwrite_canonical(Stream,Term)の場合
%goal_compile_write_canonical(TEMP,[],Stream,Term,Right1,Compiled)
goal_compile_write_canonical(TEMP,[],Stream,Term,Right1,Compiled):-
   check_body(TEMP,Right1,Right2),
   list_to_conjunction([(write_canonical(Stream,Term))|Right2],Compiled).

goal_compile_write_canonical(_,_,_,_,_,_):-
 goal_compile_built_in_error(write_canonical).

% 左辺がwriteq(Term)の場合
% goal_compile_writeq(TEMP,[],Term,Right1,Compiled)
goal_compile_writeq(TEMP,[],Term,Right1,Compiled):-
   check_body(TEMP,Right1,Right2),
   list_to_conjunction([(writeq(Term))|Right2],Compiled).

goal_compile_writeq(_,_,_,_,_):-
   goal_compile_built_in_error('writeq/1').

%左辺がwriteq(Stream,Term)の場合
%goal_compile_writeq(TEMP,[],Stream,Term,Right1,Compiled)
goal_compile_writeq(TEMP,[],Stream,Term,Right1,Compiled):-
 check_body(TEMP,Right1,Right2),
 list_to_conjunction([(writeq(Stream,Term))|Right2],Compiled).

goal_compile_writeq(_,_,_,_,_,_):-
 goal_compile_built_in_error('writeq/2').

%左辺がwrite_term(Term,Option)の場合
%goal_compile_write_term(TEMP,[],Term,Option,Right1,Compiled)
goal_compile_write_term(TEMP,[],Term,Option,Right1,Compiled):-
   check_body(TEMP,Right1,Right2),
   list_to_conjunction([(write_term(Term,Option))|Right2],Compiled).

goal_compile_write_term(_,_,_,_,_,_):-
 goal_compile_built_in_error('write_term/2').

%左辺がwrite_term(Stream,Term,Option)の場合
%goal_compile_write_term(TEMP,[],Stream,Term,Option,Right1,Compiled)
goal_compile_write_term(TEMP,[],Stream,Term,Option,Right1,Compiled):-
   check_body(TEMP,Right1,Right2),
   list_to_conjunction([(write_term(Stream,Term,Option))|Right2],Compiled).

goal_compile_write_term(_,_,_,_,_,_,_):-
 goal_compile_built_in_error('write_term/3').

%Character Input Output
%左辺がnlの場合
goal_compile_nl(TEMP,[],Right1,Compiled):-
 check_body(TEMP,Right1,Right2),
 list_to_conjunction(['nl'|Right2],Compiled).

goal_compile_nl(_,_,_,_):-
 goal_compile_built_in_error('nl/0').

%左辺がnl(Stream)の場合
goal_compile_nl(TEMP,[],Stream,Right1,Compiled):-
 check_body(TEMP,Right1,Right2),

 list_to_conjunction([nl(Stream)|Right2],Compiled).

goal_compile_nl(_,_,_,_,_):-
 goal_compile_built_in_error('nl/1').

% 左辺がget0(Term)の場合
% goal_compile_get0(TEMP,All,Term,Right1,Compiled)
goal_compile_get0(TEMP,All,Term,Right1,Compiled):-
 same_variable_list(All,[Term]),!,
 check_body(TEMP,Right1,Right2),
 list_to_conjunction([(get0(Term))|Right2],Compiled).

goal_compile_get0(_,_,_,_,_):-
 goal_compile_built_in_error.

%左辺がget0(Stream,Term)の場合
%goal_compile_get0(TEMP,All,Stream,Term,Right1,Compiled)
goal_compile_get0(TEMP,All,Stream,Term,Right1,Compiled):-
 same_variable_list(All,[Term]),!,
 check_body(TEMP,Right1,Right2),
 list_to_conjunction([(get0(Stream,Term))|Right2],Compiled).

goal_compile_get0(_,_,_,_,_,_):-
  goal_compile_built_in_error.

%左辺がpeek_char(Term)の場合
%goal_compile_peek_char(TEMP,All,Term,Right1,Compiled)
goal_compile_peek_char(TEMP,All,Term,Right1,Compiled):-
   same_variable_list(All,[Term]),!,
   check_body(TEMP,Right1,Right2),
   list_to_conjunction([(peek_char(Term))|Right2],Compiled).

goal_compile_peek_char(_,_,_,_,_):-
   goal_compile_built_in_error.

%左辺がpeek_char(Stream,Term)の場合
%goal_compile_peek_char(TEMP,All,Stream,Term,Right1,Compiled)
goal_compile_peek_char(TEMP,All,Stream,Term,Right1,Compiled):-
   same_variable_list(All,[Term]),!,
   check_body(TEMP,Right1,Right2),
   list_to_conjunction([(peek_char(Stream,Term))|Right2],Compiled).

goal_compile_peek_char(_,_,_,_,_,_):-
   goal_compile_built_in_error.

%左辺がput(Term)の場合
%goal_compile_put(TEMP,[],Term,Right1,Compiled)
goal_compile_put(TEMP,[],Term,Right1,Compiled):-
   check_body(TEMP,Right1,Right2),
   list_to_conjunction([(put(Term))|Right2],Compiled).

goal_compile_put(_,_,_,_,_):-
   goal_compile_built_in_error.

%左辺がput(Stream,Term)の場合
%goal_compile_put(TEMP,[],Stream,Term,Right1,Compiled)
goal_compile_put(TEMP,[],Stream,Term,Right1,Compiled):-
   check_body(TEMP,Right1,Right2),
   list_to_conjunction([(put(Stream,Term))|Right2],Compiled).

goal_compile_put(_,_,_,_,_,_):-
   goal_compile_built_in_error.

%Stream I/O
%左辺がopen(File,Mode,Stream)の場合
%goal_compile_open(TEMP,[],File,Mode,Stream,Right1,Compiled)
goal_compile_open(TEMP,[],File,Mode,Stream,Right1,Compiled):-
   check_body(TEMP,Right1,Right2),
   list_to_conjunction([(open(File,Mode,Stream))|Right2],Compiled).

goal_compile_open(TEMP,All,File,Mode,Stream,Right1,Compiled):-
   same_variable_list(All,[Stream]),!,
   check_body(TEMP,Right1,Right2),
   list_to_conjunction([(open(File,Mode,Stream))|Right2],Compiled).

goal_compile_open(_,All,File,Mode,Stream,Right,_):-
   F=all(All,(open(File,Mode,Stream)->Right)),
   goal_compile_built_in_error(F).

%左辺がopen(File,Mode,Stream,Option)の場合
%goal_compile_open(TEMP,[],File,Mode,Stream,Option,Right1,Compiled)
goal_compile_open(TEMP,[],File,Mode,Stream,Option,Right1,Compiled):-
   check_body(TEMP,Right1,Right2),
   list_to_conjunction([(open(File,Mode,Stream,Option))|Right2],Compiled).

goal_compile_open(TEMP,All,File,Mode,Stream,Option,Right1,Compiled):-
   same_variable_list(All,[Stream]),!,
   check_body(TEMP,Right1,Right2),
   list_to_conjunction([(open(File,Mode,Stream,Option))|Right2],Compiled).

goal_compile_open(_,_,_,_,_,_):-
 goal_compile_built_in_error.

% 左辺がclose(Stream)の場合
% goal_compile_close(TEMP,[],Stream,Right1,Compiled)
goal_compile_close(TEMP,[],Stream,Right1,Compiled):-
   check_body(TEMP,Right1,Right2),
   list_to_conjunction([(close(Stream))|Right2],Compiled).

goal_compile_close(_,_,_,_,_):-
   goal_compile_built_in_error.


%左辺がcurrent_input(Stream)の場合
%goal_compile_current_input(TEMP,[],Stream,Right1,Compiled)
goal_compile_current_input(TEMP,[],Stream,Right1,Compiled):-
   check_body(TEMP,Right1,Right2),
   list_to_conjunction([(current_input(Stream))|Right2],OrRight),
   generate_term((;),[\+(current_input(Stream))|[OrRight|[]]],Compiled).

goal_compile_current_input(TEMP,All,Stream,Right1,Compiled):-
   same_variable_list(All,[Stream]),!,
   check_body(TEMP,Right1,Right2),
   list_to_conjunction([(current_input(Stream))|Right2],Compiled).

goal_compile_current_input(_,_,_,_,_):-
   goal_compile_built_in_error.

% 左辺がcurrent_output(Stream)の場合
% goal_compile_current_output(TEMP,[],Stream,Right1,Compiled)
goal_compile_current_output(TEMP,[],Stream,Right1,Compiled):-
   check_body(TEMP,Right1,Right2),
   list_to_conjunction([(current_output(Stream))|Right2],OrRight),
   generate_term((;),[ \+(current_output(Stream))|[OrRight|[]]],Compiled).

goal_compile_current_output(TEMP,All,Stream,Right1,Compiled):-
   same_variable_list(All,[Stream]),!,
   check_body(TEMP,Right1,Right2),
   list_to_conjunction([(current_output(Stream))|Right2],Compiled).

goal_compile_current_output(_,_,_,_,_):-
   goal_compile_built_in_error.

% 左辺がset_input(Stream)の場合
% goal_compile_set_input(TEMP,[],Stream,Right1,Compiled)
goal_compile_set_input(TEMP,[],Stream,Right1,Compiled):-
   check_body(TEMP,Right1,Right2),
   list_to_conjunction([(set_input(Stream))|Right2],Compiled).

goal_compile_set_input(_,_,_,_,_):-
   goal_compile_built_in_error.

% 左辺がset_output(Stream)の場合
goal_compile_set_output(TEMP,[],Stream,Right1,Compiled):-
   check_body(TEMP,Right1,Right2),
   list_to_conjunction([(set_output(Stream))|Right2],Compiled).

goal_compile_set_output(_,_,_,_,_):-
 goal_compile_built_in_error.

%左辺がflush_outputの場合
goal_compile_flush_output(TEMP,[],Right1,Compiled):-
   check_body(TEMP,Right1,Right2),
   list_to_conjunction(['flush_output'|Right2],Compiled).

goal_compile_flush_output(_,_,_,_):-
   goal_compile_built_in_error.

% 左辺がflush_output(Stream)の場合
goal_compile_flush_output(TEMP,[],Stream,Right1,Compiled):-
   check_body(TEMP,Right1,Right2),
   list_to_conjunction([flush_output(Stream)|Right2],Compiled).

goal_compile_flush_output(_,_,_,_,_):-
   goal_compile_built_in_error.

% 左辺がset_stream_position(Stream,Position)の場合
goal_compile_set_stream_position(TEMP,[],Stream,Position,Right1,Compiled):-
   check_body(TEMP,Right1,Right2),
   list_to_conjunction([(set_stream_position(Stream,Position))|Right2],
          Compiled).
goal_compile_set_stream_position(_,_,_,_,_,_):-
   goal_compile_built_in_error.

% 左辺がformat(A,B)の場合
goal_compile_format(TEMP,All,A,B,Right1,Compiled):-
   get_variable([A,B],VarL),
   variable_join(VarL,All,[]),
   goal_compile_equal_sub2(All,Right1,Right2),
   check_body(TEMP,Right2,Right3),
   list_to_conjunction([format(A,B)|Right3], Compiled).
goal_compile_format(_,_,A,B,_,_):-
   goal_compile_built_in_error(format(A,B)).


%----- Arithmetic ------
% Z is Xの場合
goal_compile_is(TEMP,[],Z,X,Right1,Compiled):-
   check_body(TEMP,Right1,Right2),
   list_to_conjunction([(Z is X)|Right2],OrLeft),
   generate_term((;),[ OrLeft, \+(Z is X) ],Compiled).

goal_compile_is(TEMP,All,Z,X,Right1,Compiled):-
   same_variable_list(All,[Z]),!,
   check_body(TEMP,Right1,Right2),
   list_to_conjunction([(Z is X)|Right2],Compiled).

goal_compile_is(_,_,Z,X,_,_):-
   goal_compile_built_in_error((Z is X)).

% X =:= Yの場合
goal_compile_eq_col_eq(TEMP,[],X,Y,Right1,Compiled):-
   check_body(TEMP,Right1,Right2), 
   list_to_conjunction([(X =:= Y)|Right2],OrLeft),
   generate_term((;),[OrLeft|[ (X =\= Y) |[]]],Compiled).

goal_compile_eq_col_eq(_,_,_,_,_,_):-
   goal_compile_built_in_error(=:=).

%X =\= Yの場合
goal_compile_eq_sla_eq(TEMP,[],X,Y,Right1,Compiled):-
   check_body(TEMP,Right1,Right2), 
   list_to_conjunction([(X =\= Y)|Right2],OrLeft),
   generate_term((;),[OrLeft|[ (X =:= Y) |[]]],Compiled).

goal_compile_eq_sla_eq(_,_,_,_,_,_):-
   goal_compile_built_in_error(=\=).

%X < Yの場合
goal_compile_le(TEMP,[],X,Y,Right1,Compiled):-
   check_body(TEMP,Right1,Right2), 
   list_to_conjunction([(X < Y)|Right2],OrLeft),
   generate_term((;),[OrLeft|[ (X >= Y) |[]]],Compiled).

goal_compile_le(_,_,X,Y,_,_):-
   goal_compile_built_in_error((X <Y)).

% X > Yの場合
goal_compile_ge(TEMP,[],X,Y,Right1,Compiled):-
   check_body(TEMP,Right1,Right2), 
   list_to_conjunction([(X > Y)|Right2],OrLeft),
   generate_term((;),[OrLeft, (X =< Y) ],Compiled).

goal_compile_ge(_,_,X,Y,_,_):-
   goal_compile_built_in_error((X>Y)).

% X =< Yの場合
goal_compile_leq(TEMP,[],X,Y,Right1,Compiled):-
 check_body(TEMP,Right1,Right2), 
 list_to_conjunction([(X =< Y)|Right2],OrLeft),
 generate_term((;),[OrLeft|[ (X > Y) |[]]],Compiled).

goal_compile_leq(_,_,X,Y,_,_):-
   goal_compile_built_in_error((X =< Y)).

% X >= Yの場合
goal_compile_geq(TEMP,[],X,Y,Right1,Compiled):-
   check_body(TEMP,Right1,Right2), 
   list_to_conjunction([(X >= Y)|Right2],OrLeft),
   generate_term((;),[OrLeft|[ (X < Y) |[]]],Compiled).

goal_compile_geq(_,_,X,Y,_,_):-
   goal_compile_built_in_error((X >= Y)).

% Comparison of Terms
% X == Yの場合
% goal_compile_eq_eq(TEMP,All,X,Y,Right1,Compiled)
goal_compile_eq_eq(TEMP,All,X,Y,Right1,Compiled):-
   ( same_variable_list(All,[X,Y]),!,
        list_to_conjunction([(X = Y)|Right2],Compiled)
   ; All==[],!,
        check_body(TEMP,Right1,Right2),
        list_to_conjunction([(X == Y)|Right2],OrRight),
        generate_term((;),[(X \== Y),OrRight],Compiled)
   %; Add other cases!
   ).

goal_compile_eq_eq(_,_,X,Y,_,_):-
   goal_compile_built_in_error((X==Y)).

% X \== Yの場合
% goal_compile_not_eq_eq(TEMP,All,X,Y,Right1,Compiled)
goal_compile_not_eq_eq(TEMP,All,X,Y,Right1,Compiled):-
%   same_variable_list(All,[X,Y]),!,
   All==[],!,
   check_body(TEMP,Right1,Right2), 
   list_to_conjunction([(X \== Y)|Right2],OrRight),
   generate_term((;),[(X == Y)|[OrRight|[]]],Compiled).

goal_compile_not_eq_eq(_,_,X,Y,_,_):-
   goal_compile_built_in_error((X \== Y)).

%Meta Logic
%var(X)の場合
goal_compile_var(TEMP,[],X,Right1,Compiled):-
   check_body(TEMP,Right1,Right2),
   list_to_conjunction([var(X)|Right2],OrRight),
   generate_term((;),[nonvar(X)|[OrRight|[]]],Compiled).

goal_compile_var(_,_,X,_,_):-
 goal_compile_built_in_error(var(X)).

%nonvar(X)の場合
goal_compile_nonvar(TEMP,[],X,Right1,Compiled):-
   check_body(TEMP,Right1,Right2),
   list_to_conjunction([nonvar(X)|Right2],OrRight),
   generate_term((;),[var(X)|[OrRight|[]]],Compiled).

goal_compile_nonvar(_,_,X,_,_):-
   goal_compile_built_in_error(nonvar(X)).

% atom(X)の場合
goal_compile_atom(TEMP,[],X,Right1,Compiled):-
   check_body(TEMP,Right1,Right2),
   list_to_conjunction([atom(X)|Right2],OrRight),
   generate_term((;),[ \+(atom(X)) |[OrRight|[]]],Compiled).

goal_compile_atom(_,_,X,_,_):-
   goal_compile_built_in_error(atom(X)).

% float(X)の場合
goal_compile_float(TEMP,[],X,Right1,Compiled):-
   check_body(TEMP,Right1,Right2),
   list_to_conjunction([float(X)|Right2],OrRight),
   generate_term((;),[ \+(float(X)) |[OrRight|[]]],Compiled).

goal_compile_float(_,_,X,_,_):-
   goal_compile_built_in_error(float(X)).

% integer(X)の場合
goal_compile_integer(TEMP,[],X,Right1,Compiled):-
   check_body(TEMP,Right1,Right2),
   list_to_conjunction([integer(X)|Right2],OrRight),
   generate_term((;),[ \+(integer(X)) |[OrRight|[]]],Compiled).

goal_compile_integer(_,_,X,_,_):-
   goal_compile_built_in_error(integer(X)).

% number(X)の場合
goal_compile_number(TEMP,[],X,Right1,Compiled):-
   check_body(TEMP,Right1,Right2),
   list_to_conjunction([number(X)|Right2],OrRight),
   generate_term((;),[ \+(number(X)) |[OrRight|[]]],Compiled).

goal_compile_number(_,_,X,_,_):-
   goal_compile_built_in_error(number(X)).

% atomic(X)の場合
goal_compile_atomic(TEMP,[],X,Right1,Compiled):-
   check_body(TEMP,Right1,Right2),
   list_to_conjunction([atomic(X)|Right2],OrRight),
   generate_term((;),[ \+(atomic(X)) |[OrRight|[]]],Compiled).

goal_compile_atomic(_,_,_,_,_):-
   goal_compile_built_in_error('atomic').

% simple(X)の場合
goal_compile_simple(TEMP,[],X,Right1,Compiled):-
   check_body(TEMP,Right1,Right2),
   list_to_conjunction([simple(X)|Right2],OrRight),
   generate_term((;),[ compound(X) |[OrRight|[]]],Compiled).

goal_compile_simple(_,_,X,_,_):-
   goal_compile_built_in_error(simple(X)).

% compound(X)の場合
goal_compile_compound(TEMP,[],X,Right1,Compiled):-
   check_body(TEMP,Right1,Right2),
   list_to_conjunction([compound(X)|Right2],OrRight),
   generate_term((;),[ simple(X) |[OrRight|[]]],Compiled).

goal_compile_compound(_,_,X,_,_):-
   goal_compile_built_in_error(compound(X)).

% functor(Term,Name,Arity)の場合
goal_compile_functor(TEMP,[],Term,Name,Arity,Right1,Compiled):-
   check_body(TEMP,Right1,Right2),
   list_to_conjunction([functor(Term,Name,Arity)|Right2],OrLeft),
   generate_term((;),[OrLeft|[(functor(Term1,Name1,Arity1),
      \+(Term=Term1),\+(Name=Name1),\+(Arity=Arity1))|[]]],Compiled).

goal_compile_functor(TEMP,All,Term,Name,Arity,Right1,Compiled):-
   same_variable_list(All,[Name,Arity]),!,
   check_body(TEMP,Right1,Right2),
   list_to_conjunction([ functor(Term,Name,Arity) |Right2],Compiled).

goal_compile_functor(TEMP,All,Term,Name,Arity,Right1,Compiled):-
   same_variable_list(All,[Term]),!,
   check_body(TEMP,Right1,Right2),
   list_to_conjunction([ functor(Term,Name,Arity) |Right2],Compiled).

goal_compile_functor(_,_,_,_,_,_,_):-
   goal_compile_built_in_error('functor').

%arg(ArgNo,Name,Arg)の場合
goal_compile_arg(TEMP,[],ArgNo,Name,Arg,Right1,Compiled):-
   check_body(TEMP,Right1,Right2),
   list_to_conjunction([ arg(ArgNo,Name,Arg) |Right2],OrLeft),
   generate_term((;),[OrLeft|[ (arg(ArgNo,Name,Arg1),
                              \+(Arg=Arg1)) |[]]],Compiled).

goal_compile_arg(TEMP,All,ArgNo,Name,Arg,Right1,Compiled):-
   same_variable_list(All,[Arg]),!,
   check_body(TEMP,Right1,Right2),
   list_to_conjunction([ arg(ArgNo,Name,Arg) |Right2],Compiled).

goal_compile_arg(_,_,_,_,_,_,_):-
   goal_compile_built_in_error('arg'). 

% Term =.. Listの場合
 
% 以下意味不明のため comment out した
% goal_compile_eq_dot_dot(TEMP,[],Term,List,Right1,Compiled):-
%    check_body(TEMP,Right1,Right2),
%    list_to_conjunction([ (Term =.. List) |Right2],OrLeft),
%    generate_term((;),
%      [ OrLeft,
%        ( (Term1 =.. List1),\+(Term=Term1),\+(List=List1) )
%      ],
%      Compiled).

goal_compile_eq_dot_dot(TEMP,All,Term,List,Right1,Compiled):-
   get_variable(Term,TermVs),
   get_variable(List,ListVs),
   ( same_variable_list(All,TermVs)
   ; same_variable_list(All,ListVs)
   ),!,
   check_body(TEMP,Right1,Right2),
   list_to_conjunction([(Term =.. List) |Right2],Compiled).

goal_compile_eq_dot_dot(_,_,T,L,_,_):-
   goal_compile_built_in_error((T=..L)).
 
%name(Const,CharList)の場合
goal_compile_name(TEMP,[],Const,CharList,Right1,Compiled):-
   check_body(TEMP,Right1,Right2),
   list_to_conjunction([name(Const,CharList)|Right2],OrLeft),
   generate_term((;),[OrLeft|[(name(Const1,CharList1),
        \+(Const=Const1),\+(CharList=CharList1))|[]]],Compiled).

goal_compile_name(TEMP,All,Const,CharList,Right1,Compiled):-
   same_variable_list(All,Const),!,
   check_body(TEMP,Right1,Right2),
   list_to_conjunction([name(Const,CharList)|Right2],Compiled).

goal_compile_name(TEMP,All,Const,CharList,Right1,Compiled):-
   same_variable_list(All,CharList),!,
   check_body(TEMP,Right1,Right2),
   list_to_conjunction([name(Const,CharList)|Right2],Compiled).

goal_compile_name(_,_,_,_,_,_):-
   goal_compile_built_in_error(name).

%atom_chars(Const,CharList)の場合
goal_compile_atom_chars(TEMP,[],Const,CharList,Right1,Compiled):-
   check_body(TEMP,Right1,Right2),
   list_to_conjunction([atom_chars(Const,CharList)|Right2],OrLeft),
   generate_term((;),[OrLeft|[(atom_chars(Const1,CharList1),
        \+(Const=Const1),\+(CharList=CharList1))|[]]],Compiled).

goal_compile_atom_chars(TEMP,All,Const,CharList,Right1,Compiled):-
   same_variable_list(All,Const),!,
   check_body(TEMP,Right1,Right2),
   list_to_conjunction([atom_chars(Const,CharList)|Right2],Compiled).

goal_compile_atom_chars(TEMP,All,Const,CharList,Right1,Compiled):-
   same_variable_list(All,CharList),!,
   check_body(TEMP,Right1,Right2),
   list_to_conjunction([atom_chars(Const,CharList)|Right2],Compiled).

goal_compile_atom_chars(_,_,_,_,_,_):-
   goal_compile_built_in_error.

%number_chars(Const,CharList)の場合
goal_compile_number_chars(TEMP,[],Const,CharList,Right1,Compiled):-
   check_body(TEMP,Right1,Right2),
   list_to_conjunction([number_chars(Const,CharList)|Right2],OrLeft),
   generate_term((;),[OrLeft|[(number_chars(Const1,CharList1),
           \+(Const=Const1),\+(CharList=CharList1))|[]]],Compiled).

goal_compile_number_chars(TEMP,All,Const,CharList,Right1,Compiled):-
   same_variable_list(All,Const),!,
   check_body(TEMP,Right1,Right2),
   list_to_conjunction([number_chars(Const,CharList)|Right2],Compiled).

goal_compile_number_chars(TEMP,All,Const,CharList,Right1,Compiled):-
   same_variable_list(All,CharList),!,
   check_body(TEMP,Right1,Right2),
   list_to_conjunction([number_chars(Const,CharList)|Right2],Compiled).

goal_compile_number_chars(_,_,_,_,_,_):-
   goal_compile_built_in_error.

% asserta(Clause)の場合
goal_compile_asserta(TEMP,All,Clause,Right1,Compiled):-
   get_variable(Clause,ClauseV),
   variable_join(All,ClauseV,[]),
   variable_diff(All,ClauseV,All_1),
   goal_compile_equal_sub2(All_1,Right1,Right2),
   check_body(TEMP,Right2,Right3),
   list_to_conjunction([(asserta(Clause))|Right3],Compiled).

goal_compile_asserta(_,_,_,_,_):-
   goal_compile_built_in_error('asserta').

% assertz(Clause)の場合
goal_compile_assertz(TEMP,All,Clause,Right1,Compiled):-
   get_variable(Clause,ClauseV),
   variable_join(All,ClauseV,[]),
   variable_diff(All,ClauseV,All_1),
   goal_compile_equal_sub2(All_1,Right1,Right2),
   check_body(TEMP,Right2,Right3),
   list_to_conjunction([(assertz(Clause))|Right3],Compiled).

goal_compile_assertz(_,_,Clause,_,_):-
   goal_compile_built_in_error(assertz(Clause)).

% assert(Clause)の場合
goal_compile_assert(TEMP,All,Clause,Right1,Compiled):-
   get_variable(Clause,ClauseV),
   variable_join(All,ClauseV,[]),
   variable_diff(All,ClauseV,All_1),
   goal_compile_equal_sub2(All_1,Right1,Right2),
   check_body(TEMP,Right2,Right3),
   list_to_conjunction([(assert(Clause))|Right3],Compiled).

goal_compile_assert(_,_,Clause,_,_):-
   goal_compile_built_in_error(assert(Clause)).

goal_compile_retract(TEMP,ALL,CL,Right1,Compiled):-
   get_variable(CL,CL_VarList),    % retract(CL)
   variable_diff(ALL,CL_VarList,ALL_1),
   goal_compile_equal_sub2(ALL_1,Right1,Right2),
   check_body(TEMP,Right2,Right3),
   list_to_conjunction([Retract=failure|Right3],OrRight),
   generate_term((;),[Retract=success,OrRight],AndRight),
   list_to_conjunction(
      [ (retract(CL),Retract=success ; Retract=failure), '!' | AndRight],
     Compiled).

goal_compile_retract(_,_,CL,_,_):-
   goal_compile_built_in_error(retract(CL)).

goal_compile_findall(TEMP,All,FindAll,Right1,Compiled):-
% forall([X,Y,Z], findall(f(X),q(X,Y),L) -> ... ) は
% Y がallで縛られているのでfail させる
% forall([X,Z,L], findall(f(X),q(X,Y),L) -> ... ) はOK
   FindAll =.. [Pred, Term,Formula,L],
   ( Pred = findall ;  Pred = setof ;  Pred = bagof ),
   var(L),
   get_variable(Term,TermV),
   variable_diff(All,[L|TermV],All_1),
   get_variable(Formula,FormulaV),
   variable_join(All_1,FormulaV,[]),
   goal_compile_equal_sub2(All_1,Right1,Right2),
   check_body(TEMP,Right2,Right3),
   list_to_conjunction([FindAll|Right3], Compiled),!.

goal_compile_findall(_,_,FindAll,_,_,_):-
   goal_compile_built_in_error(FindAll).

%----- Miscellaneous --------
% 左辺がA = Bの場合

%%%%<<<<<<<<<<< 修正 T.Sato Oct. 2003 <<<<<<<<<<<<<
% goal_compile_equal(TEMP,All,A,B,Right,Compiled):-

goal_compile_equal(_,[],A,B,[fail],Compiled):-
   ( A==B, Compiled = true
   ; Compiled = (\+(A=B)) ),!.
 
% 場合1 "A=B"の両辺が定数の場合
goal_compile_equal(TEMP,[],A,B,Right1,Compiled):-
    constant(A),
    constant(B),!,
    ( A=B,
        check_body(TEMP,Right1,Right2),
        list_to_conjunction(Right2,Compiled)
    ; \+(A=B),
        Compiled = true
    ).

% 場合2 "A=B"の一方が変数の場合
goal_compile_equal(TEMP,All,A,B,Right1,Compiled):- 
    nonvar(A),var(B),!,
    goal_compile_equal(TEMP,All,B,A,Right1,Compiled).

% 場合2-1 Var=term の形かつ VarがAllに含まれない場合
goal_compile_equal(TEMP,All,A,B,Right1,Compiled):-
    var(A),
    \+(variable_member(A,All)),!, % occur check
    get_variable(B,BVarList),
    ( A==B,!,
        check_body(TEMP,Right1,Right2),
        list_to_conjunction([Right2],Compiled)
    ; variable_member(A,BVarList),!,  % occur check 
        Compiled = true
    ; variable_diff(All,BVarList,All_1),
        goal_compile_equal_sub2(All_1,Right1,Right2),
        check_body(TEMP,Right2,Right3),
        ( Right3=[fail], Compiled = (\+(A=B))
        ; \+(Right3=[fail]),
           list_to_conjunction([(A=B)|Right3],OrRight),
           generate_term((;),[(\+(A=B)),OrRight],Compiled)
        )
    ),!.

% 場合2-2 Var=term の形かつ VarがAllに含まれる場合    
goal_compile_equal(TEMP,All1,A,B,Right1,Compiled):-
    var(A),
    variable_member(A,All1),!,
    get_variable(B,BVarList),
    ( A==B,!,
        goal_compile_equal_sub2(All1,Right1,Right2),
        check_body(TEMP,Right2,Right3),
        list_to_conjunction([Right3],Compiled)
    ; variable_member(A,BVarList),!,
        Compiled = true
    ; variable_delete(A,All1,All2),
         % substitute B for A in Right1 = Right2
        variable_to_term(A,B,Right1,Right2),
        goal_compile_equal_sub2(All2,Right2,Right3),
        check_body(TEMP,Right3,Right4),
        list_to_conjunction(Right4,Compiled)
    ),!.
    
% 場合3 "="の片方がcompound、もう片方がconstant の場合
goal_compile_equal(_,_,A,B,_,Compiled):-
   ( compound(A), constant(B)
   ; compound(B), constant(A) ),!, % A=B fails
   Compiled = true.

% 場合4 "="の両辺がcompoundの場合    
goal_compile_equal(_,_,A,B,_,Compiled):-
    compound(A),
    compound(B),
    functor(A,AFunc,AArity),
    functor(B,BFunc,BArity),
    ( \+(AFunc==BFunc)
    ; \+(AArity==BArity)),!,
    Compiled = true.

goal_compile_equal(TEMP,All,A,B,Right,Compiled):-
    compound(A),
    compound(B),
    functor(A,AFunc,AArity),
    functor(B,AFunc,AArity),!,
    A =.. [AFunc|AList],
    B =.. [AFunc|BList],
    goal_compile_equal_sub(AList,BList,CList),
    list_to_conjunction(CList,C),
    generate_left([C],NewLEFT),
    generate_right(Right,RIGHT),
    generate_all(All,NewLEFT,RIGHT,ALL),
    goal_compile(TEMP,ALL,Compiled).

%場合4 コンパイルが失敗する場合
goal_compile_equal(_,All,A,B,Right,_):-
    F = all(All,((A=B) ->Right)),
    goal_compile_built_in_error(F).

goal_compile_equal_sub([],[],[]).    
goal_compile_equal_sub([A|AList],[B|BList],[(A=B)|CList]):-
    goal_compile_equal_sub(AList,BList,CList).

goal_compile_equal_sub2([],BodyList,BodyList):-!.
goal_compile_equal_sub2(_,[],[]).
goal_compile_equal_sub2(VarList,[Body|BodyList1],[NewBody|BodyList2]):-
    functor(Body,all,_),!,
    Body = all(All1,IMPLY),              % all(VarList, all(All1,IMPLY))
    get_free_variable(Body,BodyVar),     %   = all(All3, IMPLY)
    variable_join(VarList,BodyVar,All2), % All2 = VarList/\BodyVar
    variable_union(All1,All2,All3),
    NewBody = all(All3,IMPLY),!,
    goal_compile_equal_sub2(VarList,BodyList1,BodyList2).

goal_compile_equal_sub2(VarList,[Body|BodyList1],[NewBody|BodyList2]):-
    get_free_variable(Body,BodyVar),
    variable_join(VarList,BodyVar,All2), % All2 = VarList/\free_BodyVar
    ( All2 == [],
        NewBody=Body
    ; NewBody=all(All2,Body) ),!,
    goal_compile_equal_sub2(VarList,BodyList1,BodyList2).

%%%%%>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

%%%%<<<<<<<<<<< 修正 T.Sato Oct. 2003 <<<<<<<<<<<<<
% length(List,Length)の場合
goal_compile_length(TEMP,All,List,Length,Right1,Compiled):-
   get_variable(List,V1), variable_join(V1,All,QV1),
   get_variable(Length,V2), variable_join(V2,All,QV2),
   ( QV1==[],QV2==[],
       goal_compile_equal_sub2(All,Right1,Right2),
       check_body(TEMP,Right2,Right3),
       list_to_conjunction([length(List,Length)|Right3],OrRight),
        generate_term((;), [\+length(List,Length),OrRight],Compiled)
  ; QV1==[],same_variable_list(QV2,[Length]),
       variable_delete(Length,All,All2),
       goal_compile_equal_sub2(All2,Right1,Right2),
       check_body(TEMP,Right2,Right3),
       list_to_conjunction([length(List,Length)|Right3],Compiled)
  ; QV1\==[],QV2==[],var(List),
        % all([List,..],(length(List,Length) -> .. no Length occurs ..))
       get_variable(Right1,Right1Vars),
       \+variable_member(List,Right1Vars),
       variable_delete(List,All,All2),
       goal_compile_equal_sub2(All2,Right1,Right2),
       check_body(TEMP,Right2,Right3),
       list_to_conjunction(Right3,Compiled)
  ),!.

goal_compile_length(_,All,List,Length,Right,_):-
   F = all(All,(list(List,Length) ->Right)),
   goal_compile_built_in_error(F).

% sort(In,Out)の場合
goal_compile_sort(TEMP,All,In,Out,Right1,Compiled):-
   get_variable(In,V1), variable_join(V1,All,QV1),
   get_variable(Out,V2), variable_join(V2,All,QV2),
   ( QV1==[],QV2==[],
       goal_compile_equal_sub2(All,Right1,Right2),
       check_body(TEMP,Right2,Right3),
       list_to_conjunction([sort(In,Out)|Right3],OrRight),
        generate_term((;), [\+sort(In,Out),OrRight],Compiled)
  ; QV1==[],same_variable_list(QV2,[Out]),
       variable_delete(Out,All,All2),
       goal_compile_equal_sub2(All2,Right1,Right2),
       check_body(TEMP,Right2,Right3),
       list_to_conjunction([sort(In,Out)|Right3],Compiled)
  ; QV1\==[],QV2==[],var(In),
        % all([In,..],(sort(In,Out) -> .. no Out occurs ..))
       get_variable(Right1,Right1Vars),
       \+variable_member(List,Right1Vars),
       variable_delete(List,All,All2),
       goal_compile_equal_sub2(All2,Right1,Right2),
       check_body(TEMP,Right2,Right3),
       list_to_conjunction(Right3,Compiled)
  ),!.

goal_compile_sort(_,All,In,Out,Right,_):-
   F = all(All,(sort(In,Out) ->Right)),
   goal_compile_built_in_error(F).
%%%%%>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

%%% >>>>>>>> 追加 by T.Sato, Sept. 2003 >>>>>>>>>>>>>>
% ユーザ定義組み込み述語のコンパイル

goal_compile_user_built_in(TEMP,All,Left,ModeDecl,Right1,Compiled):-
    get_x('-',Left,ModeDecl,OutputL),
    get_variable(OutputL,OutputV),
    variable_diff(All,OutputV,All_1), % All quantifies All_1 in output vars
    goal_compile_equal_sub2(All_1,Right1,Right2),
    check_body(TEMP,Right2,Right3),
    ( Right3 = [fail],
        Compiled = (\+Left)
%%% >>>>>>>>>>>>>>>>> 29/05/2005 修正 by T.Sato >>>>>>>>>>
%%% >>>>>>>>>>>>>>>>> 21/08/2005 comment 追加 by T.Sato >>>>>>>>>>
%%%     ; \+(Right3=[fail]),!,
%%%           list_to_conjunction(Right3,F),
%%%           Compiled = (Left -> F ; true) % Left succeeds at most once
%%%
%%%  This  seems to cause confusion in FOC of
%%%  all( ..-> ..) and ( -> ; true). Maybe ( -> ; true) may be optimized
%%%  to true in the later process and makes the output wrong.
%%% <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    ; \+(Right3=[fail]),
         % Left = q(Args) の時
         %   user_q(Ans,Args) :- (Left,Ans=true ; Ans=false),!.
         % をassertし（後でファイルに書き出す）
         %   Compiled = (user_q(Ans,Args),(Ans==false ; Ans==true,Right3))
         % を返す

         list_to_conjunction([(Ans==true)|Right3],F),
         Left=..[Pred|Args],name(Pred,S1),
         fo_append("user_",S1,S2),name(UserName,S2),
         UserH=..[UserName,Ans|Args],
         UserCL = (UserH :- ((Left,Ans=true;Ans=false),!)),
         length([Ans|Args],Arity),
         functor(MGA,UserName,Arity), % MGA = most general atom
         ( clause(user_pred((MGA :- _)),true),
             is_variant(MGA,UserH)
         ; assertz(user_pred(UserCL))
         ),
         Compiled = (UserH,(Ans==false ; F))
     ),!.

goal_compile_user_built_in(_,All,Left,_,Right,_):-
   F = all(All,(Left ->Right)),
   goal_compile_built_in_error(F).


%%% >>>>>>>> 追加 by T.Sato, Mar. 2004 >>>>>>>>>>>>>>
% ユーザ定義確率組み込み述語のコンパイル

goal_compile_user_p_built_in(TEMP,All,Left,ModeDecl,Right1,Compiled):-
    get_x('-',Left,ModeDecl,OutputL),  % Left はユーザアトム
    get_variable(OutputL,OutputV),
    variable_diff(All,OutputV,All_1), % All quantifies All_1 in output vars
    goal_compile_equal_sub2(All_1,Right1,Right2),
    check_body(TEMP,Right2,Right3),

    % Left = q(Args) の時 (all(All,q(Args)->fail) ;  q(Args)&Right3)
    %   Compiled =  (all(All,not(q(Args))) ; q(Args)&Right3)
    % を返す（all(All,not(q(Args)))はコンパイルされている）。
    list_to_conjunction([Left|Right3],OrRight),
    goal_compile(TEMP,all(All,imply(left(Left),right(fail))), NotLeft),
    generate_term((;),[NotLeft,OrRight],Compiled),!.

goal_compile_user_p_built_in(_,All,Left,_,Right,_):-
   F = all(All,(Left ->Right)),
   goal_compile_built_in_error(F).


%%%%%%% For PRISM %%%%%%%%%%%%%%%%%%%%%%%%%%
% msw(I,V)の仮定：
%  (1) 必ず名前Iを持つ msw(I,xxx) がassert されている
% all([V],(msw(+I,V)-> Right1)
% all([],(msw(+I,+V)-> Right1)
%  sampling 実行するには msw(+I,W) を振り、値 W を決める。
%  この時このゴールは all([V],(W=V -> Right1[V])) と同じ
%     => (msw(I,W)& (W=\=V \/ (W=V & Right1)))

goal_compile_msw(TEMP,ALL,msw(I,V),Right1,Compiled):-
   get_variable(I,I_VarList),
   variable_join(I_VarList,ALL,[]), % no var. in I quantified by ALL
   get_variable(V,V_VarList),       % V may be a term
   variable_join(V_VarList,ALL,ZZ), % ZZ = var. in V quantified by ALL

%%%% >>>>>>>>>>> Modified on Nov. 2004 by T.Sato >>>>>>>>>>>
   ( ZZ == [],
       check_body(TEMP,Right1,Right3)
   ; ZZ \== [],
       variable_diff(ALL,V_VarList,ALL_1),
       goal_compile_equal_sub2(ALL_1,Right1,Right2),
       check_body(TEMP,Right2,Right3)  ),!,
   ( Right3 = [fail], AndRight = [ (\+(W_New=V)) ]
   ; list_to_conjunction([(W_New=V)|Right3],OrRight),
        generate_term((;),[(\+ W_New=V),OrRight],AndRight) ),!,
   list_to_conjunction([msw(I,W_New)|AndRight],Compiled).

%%%% <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

%%%%%%%%%%% Old code %%%%%%%%%%
% msw(I,V)の仮定：
%  (2) I がgivenの場合、必ず msw(I,V)が成功する
%
%   ( ZZ == [],
%       check_body(TEMP,Right1,Right3),
%       ( Right3 = [fail], AndRight = [ (\+(W_New=V)) ]
%       ; list_to_conjunction([(W_New=V)|Right3],OrRight),
%           generate_term((;),[(\+ W_New=V),OrRight],AndRight)
%       ),!,
%       list_to_conjunction([msw(I,W_New)|AndRight],Compiled)
%
%   ; ZZ \== [],
%       variable_diff(ALL,V_VarList,ALL_1),
%       goal_compile_equal_sub2(ALL_1,Right1,Right2),
%       check_body(TEMP,Right2,Right3),
%       ( Right3 = [fail],!, Compiled = fail
%       ; list_to_conjunction([msw(I,V)|Right3],Compiled)
%       )
%   ),!.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% 非標準：msw(I,V)のIというスイッチがないかもしれない
%   list_to_conjunction([values(I,_),msw(I,W_New)|AndRight],Right4),
%   generate_term((;),[(\+values(I,_)),Right4],Compiled).

goal_compile_msw(_,_,Left,_,_):-
   goal_compile_built_in_error(Left).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% >>>>>>>> 追加 by T.Sato, Sept. 2003 >>>>>>
% Left = \+G の場合
goal_compile_naf(_,_,Left,_,Compiled):-
   Left = (\+ G),
   G =(A = B),
   A == B,!,
   Compiled = true.

goal_compile_naf(TEMP,All,Left,Right1,Compiled):-
   Left = (\+ G),    % G must not be quantified by
   get_variable(G,G_VarList),  % any vars in  All
   variable_join(G_VarList,All,[]),
   goal_compile_equal_sub2(All,Right1,Right2),
   check_body(TEMP,Right2,Right3),
   list_to_conjunction([(\+ G)|Right3],OrRight),
   generate_term((;),[G,OrRight],AndRight),
   Compiled = AndRight.

goal_compile_naf(_,All,Left,Right,_):-
   F = all(All,(Left ->Right)),
   goal_compile_built_in_error(F).
%%% <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


%%%------ 以下不要（に見える）-------

%--------- use_built_in(no) ----------
% 入力が all(All,IMPLY)の場合
% <ケース 1>
% 左辺が空の場合
goal_compile2(TEMP,[],left,right(Right),Compiled):-
   goal_compile(TEMP,Right,Compiled).

goal_compile2(_,_,left,_,_):-
   format("Failed at < Case 1 >.",[]),
   retract(goal_compile_fail(_)),
   assertz(goal_compile_fail(1)),
   fail.

% <ケース 2> 左辺が一つのアトムの場合
% <ケース 2.1> 左辺が常にfalseの場合
goal_compile2(_,_,left(fail),_,true).

% <ケース 2.2>
% カットの場合
% 左辺がcutの場合
% cutをそのままにしておく
goal_compile2(TEMP,All,left(Left),RIGHT1,Compiled):-
   Left = '!',
   compound(RIGHT1),
   RIGHT1 =.. [right|Right1],!,
   goal_compile_cut(TEMP,All,Right1,Compiled).

% 左辺が true の場合
% true を取り除く
goal_compile2(TEMP,All,left(Left),RIGHT1,Compiled):-
   Left = true ,
   compound(RIGHT1),
   RIGHT1 =.. [right|Right1],!,
   goal_compile_true(TEMP,All,Right1,Compiled).

% A=B の場合
goal_compile2(TEMP,All,left(Left),RIGHT1,Compiled):-
   Left = (A = B),
   compound(RIGHT1),
   RIGHT1 =.. [right|Right1],!,
   goal_compile_equal(TEMP,All,A,B,Right1,Compiled).

% A;B の場合
goal_compile2(TEMP,All,left(Left),RIGHT1,Compiled):-
   compound(Left),
   Left = (A ; B),!,
   compound(RIGHT1),
   conjunction_to_list(A,AList),
   generate_left(AList,ALEFT),

   get_free_variable(AList,AListVar),
   get_free_variable(B,BVar),

   variable_union(AListVar,BVar,ABVar),
   variable_join(All,ABVar,AllAndABVar),

   same_variable_list(All,AllAndABVar),!,
   variable_join(All,AListVar,AAll),

   generate_all(AAll,ALEFT,RIGHT1,AALL),!,
   goal_compile(TEMP,AALL,ACompiled),

   generate_left([B],BLEFT),
   variable_join(All,BVar,BAll),
   generate_all(BAll,BLEFT,RIGHT1,BALL),!,
   goal_compile(TEMP,BALL,BCompiled),
   list_to_conjunction([ACompiled|[BCompiled|[]]],Compiled).

% A,B の場合
goal_compile2(TEMP,All,left(Left),RIGHT1,Compiled):-
   compound(Left),
   functor(Left,(','),2),!,
   conjunction_to_list(Left,LeftList),
   generate_left(LeftList,LEFT2),
   goal_compile2(TEMP,All,LEFT2,RIGHT1,Compiled).

% exist(ExistVar,Exist)の場合
goal_compile2(TEMP,All,left(Left),RIGHT1,Compiled):-
   compound(Left),
%   Left =.. [exist|[ExistVar|[Exist|[]]]],!,
   Left =.. [exist, ExistVar, Exist],!,
   fo_append(All,ExistVar,All2),
   conjunction_to_list(Exist,ExistList),
   generate_left(ExistList,LEFT2),
   goal_compile2(TEMP,All2,LEFT2,RIGHT1,Compiled).

% <ケース 2.3>
% その他の場合
goal_compile2(TEMP,All,left(Left),RIGHT,Compiled):-
      % 左辺が一つのアトムで、all(All,IMPLY)ではない事を確認
   functor(Left,Name,_),
    \+(Name=all),!, 
      % モードパターンを作る
   get_mode_pattern(All,Left,RIGHT,Mode),
      % closure clauseとcontinuation clauseを作る
   generate_closure_continuation_clause(TEMP,All,Left,RIGHT,Mode,Compiled).
  
% <ケース 3>
% 左辺が連言の場合
goal_compile2(TEMP,All1,LEFT1,RIGHT1,Compiled):-
   compound(LEFT1),  % 左辺が連言である事を確認
   functor(LEFT1,left,LE1Arity),
   LE1Arity > 1,     % all(V,(B,C -> D))の D が atom である事を確認
   compound(RIGHT1),!,
   generate_all(All1,LEFT1,RIGHT1,ALL1),   % 連言を変形
   and_to_imply(ALL1,ALL2),!,
   goal_compile(TEMP,ALL2,Compiled).

% <ケース 4>
% 左辺が all(All,IMPLY) の場合
goal_compile2(TEMP,[],left(ALL1L),RIGHT1R,Compiled):-
       % 入力がall([],all(V,(A -> B)) -> C) である事を確認
       % A,B,Cは、全て atom
   ALL1L = all(All1l,imply(LEFT1L,RIGHT1L)),
   LEFT1L =.. [left|Left1l],
   RIGHT1L =.. [right|Right1l],
   RIGHT1R =.. [right|Right1r],!,
       % B -> failを作り、実行可能にする
   generate_left(Right1l,LEFT2),
   generate_all([],LEFT2,right(fail),NotLEFT2),!,
   goal_compile(TEMP,NotLEFT2,AndRight),
       % Aを実行可能にする
   check_body(TEMP,Left1l,AndLeftList),
       % Cを実行可能にする
   check_body(TEMP,Right1r,OrRightList),
   goal_compile2_case4_sub(AndLeftList,AndRight,
                        OrRightList,All1l,Compiled).

% 入力がall(All1,all(All2,IMPLY) -> A) で All1が空でない場合
goal_compile2(_,_,left(ALLL),_,_):-
   ALLL = all(_,imply(_,_)),
   format("Failed at < Case 4 >.",[]),
   retract(goal_compile_fail(_)),
   assertz(goal_compile_fail(1)),!,
   fail.

% その他
goal_compile2(_,All,LEFT,RIGHT,_):-
   format("Compilation failed.",[]),
   generate_all(All,LEFT,RIGHT,ALL),
   format("Expression failed to be compiled:",[]),
   write_clause(ALL),
   retract(goal_compile_fail(_)),
   assertz(goal_compile_fail(1)),!,
   fail.

goal_compile2_case4_sub(_,_,[true],_,true).
 
goal_compile2_case4_sub(AndLeftList1,true,[fail],All1l,Compiled):-
   some_variables_to_new_variable(All1l,AndLeftList1,AndLeftList2),
   list_to_conjunction(AndLeftList2,Compiled).

goal_compile2_case4_sub(AndLeftList,AndRight,[fail],All1l,Compiled):-
   fo_append(AndLeftList,[AndRight],AndList1),
   some_variables_to_new_variable(All1l,AndList1,AndList2),
   list_to_conjunction(AndList2,Compiled).

goal_compile2_case4_sub(AndLeftList1,true,OrRightList,All1l,Compiled):-
   some_variables_to_new_variable(All1l,AndLeftList1,AndLeftList2),
   list_to_conjunction(AndLeftList2,OrLeft),
   list_to_conjunction(OrRightList,OrRight),
   generate_term((;),[OrLeft|[OrRight|[]]],Compiled).

goal_compile2_case4_sub(AndLeftList,AndRight,OrRightList,All1l,Compiled):-
   fo_append(AndLeftList,[AndRight],OrLeftList1),
   some_variables_to_new_variable(All1l,OrLeftList1,OrLeftList2),
   list_to_conjunction(OrLeftList2,OrLeft),
   list_to_conjunction(OrRightList,OrRight),
   generate_term((;),[OrLeft|[OrRight|[]]],Compiled).
%%%------ 以上不要（に見える）-------
%---------end of use_built_in(no) ----------

%------ closure clauseとcontinuation clauseを作る ----------

% generate_closure_continuation_clause(TEMP,All,Left,RIGHT,Mode,DefTerm)
%  all(All,imply(left(Left),RIGHT))と、モードパターンから
%  closure clauseとcontinuation clauseを作る。
%  continuation clauseを実行可能にしてTEMPに出力する。
%  all(All,imply(left(Left),RIGHT))をclosure(...)に置き換える

% 一度作ったモードパターンの時
generate_closure_continuation_clause(TEMP,All,Left,RIGHT,Mode,DefTerm):-
     % モードパターンModeがPIの中にある事を確認
     % RichModeはModeに（もしあれば）ソートが付いた登録済みのモードパターン
   search_mode_pattern(Mode,RichMode,ClosureNumber),!,
     % continuation clauseを作る
     % ContVarは、cont(y,g(w))のg(w)
   generate_cont_clause(All,Left,RIGHT,RichMode,Cont,ContVar),!,
     % continuation clauseを実行可能にする
   transform_cont(TEMP,Cont),          
     % closure(x,g(w))を作る
   generate_definite_clause_body(ContVar,RichMode,Left,ClosureNumber,DefTerm).


% 新しいモードパターンの時
generate_closure_continuation_clause(TEMP,All,Left,RIGHT,Mode,DefTerm):-
     % モードパターンをPIに入れる
   store_mode_pattern(Mode,ClosureNumber),
     % closure clauseを作る
   generate_closure_clause(Mode,ClosureNumber,Closure),
     % continuation clauseを作る
   generate_cont_clause(All,Left,RIGHT,Mode,Cont,ContVar),
     % closure clause をキューに入れる
   assertz((closure(Closure,Mode,ClosureNumber))),!,
     % continuation clauseを実行可能にする
   transform_cont(TEMP,Cont),          
     % closure(x,g(w))を作る
   generate_definite_clause_body(ContVar,Mode,Left,ClosureNumber,DefTerm).

% get_mode_pattern(All,Left,RIGHT,Mode)
%  モードパターンを初めて作る
get_mode_pattern([],Left,_,Left):-
   \+compound(Left),
   ground(Left),!.

get_mode_pattern(All,Left,RIGHT,Mode):-
       % LeftとRIGHTから必要な情報を取り出す
   compound(Left),
   Left =.. [Name|LeftArgList],
   compound(RIGHT),
   RIGHT =.. [right|_],!,
   get_free_variable(RIGHT,RIVar),
       % ユーザ定義のソート情報を得る
   ( clause(fo_sort(Pat),true),
       Pat=..[Name|SortL]
   ; SortL = [] ),!,
       % モードパターンのリストを作る
   get_mode_pattern_dl(SortL,LeftArgList,All,ModeList),

%%% >>>>>>>> 修正 by T.Sato, Sept. 2003 >>>>>>>>>>>>>>
   modify_mode_pattern(LeftArgList,RIVar,ModeList,ModeList2),
%%% <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

     % モードパターンを完成
   Mode =.. [Name|ModeList2].

get_mode_pattern_dl(_,[],_,[]).
get_mode_pattern_dl(SortL,[Arg|Rest],All,[Mode|Z]):-
   ( SortL=[SortName|SortL2],!,
        create_mode_pattern(SortName,Arg,All,Mode)
   ; SortL=[],
        create_mode_pattern(_,Arg,All,Mode),
        SortL2=SortL
   ),!,
   get_mode_pattern_dl(SortL2,Rest,All,Z).

% create_mode_pattern(Sort,LeftArg,All,RIVar,ModeList1-ModeList2)
%  引数(LeftArg)とAllと右辺に含まれる変数(RIVar)から
%  モードパターンを調べ、リストの先頭につける

%%% >>>>>>>> 修正 by T.Sato, Oct. 2003 >>>>>>>>>>>>>>>>>>>>>>>>>
create_mode_pattern(SortName,Arg,All,Mode):-
   get_variable(Arg,ArgVars),
   variable_join(All,ArgVars,X),
   ( X == [],               % All で全く限量化されていない項は入力にする
      ( var(SortName),      % ソート宣言によるソート情報がなかった
          nonvar(Arg),      % Argとfo_sort(_,_)がunifyしないようにする
          Arg =.. [fo_sort,SortName2,_],
          Mode = ['+',SortName2]
      ; nonvar(SortName),Mode=['+',SortName]
      ; Mode = '+' )
   ; X \==[], Mode = '-' ),!. % それ以外はとりあえず出力に（後で修正）する


% 引数が変数であり、Allに含まれ、左辺に単独で現れ、且つ右辺に含まれない場合、
% 対応するモードパターンを'*'に修正する
% Original codes では q(X,X,Y) の X に対し '*' を与えてしまい、
% 論理的なバグである
modify_mode_pattern(LeftArgL,RIVar,ModeL,ModeL2):-
   modify_mode_pattern2(LeftArgL,LeftArgL,RIVar,1,ModeL,ModeL2).

modify_mode_pattern2([LeftArg|X],LeftArgL,RIVar,M,[Mode|Y],ModeL2):-
   ( Mode == '-',         % M = LeftArg s position in the LeftArgL
        var(LeftArg),
        \+var_occur_else(LeftArg,LeftArgL,1,M),
        \+variable_member(LeftArg,RIVar),
        ModeL2 =['*'|Z]
   ;  ModeL2 =[Mode|Z] ),!,
   M1 is M+1,
   modify_mode_pattern2(X,LeftArgL,RIVar,M1,Y,Z).
modify_mode_pattern2([],_,_,_,[],[]).

var_occur_else(Var,[Arg|X],N,M):-
   ( N \== M,
       get_variable(Arg,ArgVars),
       variable_member(Var,ArgVars)
   ; N1 is N+1,
       var_occur_else(Var,X,N1,M) ),!.
%%% <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

% store_mode_pattern(Mode,ClosureNumber)
%  search_mode_pattern(Mode,_,_) が失敗しているのでModeは
%  未登録. モードパターンをpiにしまう(assertする)
store_mode_pattern(Mode,ClosureNumber):-
     % 同じ名前、同じアリティのモードパターンの数を数える
     % Mode = q(+,[-,list])をそのまま、assertする
   Mode=Mode1,
   generate_general_mode_pattern(Mode1,GeneralMode),
   count_mode_pattern(GeneralMode,ClosureNumber), 
   assertz((pi(Mode1,ClosureNumber))).


% count_mode_pattern(GeneralMode,ClosureNumber)
%  同じ名前、同じアリティのモードパターンの数を数える。
count_mode_pattern(GeneralMode,ClosureNumber):-
   findall(GeneralMode,clause(pi(GeneralMode,_),true),X),
   length(X,ClosureNumber).

% generate_general_mode_pattern(Mode,GMode) 
%  与えられたモードパターンから、引数を全て変数にした
%  モードパターンを生成する。 
generate_general_mode_pattern(Mode,GMode):-
   Mode =.. [ModeHead|ModeBody],
   list_to_new_variable_list(ModeBody,VarModeBody),
   GMode =.. [ModeHead|VarModeBody].

% search_mode_pattern(Mode,RichMode,ClosureNumber)
%  与えられたモードパターンModeの情報を探してくる。
%  もし、Modeにソート情報がついたRichModeがあれば、
%  それを返す.
%  モードパターンが登録されてなかったら、失敗する。
% Mode = add(-,*,+) => RichMode = add(-,*,[+,num])
search_mode_pattern(M1,RichMode,ClosureNumber):-
   clause(pi(M2,ClosureNumber),true),
   mode_unify(M1,M2,M3),!,  % may fail
   ( M2 == M3,
       RichMode = M2
   ; M2\==M3,
       RichMode = M3,
       retract(pi(M2,ClosureNumber)),
       assertz(pi(M3,ClosureNumber))
   ).

mode_unify(M1,M2,M3):-
   M1=..[Name|Margs1],
   M2=..[Name|Margs2],
   mode_unify2(Margs1,Margs2,Margs3),!,
   M3=..[Name|Margs3].
mode_unify2([A|X],[B|Y],[C|Z]):-
   ( A==B, C=A
   ; A=[Sign,SortName],B=Sign, C=A
   ; B=[Sign,SortName],A=Sign, C=B ),!,
   mode_unify2(X,Y,Z).
mode_unify2([],[],[]).


% generate_closure_clause(Mode,ClosureNumber,Closure)
%  モードパターンからclosure clauseを作る
%  ClosureNumberは、closure clauseのHeadの名前に使う
generate_closure_clause(Mode,ClosureNumber,Closure):-
     % モードパターンから必要な情報を取り出す
   Mode =.. [ModeName|ModeArgList],!,
     % モードパターンから新変数のリストを作る
     % IOListは、'+'と'-'と'*'に対応したリスト
     % YZListは、'-'と'*'に対応したリスト(Allに使う)
     % XListは、'+'に対応したリスト(最後にCが付いてる)
     % YListは、'-'に対応したリスト(最後にCが付いてる)
   generate_closure_arg(ModeArgList,IOList,YZList,XList,YList),
     % closure clauseのHeadの名前を作る
   append_name('closure_',ModeName,ClosureNumber,ClosureName),
     % p(x,,y,,z)を作る
   generate_term(ModeName,IOList,IOTerm),
     % cont(y,C)を作る
   generate_term('cont',YList,ContTerm),
     % 左辺を作る
   generate_left([IOTerm],LEFT),
     % 右辺を作る
   generate_right([ContTerm],RIGHT),
     % all(All,(p(x,,y,,z) -> cont(y,C)))を作る
   generate_all(YZList,LEFT,RIGHT,ALL),
     % closure_pn(x,C)を作る
   generate_term(ClosureName,XList,ClosureHead),
     % closure clauseを作る
   generate_clause(ClosureHead,[ALL],Closure).

% generate_closure_arg(ModeArgList,IOList,YZList,XList,YList)
%  モードパターンから新変数のリストを作る
%  IOListは、'+'と'-'と'*'に対応したリスト
%  YZListは、'-'と'*'に対応したリスト(Allに使う)
%  XListは、'+'に対応したリスト(最後にCが付いてる)
%  YListは、'-'に対応したリスト(最後にCが付いてる)
generate_closure_arg(ModeArgList,IOList,YZList,XList,YList):-
   generate_closure_arg_dl(ModeArgList,IOList-[],
                 YZList-[],XList-[],YList-[]).
     % 全てのモードパターンを調べたら終了
   generate_closure_arg_dl([],IOList-IOList,
                 YZList-YZList,[C|XList]-XList,[C|YList]-YList).

% モードパターンが'+'の時、
% xに関係するリストの先頭に新変数を付ける
% ここで fo_sort(list,V) などが continuation 節のヘッドに引数として生成される.
generate_closure_arg_dl([ ['+',Fo_SortName] | ModeArgList ],IOList1-IOList3,
             YZList1-YZList2,XList1-XList3,YList1-YList2):-
   IOList1 = [NewVar|IOList2],
   generate_term(fo_sort,[Fo_SortName, NewVar],Fo_SortTerm),
   XList1 = [Fo_SortTerm|XList2],!,
   generate_closure_arg_dl(ModeArgList,IOList2-IOList3,
                         YZList1-YZList2,XList2-XList3,YList1-YList2).

generate_closure_arg_dl(['+'|ModeArgList],IOList1-IOList3,
             YZList1-YZList2,XList1-XList3,YList1-YList2):-
   IOList1 = [NewVar|IOList2],
   XList1 = [NewVar|XList2],!,
   generate_closure_arg_dl(ModeArgList,IOList2-IOList3,
            YZList1-YZList2,XList2-XList3,YList1-YList2).

% モードパターンが'-'の時、
% yに関係するリストの先頭に新変数を付ける
generate_closure_arg_dl(['-'|ModeArgList],IOList1-IOList3,
            YZList1-YZList3,XList1-XList2,YList1-YList3):-
   IOList1 = [NewVar|IOList2],
   YZList1 = [NewVar|YZList2],
   YList1 = [NewVar|YList2],!,
   generate_closure_arg_dl(ModeArgList,IOList2-IOList3,
            YZList2-YZList3,XList1-XList2,YList2-YList3).

% モードパターンが'*'の時、
% zに関係するリストの先頭に新変数を付ける
generate_closure_arg_dl(['*'|ModeArgList],IOList1-IOList3,
           YZList1-YZList2,XList1-XList2,YList1-YList2):-
   IOList1 = [_|IOList2],!,
   generate_closure_arg_dl(ModeArgList,IOList2-IOList3,
           YZList1-YZList2,XList1-XList2,YList1-YList2).

% generate_cont_clause(All,Left,RIGHT,Mode,Cont,ContVar2)
% continuation clause(=Cont)を作る
% ContVar2は、cont(y,g(w))のg(w)
generate_cont_clause(All,Left,RIGHT,Mode,Cont,ContVar2):-

      % continuation clauseの左辺(y=t)（=ContLEFT）を作る
      % tを持ってくる
    get_x('-',Left,Mode,LeftTList),

      % continuation clauseの左辺(y=t)を作る
      % LeftTListは、continuation clauseの右辺のtをリストにしたもの
      % NewVarListは、新変数のリスト(y)
      % ContLEFTは、continuation clauseの左辺(y=t)
    generate_cont_left(LeftTList,NewVarList,ContLEFT),

      % continuation clauseのBodyを作る
      %  Bodyは、all(ContAll,(y=t -> RIGHT))の形をしている
      %  RIVarはFvar(RIGHT)
    get_free_variable(RIGHT,RIVar),

      % LeftTVarはFvar(t)
    get_free_variable(LeftTList,LeftTVar),

      % Fvar(RIGHT) or Fvar(t)を作る
    variable_union(RIVar,LeftTVar,RIVarOrLeftTVar),

      % all(ContAll,(y=t -> RIGHT))のContAll(=y1)を作る
    variable_join(All,RIVarOrLeftTVar,ContAll),

      % all(ContAll,(y=t -> RIGHT))(=ContALL)を作る
    generate_all(ContAll,ContLEFT,RIGHT,ContALL),

      % continuation clauseのHeadを作る
      % w = ((Fvar(RIGHT) or Fvar(t)) \ y1)
    variable_diff(RIVarOrLeftTVar,ContAll,W),

      % StackNumberを取ってくる
    get_stack_number(StackNumber),

      % fnを作る、例 f0
    append_name(f,StackNumber,ContVarName),

      % fn(w)(=ContVar1)を作る
    generate_term(ContVarName,W,ContVar1),

      % cont(y,fn(w))のy,fn(w)を作る
    fo_append(NewVarList,[ContVar1],ContArgList),

      % cont(y,fn(w))(=ContHead)を作る
    generate_term(cont,ContArgList,ContHead),
% 追加説明：
%   cont(y,fn(w)):-  all(y1,     (y=t -> RIGHT))
%   ContHead     :-  all(ContAll,(ContLEFT -> RIGHT))
%      fn(w) = ContVar1, w = W, y1 = ContAll (=< Fvar(t))
%      ContLEFT = (NewVarList(=y) = LeftTList(=t))
%   = ContALL が出来た。

   ( RIGHT = right(fail),     % cont(fn(w)):- fail となる場合
       ContVar2 = ContVar1,   % continuation clauseを作らない
       Cont = []
   ; % continuation clauseを作る
      ( RIGHT=right(Right),   % 末尾最適化可能な場合
          compound(Right),
          Right =.. [cont|CArgList],		
          % cont(y,fn(w)):- all(y1,(y=t -> cont(w)) の形

%%% >>>>>>>> 修正 by T.Sato, Sept. 2003 >>>>>>>>>>>>>
% TRO (Tail Recursive Optimization)
%   cont(y,f(w)) <- all(y1,(y=t -> cont(t2,w)) で y1=t=t2,  の時、
%   cont(y,f(w)) <- cont(y,w)  が生成される事を防ぐ
%   W = w (に対応する変数リスト、以下同様), ContAll = y1,
%   NewVarList = y = 新変数のリスト, LeftTList = t
          diff_variable_list(LeftTList),
          same_variable_list(ContAll,LeftTList),
          get_last(CArgList,_,CArgList2), % t2 = CArgList2
          diff_variable_list(CArgList2),  
          same_variable_list(LeftTList,CArgList2),
          W = [ContVar2],
%%% <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
%%% Original codes:
%%%   |t2|=0は確認したが |y|=0 を確認していないので
%%%   fail_hmm の Unsuccessful definition (II) に対し誤った
%%%   結果を与える
%%%
%%%          get_last(CArgList,_,CArgList2),
%%%          varlist(CArgList2),
%%%          get_last(W,ContVar2,[]),!, % W=[ContVar2] の形のハズ

%format(" ** TR Opt. **~n",[]),
%format("  Right =~w, ContVar1 = ~w, W =~w~n",[RIGHT,ContVar1,W]),

          Cont = []

      ; ContVar2 = ContVar1,  % 末尾最適化不可能
          generate_clause(ContHead,[ContALL],Cont)
     )
   ).

% generate_cont_left(LeftTList,NewVarList,ContLEFT)
%  continuation clauseの左辺(y=t)を作る
%  LeftTListは、continuation clauseの右辺のtをリストにしたもの
%  NewVarListは、新変数のリスト(y)
%  ContLEFTは、continuation clauseの左辺(y=t)
generate_cont_left(LeftTList,NewVarList,ContLEFT):-
   generate_cont_left_arg(LeftTList,NewVarList,ContLeft),
   generate_left(ContLeft,ContLEFT).
generate_cont_left_arg([],[],[]).

generate_cont_left_arg([LeftT|LeftTList],NewVarList1,ContLeft1):-
   generate_term((=),[NewVar,LeftT],ContLeftArg),
   NewVarList1 = [NewVar|NewVarList2],
   ContLeft1 = [ContLeftArg|ContLeft2],!,
   generate_cont_left_arg(LeftTList,NewVarList2,ContLeft2).

% get_stack_number(StackNumber)
%  StackNumberを持ってくる
get_stack_number(StackNumber):-
     % StackNumberを持ってくる
   retract(stack_number(StackNumber)),
     % 次のStackNumberを記録しとく
   NewStackNumber is StackNumber +1,
   assertz(stack_number(NewStackNumber)).

% get_x(TargetMode,Term,ModePattern,TargetArgList)
%  入力TermからModePatternに従って、
%  TargetModeに対応する項を取出し、リストにする
%  TargetModeが'-'の時は、TargetArgListは、tのリストとなる
%  TargetModeが'+'の時は、TargetArgListは、sのリストとなる
%  TargetModeが'*'の時は、TargetArgListは、uのリストとなる

get_x(_,Term,ModePattern,[]):-
    \+compound(Term),Term==ModePattern,!.

get_x(TargetMode,Term,ModePattern,TargetArgList):-
     % TermとModePatternの名前が一致するかを確認
     % TermとModePatternのアリティが一致するかを確認
   compound(Term),
   compound(ModePattern),
   functor(Term,TermName,TermArity),
   functor(ModePattern,TermName,TermArity),!,
     % TermとModePatternから必要な項のリストを作る
   Term =.. [TermName|TermArgList],
   ModePattern =.. [TermName|ModePatternList],
   get_x_dl(TargetMode,TermArgList,ModePatternList,TargetArgList-[]).

get_x_dl(_,[],[],TargetArgList-TargetArgList).

get_x_dl('+',[TermArg1|TermArgList],[['+'|_]|ModePatternList],
         TargetArgList1-TargetArgList3):-
   ( nonvar(TermArg1),
       TermArg1 = fo_sort(_,TermArg2)
   ; TermArg2 = TermArg1 ),
   TargetArgList1 = [TermArg2|TargetArgList2],!,
   get_x_dl('+',TermArgList,ModePatternList,
                 TargetArgList2-TargetArgList3).

get_x_dl(TargetMode,[TermArg|TermArgList],[TargetMode|ModePatternList],
         TargetArgList1-TargetArgList3):-
   TargetArgList1 = [TermArg|TargetArgList2],!,
   get_x_dl(TargetMode,TermArgList,ModePatternList,
          TargetArgList2-TargetArgList3).

get_x_dl(TargetMode,[_|TermArgList],[_|ModePatternList],
         TargetArgList1-TargetArgList2):-
   get_x_dl(TargetMode,TermArgList,ModePatternList,
          TargetArgList1-TargetArgList2).

% generate_definite_clause_body(ContVar,Mode,Left,ClosureNumber,DefBody)
%  closure(x,g(w))を作る
%  ContVarはg(w)、DefBodyはclosure(x,g(w))

generate_definite_clause_body(ContVar,Mode,Left,ClosureNumber,DefBody):-
     % ClosureNameを作る
     %  例 closure_append0
   functor(Left,LeftName,_),!,
   append_name('closure_',LeftName,ClosureNumber,ClosureName),
     % closure(s,g(w))のsを持ってくる
   get_x('+',Left,Mode,LeftSList),
     % closure(s,g(w))のs,g(w)を作る
   fo_append(LeftSList,[ContVar],ClosureArg),
     % closure(s,g(w))を作る
   generate_term(ClosureName,ClosureArg,DefBody).


%----------continuation clauseの変形------------------

% transform_cont(TEMP,Cont)
%  continuation clauseを実行可能にする

% 末尾最適化によって、Contが呼ばれない場合
% 何もしない。
transform_cont(_,[]).

transform_cont(TEMP,Cont):-
    % all(y1,(y=t -> F))(Right1)のFをゴールコンパイルして、
    % G(Right2)を作る
   get_head_body(Cont,ContHead,ContBodyList),
   ContBodyList = [ContBody|[]],
   ContBody = all(_,imply(LEFT,RIGHT1)),!,
   RIGHT1 =.. [right|Right1],!,
   check_body(TEMP,Right1,Right2),

    % \+(y=t)を作って、出力用ファイルに書く
   transform_cont1(LEFT,ReList1),
    % y=t,Gを作って、出力用ファイルに書く
   transform_cont2(LEFT,Right2,ReList2),
   fo_append(ReList1,[ReList2],ReList),
   step2b_d(TEMP,[ReList],ContHead).

% transform_cont1(LEFT,ReList1)
%  \+(y=t)を作る
%  左辺が空の場合、何もしない
transform_cont1(left,_).

% 左辺が空でない場合
transform_cont1(LEFT1,ReList1):-
   compound(LEFT1),
   LEFT1 =.. [left|Left1],!,
    % y=t(Left1)から\+(y=t)(Left2)を作る
   transform_formula_sub2([],Left1,ReList1).

% y=t,Gを作る
%  左辺が空の場合
transform_cont2(left,Right2,Right2).

% 左辺が空でない場合
transform_cont2(LEFT,Right2,ReList2):-
   compound(LEFT),
   LEFT =.. [left|Left],!,
   fo_append(Left,Right2,ReList2).

%--------unfold/fold---------------------

% transform_formula(TEMP1,Clause,X,Mode,C,ClosureHead,Re1)
%  Clauseからall([w],imply(left(E),right(cont(t,C))))を作り、
%  ゴールコンパイルする
%  結果は、Re1となる
%  Re1は、[[\+(s1=t1)],[s1=t1,\+(s2=t2)],[s1=t1,s2=t2,G]]
%  の形をしている
%  ソート情報があれば、\+(s=t)の簡単化を試みる

transform_formula(_,Clause,X,Mode,C,ClosureHead,_):-
%{}
  clause(fo_trace(yes),true),
  \+(\+((numbervars([Clause,X,C,ClosureHead],0,_),
      format("~nCALL transform_formula/7:~n  Mode = ~w, Head = ~w~n  Unfold CL = ~w~n",
      [Mode,ClosureHead,Clause])))),
  fail.

transform_formula(TEMP,Clause,X,Mode,C,ClosureHead,Result):-
     % Clauseから
     % all(w,(E -> cont(t,C)))を作る
   get_head_body(Clause,Head,BodyList),

     % z(ClauseVar)を取出す
   get_free_variable(Clause,Y),
   variable_unique(Y,ClauseVar),

     % v(HeadXVar)を取出す
   get_x('+',Head,Mode,HeadXList),
   get_variable(HeadXList,HeadXVar),

     % wを作る
     % W = z \ v
   variable_diff(ClauseVar,HeadXVar,W),
   variable_unique(W,Wu),

     % Right = cont(t,C)を作る
   get_x('-',Head,Mode,HeadTList),
   fo_append(HeadTList,[C],ContArgList),
   generate_term(cont,ContArgList,Right),

     % all(w,(E -> cont(t,C)))を作る
     % RIGHT = right(Right)
     % LEFT = left(Bdy_1,..,Bdy_k)
   generate_left(BodyList,LEFT),
   generate_right([Right],RIGHT),
   generate_all(Wu,LEFT,RIGHT,ALL1),!,

     % all(w,(E -> cont(t,C)))をゴールコンパイルして、
     % G(ALL2)を得る
   goal_compile(TEMP,ALL1,ALL2),

     % p'(x,C):- (\+(x=s) or x=s,G1) & (\+(x=s2) or x=s2,G2) ...を作る
     % ソート情報があれば \+x=s[v]の簡単化を試みる
   transform_formula_sub(ClosureHead,X,HeadXList,ALL2,Result1,Result2),
   get_variable(HeadXList,HeadXVar),

     % Resultを[Result1,Result2]とする
   fo_append(Result1,[Result2],Result3),

     % 変数を新しいのにしておく
   some_variables_to_new_variable(HeadXVar,Result3,Result).


% transform_formula_sub(ClosureHead,X,HeadXList,ALL2,Result1,Result3)
%  closure(x,C)のXと、HeadXListと、
%  ALL1をゴールコンパイルした結果(ALL2)から
%  Result1((all([v],\+x=s[v])のリスト)と
%  Result2(exist([v],x=s[v],ALL2[v])のリストを作る
%  X=HeadXList が x=s[v] に対応
% ソート情報があれば \+x=s[v]の簡単化を試みる

transform_formula_sub(ClosureHead,X,HeadXList,ALL2,Result1,Result3):-
   get_fo_sort(ClosureHead,Fo_SortList),
   ( Fo_SortList = [],!,
        transform_formula_sub(X,HeadXList,ALL2,Result1,Result3)
   ; \+var_all_single_occ(HeadXList),!,
           % X=[X1,X2], HeadXList=[Y,Y|Z] の時、誤って
           % all([Y,Z], \+(X1=Y & X2=[Y|Z]))
           %   = (all([Y,Z],\+X1=Y) \/ all([Y,Z],\+X2=[Y|Z])) とする事を防ぐ
        transform_formula_sub(X,HeadXList,ALL2,Result1,Result3)
   ; var_all_single_occ(HeadXList), % ソート情報による簡単化
           % ClosureHeadの中にfo_sortがある場合
           % x=s,G(Result2)と、x=s(Result3a)を作る
        transform_formula_sub1(X,HeadXList,ALL2,Result3,Result3a),
           % x=sの中で、Var=Varの形のものを
           % 左の方に持ってくる
        arrange_x_equal_s(Result3a,Result3b), 
           % x=s(Result3a)から、\+(x=s)(Result1)を作る
           % Result1a は、[[\+(x1=s1)],[x1=s1,\+(x2=s2)],[..],,,]
           % の形をしている
        transform_formula_sub2([],Result3b,Result1a),
           % fo_sortを使う
           % 例 fo_sort(list,A)がある場合、
           % \+A=[B|C]を A=[]に置き換え、\+A=[]をA=[_|_]に置き換える
        app_sort_eq(Fo_SortList,Result1a,Result1)
   ).


%%%>>>>>>>>>>> 追加 T.Sato Oct. 2003 >>>>>>>>>>>>

var_all_single_occ(Term) :-
   get_variable(Term,TermVars),  % get_variable は変数の出現
   diff_variable_list(TermVars). % 回数を保存

% app_sort_eq(SortL,F1,F2)
%  SortL =  [fo_sort(mysort,t1),...] の形のリスト
%  F1 = [[\+(x1=s1)],[x1=s1,\+(x2=s2)],[..],,,] の形
%  ソート情報を利用し、
%  F1の各 EQs = [x1=s1,\+(x2=s2)] in F1 の最後にある
%  \+(x2=s2)を (x2=t1;...;x2=tk) に置き換えた式
%  F2を作る
%  以下 SortPats =[t1,...,tk] は指定されたsort パタン
%  \+(fo_sort(list,V) = [E|F]) 且つ 'E','F'が他に
%  出現してない時 V=[]に変える.

app_sort_eq(SortL,[EQs|X],[NewEQs|Y]):-
   fo_append(A,[NegEQ],EQs),
   ( NegEQ = \+(V = B),
        var(V),
        app_sort_eq2(SortL,V,B,EQ2),
        fo_append(A,[EQ2],NewEQs)
   ; NewEQs=EQs ),!,
   app_sort_eq(SortL,X,Y).
app_sort_eq(_,[],[]).

app_sort_eq2([Fo_Sort|Rest],V,B,EQ2):-!,
   Fo_Sort = fo_sort(SortName,W),
   ( V == W,
       clause(fo_sort(SortName,SortPats),true),
       remove_neg_pat(SortPats,B,RestPats),!,
       mk_or_eq(V,RestPats,EQ2)
   ; app_sort_eq2(Rest,V,B,EQ2) ).
app_sort_eq2([],V,B,EQ2):-
   EQ2 = \+(V = B).

% fails if NegPat is not in SortPats
remove_neg_pat([Pat|X],NegPat,X):-
   is_variant(Pat,NegPat),!.
remove_neg_pat([Pat|X],NegPat,[Pat|Y]):-
   remove_neg_pat(X,NegPat,Y).

is_variant(F1,F2):-
   \+(\+((numbervars(F1,0,_),numbervars(F2,0,_),F1==F2))).

mk_or_eq(V,[Pat],(V=Pat)):-!.
mk_or_eq(V,[Pat|Rest],OR_EQ):-
   mk_or_eq(V,Rest,OR_Right),
   OR_EQ =.. [(;),(V=Pat),OR_Right].
%%%<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


transform_formula_sub(X,HeadXList,ALL2,Result1,Result3):-
     % \+x=s(Result1)と x=s,G(Result3)を作る
   transform_formula_sub1(X,HeadXList,ALL2,Result3,Result3a),
     % x=sの中で、Var=Varの形のものを
     % 左の方に持ってくる
   arrange_x_equal_s(Result3a,Result3b),
     % x=s(Result3a)から、\+(x=s)(Result1)を作る
     % Result1は、[[\+(x1=s1)],[x1=s1,\+(x2=s2)],[..],,,]
     % の形をしている
   transform_formula_sub2([],Result3b,Result1).

% transform_formula_sub1(X,HeadXList,ALL2,Result2,Result2a)
%  x=s,G(Result2)とx=s(Result2a)を作る
transform_formula_sub1([X|Xs],[HeadX|HeadXList],ALL2,
                       [Re2|Result2],[Re2|Result2a]):-
   Re2 =.. [(=),X,HeadX],!,
   transform_formula_sub1(Xs,HeadXList,ALL2,Result2,Result2a).
transform_formula_sub1([],[],ALL2,[ALL2|[]],[]).

% transform_formula_sub2(Result2c,Result2b,Result1a)
%  x=s2 (Result2c) と x=s(Result2b)から、
%  x=s2 & \+(x=s)(Result1a)を作る. Result2cは初めは[].
%  Result1は、[[\+(x1=s1)],[x1=s1,\+(x2=s2)],[..],,,]
%  の形をしている
transform_formula_sub2(Result2c,[Re2|Result2b],[Re1|Result1]):-
   Re3 =.. [(\+), Re2],
   fo_append(Result2c,[Re3],Re1),!,
   transform_formula_sub2([Re2|Result2c],Result2b,Result1).
transform_formula_sub2(_,[],[]).

% arrange_x_equal_s(Result1a,Result1)
%  x=sの中で、Var=Varの形のものを
%  左の方に持ってくる
arrange_x_equal_s(Result1a,Result1):-
   arrange_x_equal_s(Result1a,Var_equal_Var,Var_equal_nonVar),
   fo_append(Var_equal_Var,Var_equal_nonVar,Result1).
arrange_x_equal_s([],[],[]).

% 変数=変数の場合
arrange_x_equal_s([Re1|Result1a],Var_equal_Var1,Var_equal_nonVar):-
   Re1 = (A=B),
   var(A),
   var(B),
   Var_equal_Var1 = [Re1|Var_equal_Var2],!,
   arrange_x_equal_s(Result1a,Var_equal_Var2,Var_equal_nonVar).

% 変数=変数でない場合
arrange_x_equal_s([Re1|Result1a],Var_equal_Var,Var_equal_nonVar1):-
   Var_equal_nonVar1 = [Re1|Var_equal_nonVar2],!,
   arrange_x_equal_s(Result1a,Var_equal_Var,Var_equal_nonVar2).
arrange_x_equal_s([],[],[]).

% arrange_x_equal_s2(Result3a,Result3b)
%  x=sの中で、Var=Varの形のものを
%  右の方に持ってくる
arrange_x_equal_s2(Result1a,Result1):-
   arrange_x_equal_s(Result1a,Var_equal_Var,Var_equal_nonVar),
   fo_append(Var_equal_nonVar,Var_equal_Var,Result1).

%--------------------------------------------------------
% 節の簡単化
%  reduction_of_goal1(Head1L,Body1L,Head2L,Body2L)
%  Body が \+ X=X などの常に失敗するゴールを含む時は
%  fail する. 簡単化できなくなるまで繰り返す
% Body1L はconjunction を表す
reduction_of_goal1(Head1L,Body1L,Head5L,Body5L):-
        % Headに含まれてる変数を保存する
    get_variable(Head1L,Head1Var),!,
        % 節の簡単化を行う
    reduction_of_goal1_sub1(Head1L,Body1L,
                   Head2L,[],Body3L,Head1Var),
        % trueは常に成功するので、取り除く
%%%>>>>>>> 修正 yuizumi 01/May/2008 >>>>>>>
%   delete(true,Body3L,Body2L),
    fo_delete(true,Body3L,Body2L),
%%%<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

%%%>>>>>>>>> 追加 T.Sato Oct. 2003 >>>>>>>>
    ( is_variant([Head1L,Body1L],[Head2L,Body2L]),
        [Head5L,Body5L]=[Head1L,Body1L]
    ; reduction_of_goal1(Head2L,Body2L,Head5L,Body5L)
    ),!.
%%%<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

% reduction_of_goal1_sub1(Head1L,Body1L,
%     Head2L,ReBody2L,Body3L,VarL)
% 節の簡単化を行う

% Body1Lが[]の時
reduction_of_goal1_sub1(Head2L,[],Head2L,ReBody2L,
             Body2L,_):-
 % 簡単化したBodyを反転したものが得られているので、
 % fo_reverseする。
    fo_reverse(ReBody2L,Body2L).

% B1が連言の場合
% [B|BodyL] は (B1 & BodyL)を表す

reduction_of_goal1_sub1(Head1L,[B1|Body1L],
      Head2L,Body2Temp,Body2L,HeadVars):-
    compound(B1),
    functor(B1,(','),2),
    conjunction_to_list(B1,B1L),
    fo_append(B1L,Body1L,Body1L2),!,
    reduction_of_goal1_sub1(Head1L,Body1L2,Head2L,
             Body2Temp,Body2L,HeadVars).

% B1が連言以外
reduction_of_goal1_sub1(Head1L,[B1|Body1L],
               Head2L,Body2Temp,Body2L,LeftVars):-
   reduction_of_goal1_sub2(Head1L,Head1L2,B1,
              Body1L,Body1L2,Body2Temp,BTemp2,LeftVars,VarL2),!,
   reduction_of_goal1_sub1(Head1L2,Body1L2,Head2L,
               BTemp2,Body2L,VarL2).

% B1 = \+(A=B)が常に失敗する場合
reduction_of_goal1_sub2(_,_,B1,_,_,_,_,LeftVars,_):-
    B1 = (\+(A=B)),  % head(..HeadVars..):- .. (\+(A=B)),..
                     % |<-----  LeftVars ----->|
    ( var(B),
       \+(variable_member(B,LeftVars))
    ; A==B 
%%    ; unify(A,B,UnifyList),   %% 何をやっているか不明
%%          % \+(A=B) は (s=t -> false)なので、
%%          % 全ての s=t に新変数が含まれているなら、B1は常に失敗する
%%        reduction_of_goal1_sub2(UnifyList,LeftVars)
    ),!,
   fail.

% B1= \+(A=B)で、AとBが常にユニファイ不可能な場合
reduction_of_goal1_sub2(HeadL,HeadL,B1,
       BodyL,BodyL,BTemp,[true|BTemp], LeftVars,LeftVars):-
   B1=(\+(A=B)),
   \+(unifiable(A,B)),!.


%  B1が選言の場合
reduction_of_goal1_sub2(HeadL,HeadL,B1,BodyL,BodyL,
            BTemp,BTemp2,LeftVars,LeftVars2):-
   compound(B1),
   functor(B1,';',2),
   disjunction_to_list(B1,B1_OR),!,
        % B1_OR のdisjunctがすべてfail する場合
        % この呼び出しは fail する

   reduction_of_goal1_sub2_or(0,B1_OR,B2_OR-[],LeftVars),

% reduction_of_goal1_sub2_or(0,[A==false,(A==true,cont(B))],_7850-[],[A,_2687,_2673,B]).
% reduction_of_goal1_sub2_or(0,[A==false,(A==true,cont(C,E))],X-[],[A,Y,Z,B]).
% reduction_of_goal1_sub2_or(0,[(A==true,cont(C,E))],X-[],[A,Y,Z,B]).
% reduction_of_goal1_sub2_or(0,[(A==true,cont)],X-[],[A,Y,Z,B]).
% reduction_of_goal1_sub2_or(0,[A==false,(A==true,cont)],X-[],[A,Y,Z,B]).

   list_to_disjunction(B2_OR,B2),
   get_variable(B2,B2Var),
   variable_union(B2Var,LeftVars,LeftVars2),
   [B2|BTemp] = BTemp2.

%   疑問、削るべし
% B1= cont(...)の場合
% 呼び出すContが無い時は失敗する。
reduction_of_goal1_sub2(Head1List,Head1List,B1,Body1List,Body1List,
        Body2Temp,[B1|Body2Temp],VarList1,LeftVars2):-
   compound(B1),
   functor(B1,cont,ContArity),
   functor(Cont,cont,ContArity),!,  % Cont はmost general atom
   ( clause(cont_for_step3(Cont),true)
   ; clause(cont_for_step3(Cont1),true),
       Cont1 =.. [(:-), ContHead | _],
       unifiable(Cont,ContHead)
   ),
   get_variable(B1,B1Var),
   variable_union(B1Var,VarList1,LeftVars2).

% B1=(A=B)の場合
%  A=Bを実行する
reduction_of_goal1_sub2(H1L,H1L2,B1,
      Body1L,Body1L2,BTemp,BTemp2,VarL1,VarL2):-
   B1 = (A=B),!,
   unifiable(A,B),!,
   reduction_of_goal1_sub3(H1L,H1L2,A,B,
       Body1L,Body1L2,BTemp,BTemp2,VarL1,VarL2).

% B1がその他の場合
reduction_of_goal1_sub2(HL,HL,B1,BL,BL,BTemp,[B1|BTemp],VarL1,VarL2):-
   get_variable(B1,B1Var), 
   variable_union(B1Var,VarL1,VarL2).

%\+A=Bが常に失敗するかを調べる
reduction_of_goal1_sub2([],_).

reduction_of_goal1_sub2([(A=B)|UnifyList],VarList):-
   ( var(A), \+(variable_member(A,VarList))
   ; var(B), \+(variable_member(B,VarList))
   ),!,
   reduction_of_goal1_sub2(UnifyList,VarList).

reduction_of_goal1_sub2([(A=B)|UnifyL],VarL):-
   ground(A),
   ground(B),
   unifiable(A,B),!,
   reduction_of_goal1_sub2(UnifyL,VarL).

% [B|B1L] は 選言を表す
reduction_of_goal1_sub2_or(N,[B|B1L], X-Z, VarL):-
   reduction_of_goal1_sub2_or2(N,N1,B,X-Y, VarL),!,
   reduction_of_goal1_sub2_or(N1,B1L, Y-Z, VarL).

reduction_of_goal1_sub2_or(1,[],Z-Z,_).

%% >>>>>>>>>>>>>> 修正 T.Sato, Oct. 2003 >>>>>>>>>>>>
reduction_of_goal1_sub2_or2(_,1,B,X-Y,VarL):-
% B は単なる disjunct, 簡単化したB2を返す
   ( conjunction_to_list(B,BL),
       reduction_of_goal2(BL,BL2,VarL),  % fails if BL is always false
       list_to_conjunction(BL2,B2),
       X =[B2|Y]
   ; X=Y ),!.

%%% <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
% Original codes:
%reduction_of_goal1_sub2_or2(_,1,B,[B2|B2List2]-B2List2,VarList):-
%   conjunction_to_list(B,BList),
%   reduction_of_goal2(BList,BList2,VarList),  % fail if B is always false
%   list_to_conjunction(BList2,B2).

reduction_of_goal1_sub2_or2(N,N,_,B2List2-B2List2,_).

%A=Bを実行する(check_restまで)
%-------------------------------
reduction_of_goal1_sub3(Head1List,Head1List2,A,B,
    Body1List,Body1List2,Body2Temp,Body2Temp2,VarList1,VarList2):-
    var(A),
    ( var(B),
         \+(variable_member(B,VarList1)),
         VarList2 = [B|VarList1]
    ; ground(B),
         VarList2 = VarList1
    ),
    variable_to_term(A,B,Head1List,Head1List2),
    variable_to_term(A,B,Body1List,Body1List2),
    variable_to_term(A,B,Body2Temp,Body2Temp2).

reduction_of_goal1_sub3(Head1List,Head1List2,A,B,
      Body1List,Body1List2,Body2Temp,Body2Temp2,VarList1,VarList2):-
    var(B),
    ground(A),
    VarList2 = [B|VarList1],
    variable_to_term(B,A,Head1List,Head1List2),
    variable_to_term(B,A,Body1List,Body1List2),
    variable_to_term(B,A,Body2Temp,Body2Temp2).

reduction_of_goal1_sub3(Head1List,Head1List,A,B,
       Body1List,Body1List,Body2Temp,[true|Body2Temp],
       VarList1,VarList1):-
     ground(A),
     ground(B),
     unifiable(A,B).

reduction_of_goal1_sub3(Head1List,Head1List2,A,B,
        Body1List,Body1List2,Body2Temp,Body2Temp2,VarList1,VarList2):-
     unify(A,B,UnifyList),
     reduction_of_goal1_sub4(UnifyList,Head1List,Head1List2,
     Body1List,Body1List2,Body2Temp,Body2Temp2),
     get_variable((A=B),Var),
     variable_union(Var,VarList1,VarList2).

reduction_of_goal1_sub3(Head1List,Head1List,A,B,
       Body1List,Body1List,Body2Temp,[(A=B)|Body2Temp],
       VarList1,VarList2):-
   var(A),
   check_rest(A,B,Body1List),
   get_variable((A=B),Var),
   variable_union(Var,VarList1,VarList2).

reduction_of_goal1_sub3(Head1List,Head1List,A,B,
     Body1List,Body1List,Body2Temp,[(A=B)|Body2Temp],
     VarList1,VarList2):-
   var(A),
   check_rest(A,B,Body1List),
   get_variable((A=B),Var),
   variable_union(Var,VarList1,VarList2).

reduction_of_goal1_sub3(Head1List,Head1List,A,B,
     Body1List,Body1List,Body2Temp,[(A=B)|Body2Temp],
     VarList1,VarList2):-
   get_variable((A=B),Var),
   variable_union(Var,VarList1,VarList2).

reduction_of_goal1_sub4([],Head1List,Head1List,
   Body1List,Body1List,Body2Temp,Body2Temp).

reduction_of_goal1_sub4([(A=B)|UnifyList],Head1List,Head1List3,
       Body1List,Body1List3,Body2Temp,Body2Temp3):-
   var(A),
   variable_to_term(A,B,Head1List,Head1List2),
   variable_to_term(A,B,Body1List,Body1List2),
   variable_to_term(A,B,Body2Temp,Body2Temp2),!,

   reduction_of_goal1_sub4(UnifyList,Head1List2,Head1List3,
   Body1List2,Body1List3,Body2Temp2,Body2Temp3).

reduction_of_goal1_sub4([(A=B)|UnifyList],Head1List,Head1List3,
      Body1List,Body1List3,Body2Temp,Body2Temp3):-
   var(B),
   variable_to_term(B,A,Head1List,Head1List2),
   variable_to_term(B,A,Body1List,Body1List2),
   variable_to_term(B,A,Body2Temp,Body2Temp2),!,
   reduction_of_goal1_sub4(UnifyList,Head1List2,Head1List3,
   Body1List2,Body1List3,Body2Temp2,Body2Temp3).

reduction_of_goal1_sub4([(A=B)|UnifyList],Head1List,Head1List3,
   Body1List,Body1List3,Body2Temp,Body2Temp3):-
   ground(A),
   ground(B),
   Body2Temp2 = [true|Body2Temp],!,
   reduction_of_goal1_sub4(UnifyList,Head1List,Head1List3,
   Body1List,Body1List3,Body2Temp2,Body2Temp3).

check_rest(_,_,[]).
check_rest(A,B,[B1|_]):-
   B1 = (C=D),
   var(C),
   A == C,
   \+(unifiable(B,D)),!,
   fail.
check_rest(A,B,[_|Body1List]):-
   check_rest(A,B,Body1List).

%-----------------------------
% reduction_of_goal2(Body1L,Body2L,VarL)
% Body1L は連言だと思っている
reduction_of_goal2(Body1L, Body2L, VarL):-
   reduction_of_goal2_dl(Body1L,Y-[],VarL),
%%%>>>>>>> 修正 yuizumi 01/May/2008 >>>>>>>
%  delete(true,Y,Body2L).
   fo_delete(true,Y,Body2L).
%%%<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


% Bodyが[]の場合
reduction_of_goal2_dl([], Body2L-Body2L, _).

% B1が連言の場合
reduction_of_goal2_dl([B1|Body1L],Body2L1-Body2L3,VarL):-
   compound(B1),
   functor(B1,(','),2),
   conjunction_to_list(B1,B1L),
   fo_append(B1L,Body1L,Body1L2),!,
   reduction_of_goal2_dl(Body1L2,Body2L1-Body2L3,VarL).

% それ以外
reduction_of_goal2_dl([B1|Body1L],Body2L1-Body2L3,VarL1):-
   reduction_of_goal2_sub(B1,Body1L,Body2L1-Body2L2,
        VarL1,VarL2),!,
   reduction_of_goal2_dl(Body1L,Body2L2-Body2L3,VarL2).

% \+A=Bが常に失敗する場合
reduction_of_goal2_sub(B1,_,_,VarList1,_):-
   B1 = \+(_=B),
   var(B),
   \+(variable_member(B,VarList1)),!,
   fail.

% \+A=Bが常に失敗する場合
reduction_of_goal2_sub(B1,_,_,_,_):-
   B1 = \+(A=B),
   ground(A),
   ground(B),
   unifiable(A,B),!,
   fail.

% \+A=Bが常に失敗する場合
reduction_of_goal2_sub(B1,_,_,VarList1,_):-
   B1 = \+(A=B),
   unify(A,B,UnifyList),
   reduction_of_goal1_sub2(UnifyList,VarList1),!,
   fail.

% B1 = (A=B)の場合(Bが新変数の場合)
reduction_of_goal2_sub(B1,_,[true|BodyL]-BodyL,VarL,VarL):-
   B1 = (A=B),
   var(B),
   ( \+(variable_member(B,VarL))
   ; A == B
   ),!,
  call(A=B).

% B1 = (\+A=B)が常に成功する場合
reduction_of_goal2_sub(B1,_,[true|BodyL]-BodyL,VarL,VarL):-
   B1 = (\+(A=B)),
   \+(unifiable(A,B)).

% B1が選言の場合
reduction_of_goal2_sub(B1,_,[B2|BodyL]-BodyL,
                VarL1,VarL2):-
   compound(B1),
   functor(B1,';',2),
   disjunction_to_list(B1,B1_OR),
   reduction_of_goal1_sub2_or(0,B1_OR,B2_OR-[],VarL1),!,
      % B2_OR=[] => B2=fail [B2|BodyL]は連言なので、
      % ここで fail する.
   B2_OR \== [],
   list_to_disjunction(B2_OR,B2),
   get_variable(B2,B2Var),
   variable_union(B2Var,VarL1,VarL2).


% B1 = (cont(_,..) ; D) で cont(_,..)の定義が assert されていないとき、
% fail する 
reduction_of_goal2_sub(B1,_,[B1|Body2L2]-Body2L2,VarL1,VarL2):-
   compound(B1),
   functor(B1,cont,ContArity),
   functor(Cont,cont,ContArity),!,
   ( clause(cont_for_step3(Cont),true)
   ; clause(cont_for_step3(Cont1),true),
       Cont1 =.. [(:-)|[ContHead|_]],
       unifiable(Cont,ContHead)
   ),
   get_variable(B1,B1Var),
   variable_union(B1Var,VarL1,VarL2). 

% B1がA=Bで、AもBもgroundの場合
reduction_of_goal2_sub(B1,_,[true|Body2L2]-Body2L2,
            VarL1,VarL1):-
   B1 = (A=B),
   ground(A),
   ground(B),!,
   unifiable(A,B).

% B1がA=Bの場合
reduction_of_goal2_sub(B1,Body1L,[B1|Body2L2]-Body2L2,
            VarL1,VarL2):-
   B1 = (A=B),!,
   unifiable(A,B),!,
   reduction_of_goal2_sub2(A,B,Body1L),
   get_variable((A=B),Var),
   variable_union(Var,VarL1,VarL2).

% B1がその他の場合
reduction_of_goal2_sub(B1,_,[B1|BodyL]-BodyL,VarL1,VarL2):-
   get_variable(B1,B1Var),
   variable_union(B1Var,VarL1,VarL2).

reduction_of_goal2_sub2(A,B,BodyL):-
   var(A),!,
      % (A=B) がBodyL中の(C=D)と両立しないとfail する
   check_rest(A,B,BodyL).
reduction_of_goal2_sub2(_,_,_).

%--------------------------------------------------------------------
% fo_sort(list,Var) <= 廃止する

% check_x(X1,X2)
%  closure clauseの頭 closure(x,f(w))のxのうち、
%  X=fo_sort(_,Var)のものをVarに置き換える
check_x([],[]).
check_x(X1,X2):-
   check_x_dl(X1,X2-[]).

check_x_dl([],X2-X2).
check_x_dl([X|X1],X2-X4):-
   check_x_dl_sub(X,X2-X3),!,
   check_x_dl(X1,X3-X4).

check_x_dl_sub(X,[Var|X2]-X2):-
   nonvar(X),
   X = fo_sort(_,Var).
check_x_dl_sub(X,[X|X2]-X2).


%remove_fo_sort(ALL,ALL).

% remove_fo_sort(ALL1,ALL2)
%  ALL1中のfo_sort(_,Var)をVarに置き換える
remove_fo_sort(ALL1,ALL2):-
   ALL1 =.. ALL1List,
   remove_fo_sort_dl(ALL1List,ALL2List-[]),
   ALL2 =.. ALL2List,!.

remove_fo_sort_dl([Term|ALL1],ALL2-ALL4):-
   remove_fo_sort_dl_sub(Term,ALL2-ALL3),!,
   remove_fo_sort_dl(ALL1,ALL3-ALL4).
remove_fo_sort_dl([],ALL2-ALL2).

remove_fo_sort_dl_sub(Term,[Term|ALL2]-ALL2):-
   simple(Term).
remove_fo_sort_dl_sub(Term1,[Var|ALL2]-ALL2):-
   compound(Term1),
   Term1 = fo_sort(_,Var).
remove_fo_sort_dl_sub(Term1,[Term2|ALL2]-ALL2):-
   compound(Term1),
   remove_fo_sort(Term1,Term2).


% get_fo_sort(ClosureHead,Fo_SortList)
%  ClosureHeadからfo_sort(Fo_SortName,Var)を取ってきて
%  リストにする
get_fo_sort(ClosureHead,Fo_SortList):-
   ClosureHead =.. ClosureHeadList,
   get_fo_sort_dl(ClosureHeadList,Fo_SortList-[]).
get_fo_sort_dl([],Fo_SortList-Fo_SortList).

get_fo_sort_dl([Term|ClosureHeadList],
                 Fo_SortList1-Fo_SortList3):-
   get_fo_sort_sub(Term,Fo_SortList1-Fo_SortList2),!,
   get_fo_sort_dl(ClosureHeadList,Fo_SortList2-Fo_SortList3).

get_fo_sort_sub(Term,[Term|Fo_SortList2]-Fo_SortList2):-
   compound(Term),
   Term = fo_sort(_,_).
get_fo_sort_sub(_,Fo_SortList2-Fo_SortList2).


use_fo_sort(TEMP1,[Fo_Sort|Fo_SortList],Clause):-
   Fo_Sort = fo_sort(Fo_SortName,Var),
   clause(fo_sort(Fo_SortName,List),true),
   use_fo_sort_sub(TEMP1,Var,List,Clause),!,
   use_fo_sort(TEMP1,Fo_SortList,Clause).
use_fo_sort(_,[],_).

% Clause の Var をソートの値の一つ L で置き換えたものを
% TEMP1 に書き込む.
use_fo_sort_sub(TEMP1,Var,[L|List],Clause):-
   Clause =.. ClauseList,
   variable_to_term(Var,L,ClauseList,Clause2List),
   Clause2 =.. Clause2List,
   write_clause(TEMP1,Clause2),!,
   use_fo_sort_sub(TEMP1,Var,List,Clause).
use_fo_sort_sub(_,_,[],_).

%----------misc------------------------------------------

% fo_reverse(List1,List2)
%  List1を反転させたリストがList2
fo_reverse(List1,List2):-
   fo_reverse_dl(List1,List2-[]).

fo_reverse_dl([],List2-List2).

fo_reverse_dl([L1|List1],List2-List3):-
   fo_reverse_dl(List1,List2-[L1|List3]).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Not built_in for B-Prolog
% simple(X)
%  Xは変数または定数である。
simple(X):-
  (var(X); constant(X)).

% constant(X)
%  Xは定数である。
constant(X):-
   ( atom(X)
   ; integer(X)
   ; float(X) ),!.

% ground(X)
%  Xは基底項である。
%ground(X):-
%   constant(X).
%ground(X):-
%   compound(X),
%   X=..[_|XArgList],
%   ground_arg(XArgList).
%ground_arg([]).
%ground_arg([XArg|XArgList]):-
%   ground(XArg),!,
%   ground_arg(XArgList).

% member(L,List)
%  LはListの要素である。
%member(L,[L|_]).
%member(L,[_|List]):-
%   member(L,List).

% length(List,N)
%  Listの長さはNである。
%length(List,N):-
% length(List,0,N).
%
%length([],N,N).
%length([_|List],T,N):-
%  T1 is T+1,!,
%  length(List,T1,N).

% fo_append(List1,List2,List3)
%  List1とList2を繋げたリストがList3
fo_append([],List3,List3).
fo_append([L1|List1],List2,[L1|List3]):-
   fo_append(List1,List2,List3).

fo_append([],[],List4,List4).
fo_append([],[L2|List2],List3,[L2|List4]):-
   fo_append([],List2,List3,List4).
fo_append([L1|List1],List2,List3,[L1|List4]):-
   fo_append(List1,List2,List3,List4).

% fo_list(X)
%  Xはリストである。
fo_list([]).
fo_list([_|_]).

% flat(List1,List2)
%  List1を一段階だけ平坦化したリストがList2
%    ?- flat([A,[B,C,[D,E],F],G,[H,[I,J],K],L],JJ).
%  JJ = [A,B,C,[D,E],F,G,H,[I,J],K,L]
flat(List1,List2):-
   flat_dl(List1,List2-[]).
flat_dl([],List2-List2).

flat_dl([L1|List1],List2-List4):-
   flat_dl_sub(L1,List2-List3),!,
   flat_dl(List1,List3-List4).
flat_dl_sub(L1,List2-List3):-
   fo_list(L1),!,
   fo_append(L1,List3,List2).
flat_dl_sub(L1,[L1|List2]-List2).

% list_member(List1,List2)
%  List1の各要素をリストにする。
%    ?- list_member([A,B],Y).
%    Y=[[A],[B]]
%  ただし、
%    ?- list_member([X],Y).
%    Y=[X]
%  となる。
list_member([],[]).
list_member([L1|[]],[L1|[]]).

list_member(List1,List2):-
   list_member_dl(List1,List2-[]).
list_member_dl([],List2-List2).

list_member_dl([L1|List1],List2-List4):-
   list_member_dl_sub(L1,List2-List3),!,
   list_member_dl(List1,List3-List4).
list_member_dl_sub(L1,[[L1|[]]|List2]-List2).

% product(List1,List2,List3)
%  List3はList1とList2の直積
product([],_,[]).
product(_,[],[]).
product(List1,List2,List3):-
   product_dl(List1,List2,List3-[]).

product_dl([],_,List3-List3).
product_dl([L1|List1],List2,List3-List5):-
   product_dl_sub(L1,List2,List3-List4),!,
   product_dl(List1,List2,List4-List5).

product_dl_sub(_,[],List3-List3).
product_dl_sub(L1,[L2|List2],List3-List5):-
   product_dl_sub2(L1,L2,List3-List4),!,
   product_dl_sub(L1,List2,List4-List5).

product_dl_sub2(L1,L2,[[L1|[L2|[]]]|List3]-List3).

% variable_to_term(Var,Term,List1,List2)
%  List中のVarをTermに変える（代入,substitution）
variable_to_term(_,_,[],[]).

variable_to_term(Var,Term,[L1|List1],List2):-
   Var == L1,!,
   List2 = [Term|List3],!,
   variable_to_term(Var,Term,List1,List3).

variable_to_term(Var,Term,[L1|List1],List3):-
   compound(L1),!,
   L1=.. LList1,
   variable_to_term(Var,Term,LList1,LList2),
   L2 =.. LList2,
   List3 = [L2|List4],!,
   variable_to_term(Var,Term,List1,List4).

variable_to_term(Var,Term,[L1|List1],[L1|List2]):-
   variable_to_term(Var,Term,List1,List2).

% variable_list(List)
%  Listの要素は全て変数である。
variable_list([]).

variable_list([L|List]):-
   var(L),!,variable_list(List).

% same_variable_list(List1,List2)
%  変数のリストList1とList2は、
%  同じ変数の集合を表す
same_variable_list([],[]).
same_variable_list(_,[]):-
   fail.
same_variable_list([L1|List1],List2):-
   var(L1),
   variable_delete(L1,List1,List3),
   variable_delete(L1,List2,List4),!,
   same_variable_list(List3,List4).

%%% >>>>>>>> 追加 by T.Sato, Sept. 2003 >>>>>>>>>>>>>
% diff_variable_list(List)
%  List は相異なる変数のリストである
diff_variable_list([X|Y]):-!,
   var(X),
   \+ variable_member(X,Y),
   diff_variable_list(Y).
diff_variable_list([]).
%%% <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

% variable_member(Var,List)
%  変数VarはListの要素である。
variable_member(Var,[L|_]):-
   Var == L.
variable_member(Var,[_|List]):-
   variable_member(Var,List).

% variable_delete(Var,List1,List2)
%  List1から変数Varを取り除いたリストがList2
variable_delete(Var,List1,List2):-
   var(Var),!,
   variable_delete_dl(Var,List1,List2-[]).
variable_delete_dl(_,[],List2-List2).

variable_delete_dl(Var,[L1|List1],List2-List4):-
   variable_delete_dl_sub(Var,L1,List2-List3),!,
   variable_delete_dl(Var,List1,List3-List4).

variable_delete_dl_sub(Var,L1,List2-List2):-
   Var==L1.
variable_delete_dl_sub(_,L1,[L1|List2]-List2).

% variable_unique(List1,List2)
%  List2はList1から重複する変数を取り除いたリスト。
variable_unique([],[]).

variable_unique([L1|List1],[L1|List3]):-
   var(L1),!,
   variable_delete(L1,List1,List2),!,
   variable_unique(List2,List3).
variable_unique([L1|List1],[L1|List2]):-
   variable_unique(List1,List2).

% variable_diff(List1,List2,List3)
%  List3はList1とList2の差集合
variable_diff(List1,[],List1).
variable_diff([],_,[]).

variable_diff([L1|List1],List2,List3):-
   variable_member(L1,List2),!,
   variable_diff(List1,List2,List3).

variable_diff([L1|List1],List2,[L1|List3]):-
   variable_diff(List1,List2,List3).

% variable_union(List1,List2,List3)
%  List3はList1とList2の和集合
%  ただし、List1とList2の要素は全て変数とする。
variable_union([],List3,List3).
 
variable_union([L1|List1],List3,List5):-
   var(L1),
   variable_member(L1,List3),!,
   variable_delete(L1,List1,List2),
   variable_delete(L1,List3,List4),
   List5 = [L1|List6],!,
   variable_union(List2,List4,List6).

variable_union([L1|List1],List3,List4):-
   var(L1),!,
   variable_delete(L1,List1,List2),
   List4 = [L1|List5],!,
   variable_union(List2,List3,List5).

variable_union([L1|List1],List2,[L1|List3]):-
   variable_union(List1,List2,List3).

% variable_join(List1,List2,List3)
%  List3は、List1とList2の積集合

variable_join([],_,[]).
variable_join([L1|List1],List2,W):-
   variable_member(L1,List2),!,
   W = [L1|List3],
   variable_join(List1,List2,List3).

variable_join([_|List1],List2,List3):-
   variable_join(List1,List2,List3).

% exist_common_variable(List1,List2)
%  List1とList2に共通な変数が存在する。
exist_common_variable([L1|_],List2):-
   variable_member(L1,List2).

exist_common_variable([_|List1],List2):-
   exist_common_variable(List1,List2).

% get_variable(Term,VarList)
%  VarListは、Term中の変数のリスト
%  変数の複数回の出現を保存する
%   ?- get_variable([a,f(X),g(X)],Z) => Z = [X,X]

get_variable(Term,[Term]):-
   var(Term).
get_variable(Term,[]):-
   constant(Term).
get_variable(Term,VarList):-
   compound(Term),!,
   Term =.. [_|ArgList],
   get_variable_dl(ArgList,VarList-[]).

get_variable_dl([Arg|ArgList],VarList1-VarList3):-
   get_variable_dl2(Arg,VarList1-VarList2),!,
   get_variable_dl(ArgList,VarList2-VarList3).
get_variable_dl([],VarList-VarList).

get_variable_dl2(Arg,[Arg|VarList]-VarList):-
   var(Arg).
get_variable_dl2(Arg,VarList-VarList):-
   constant(Arg).
get_variable_dl2(Arg,VarList1-VarList2):-
   compound(Arg),!,
   Arg =..[_|ArgArgList],
   get_variable_dl(ArgArgList,VarList3-[]),
   fo_append(VarList3,VarList2,VarList1).

% get_free_variable(Term,FVar)
%  Termに含まれる自由変数をリストにしたものがFVarである
get_free_variable(Term,FVar):-
   get_variable(Term,Var),        % Termに含まれる変数
   get_bound_variable(Term,BV),   % Termに含まれる束縛変数
   variable_diff(Var,BV,FVar).    % Termに含まれる自由変数は、
                                  % Termに含まれる変数から束縛変数を
                                  % 取り除いたもの

% get_bound_variable(Term,NonFVar)
%  Termに含まれる束縛変数をリストにしたものがNonFVar
get_bound_variable(List,BV):-
   fo_list(List),!,     % 入力がリストの場合
   get_bound_variable_dl_list(List,BV,[]).
get_bound_variable(Term,BV):-
   get_bound_variable_dl(Term,BV,[]).

get_bound_variable_dl(Term,BV1,BV2):-
   nonvar(Term),
   ( (Term =(F1 ; F2) ; Term =(F1 , F2)),!,
       get_bound_variable_dl(F1,BV1,Y),
       get_bound_variable_dl(F2,Y,BV2)
   ; Term = not(F1),
       get_bound_variable_dl(F1,BV1,BV2)
   ),!.
get_bound_variable_dl(Term,BV1,BV2):-
     % Termがall(V,(A->B))の場合
     % 束縛変数は、A中にある束縛変数と、
     % B中にある束縛変数と、
   nonvar(Term),
   Term = all(All,IMPLY),!,
   IMPLY= imply(LEFT,RIGHT),
   get_bound_variable_dl(LEFT,X,Y),
   get_bound_variable_dl(RIGHT,Y,All),
   fo_append(X,BV2,BV1).
get_bound_variable_dl(Term,BV1,BV2):-
     % Termがexist(V,A)の場合
     % 束縛変数は、A中の束縛変数とV
   nonvar(Term),
   Term = exist(EVar,EArg),!,
   get_bound_variable_dl(EArg,EBV,EVar),
   fo_append(EBV,BV2,BV1).
get_bound_variable_dl(Term,BV,BV):-
   ( var(Term) ; constant(Term)),!.
get_bound_variable_dl(Term,BV1,BV2):-
   Term =.. TList,
  get_bound_variable_dl_list(TList,BV1,BV2).

get_bound_variable_dl_list([Term|X],BV1,BV3):-
   get_bound_variable_dl(Term,BV1,BV2),!,
   get_bound_variable_dl_list(X,BV2,BV3).
get_bound_variable_dl_list([],BV,BV).

%% 追加 by kameya

write_clause(Stream,Clause) :- portray_clause(Stream,Clause).
write_clause(Clause) :- portray_clause(Clause).
write_clause2(Stream,Clause) :- portray_clause(Stream,Clause).

%% 無効化 by kameya
%%
%% % write_clause(Stream,Clause)
%% %  StreamにClauseを出力する
%% write_clause(Stream,Clause):-
%%    ( functor(Clause,(:-),1),
%%        Clause =..[(:-),BDY],
%%        functor(BDY,include,_),
%%        write(Stream,(:-)),
%%        writeq(Stream,BDY)
%% %  ; write(Stream,Clause) ),!,
%%    ; writeq(Stream,Clause) ),!,          % 修正 by yuizumi
%%    write(Stream,'.'),
%%    nl(Stream).
%%
%% % write_clause(Clause)
%% %  Clauseを画面に表示する
%% write_clause(Clause):-
%%    ( functor(Clause,(:-),1),
%%        Clause =..[(:-),BDY],
%%        functor(BDY,include,_),
%%        write((:-)),
%%        writeq(BDY)
%% %  ; write(Clause) ),!,
%%    ; writeq(Clause) ),!,          % 修正 by yuizumi
%%    write('.'),
%%    nl.
%%
%% % write_clause2(Stream,Clause)
%% %  StreamにClauseを出力する
%% %  この時、一度しか現れない変数を'_'に置き換え、
%% %  また、numbervarsで変数名を分かりやすくする。
%% write_clause2(Stream,Clause):-
%%  %  一度しか出てこない変数を'_'に置き換える
%%    Clause =.. ClauseList,
%%    get_variable(Clause,ClauseVar),
%%    singleton_variable_to_under_score(ClauseList,ClauseVar,Clause2List),
%%    Clause2 =.. Clause2List,
%%    copy_term(Clause2,Clause3),
%%    numbervars(Clause3,0,_),
%%    write_clause3(Stream,Clause3).
%%
%% % write_clause3(Stream,Clause)
%% %  節を整形して出力する。
%% write_clause3(Stream,Clause):-
%%    get_head_body(Clause,Head,BodyList),
%%    write_clause3_sub(Stream,Head,BodyList).
%%
%% write_clause3_sub(Stream,Head,[]):-
%%    write_clause(Stream,Head).
%%
%% write_clause3_sub(Stream,Head,[Body|[]]):-
%%    compound(Body),
%%    functor(Body,';',2),
%% %  write(Stream,Head),
%%    writeq(Stream,Head),             % 修正 by yuizumi
%%    write(Stream,':-'),
%%    nl(Stream),
%%    write_clause3_or(Stream,1,Body),
%%    write(Stream,'.'),
%%    nl(Stream).
%%
%% write_clause3_sub(Stream,Head,[Body|[]]):-
%%    generate_clause(Head,[Body|[]],Clause),
%%    write_clause(Stream,Clause).
%%
%% write_clause3_sub(Stream,Head,[Body1|[Body2|[]]]):-
%%    ( simple(Body1)
%%    ; compound(Body1),
%%       \+(functor(Body1,';',2))
%%    ),
%%    ( simple(Body2)
%%    ; compound(Body2),
%%       \+(functor(Body2,';',2))
%%    ), 
%%    generate_clause(Head,[Body1|[Body2|[]]],Clause),
%%    write_clause(Stream,Clause).
%%
%% write_clause3_sub(Stream,Head,BodyList):-
%% %  write(Stream,Head),
%%    writeq(Stream,Head),          % 修正 by yuizumi
%%    write(Stream,':-'),
%%    nl(Stream),
%%    write_clause3_sub2(Stream,1,BodyList).
%%
%% write_clause3_sub2(Stream,N,[Body|[]]):-
%%    compound(Body),
%%    functor(Body,';',2),
%%    write_clause3_or(Stream,N,Body),
%%    write(Stream,'.'),
%%    nl(Stream).
%%
%% write_clause3_sub2(Stream,N,[Body|[]]):-
%%    write_clause4(Stream,N),
%%    write_clause(Stream,Body).
%%
%% write_clause3_sub2(Stream,N,[Body|BodyList]):-
%%    compound(Body),
%%    functor(Body,';',2),
%%    write_clause3_or(Stream,N,Body),
%%    write(Stream,(',')),
%%    nl(Stream),!,
%%    write_clause3_sub2(Stream,N,BodyList).
%%
%% write_clause3_sub2(Stream,N,[Body|BodyList]):-
%%    write_clause4(Stream,N),
%% %  write(Stream,Body),
%%    writeq(Stream,Body),          % 修正 by yuizumi
%%    write(Stream,(',')),
%%    nl(Stream),!,
%%    write_clause3_sub2(Stream,N,BodyList).
%%
%% write_clause3_sub3(Stream,N,[Body|[]]):-
%%    compound(Body),
%%    functor(Body,';',2),
%%    write_clause3_or(Stream,N,Body),
%%    nl(Stream).
%%
%% write_clause3_sub3(Stream,N,[Body|[]]):-
%%    write_clause4(Stream,N),
%% %  write(Stream,Body),
%%    writeq(Stream,Body),          % 修正 by yuizumi
%%    nl(Stream).
%%
%% write_clause3_sub3(Stream,N,[Body|BodyList]):-
%%    compound(Body),
%%    functor(Body,';',2),
%%    write_clause3_or(Stream,N,Body),
%%    write(Stream,(',')),
%%    nl(Stream),!,
%%    write_clause3_sub3(Stream,N,BodyList).
%%
%% write_clause3_sub3(Stream,N,[Body|BodyList]):-
%%    write_clause4(Stream,N),
%% %  write(Stream,Body),
%%    writeq(Stream,Body),          % 修正 by yuizumi
%%    write(Stream,(',')),
%%    nl(Stream),!,
%%    write_clause3_sub3(Stream,N,BodyList).
%%
%% write_clause3_or(Stream,N,OR):-
%%    write_clause4(Stream,N),
%%    write(Stream,'(' ),
%%    nl(Stream),
%%    N1 is N + 1,
%%    disjunction_to_list(OR,ORList),
%%    write_clause3_or_sub(Stream,N,N1,ORList),
%%    write_clause4(Stream,N),
%%    write(Stream,')' ).
%%
%% write_clause3_or_sub(Stream,_,N1,[O|[]]):-
%%    conjunction_to_list(O,OList),
%%    write_clause3_sub3(Stream,N1,OList).
%%
%% write_clause3_or_sub(Stream,N,N1,[O|ORList]):-
%%    conjunction_to_list(O,OList),
%%    write_clause3_sub3(Stream,N1,OList),
%%    write_clause4(Stream,N),
%%    write(Stream,';'),
%%    nl(Stream),!,
%%    write_clause3_or_sub(Stream,N,N1,ORList).
%%
%% % write_clause4(Stream,N)
%% %  空白を4*N文字出力する。
%% write_clause4(_,0).
%% write_clause4(Stream,N):-
%%    N > 0,
%%    write(Stream,'    '),
%%    N1 is N - 1,!,
%%    write_clause4(Stream,N1).
%%
%% % write_list(List)
%% %  Listの要素を画面に表示する
%% write_list([L|List]):-
%%    write_clause(L),!,
%%    write_list(List).
%% write_list([]).

% generate_term(Name,ArgList,Term)
%  NameとArgのリストからTermを作る
generate_term(Name,[],Name).

generate_term(Name,ArgList,Term):-
   Term =.. [Name|ArgList].

% generate_clause(Head,BodyList,Clause)
%  HeadとBodyのリストから節を作る
generate_clause(Head,[],Head):-!.
generate_clause(Head,[Body],Clause):-!,
   generate_term((:-),[Head,Body],Clause).
generate_clause(Head,BodyList,Clause):-
   fo_list(BodyList),!,
   list_to_conjunction(BodyList,Body),
   generate_term((:-),[Head,Body],Clause).

% generate_left(LeftList,LEFT)
%  ->の左辺を作る
generate_left(LeftList,LEFT):-
   generate_term(left,LeftList,LEFT).

% generate_right(RightList,RIGHT)
%  ->の右辺を作る
generate_right(RightList,RIGHT):-
   generate_term(right,RightList,RIGHT).

% generate_imply(LEFT,RIGHT,IMPLY)
%  LEFTとRIGHTからimply(LEFT,RIGHT)を作る
generate_imply(LEFT,RIGHT,IMPLY):-
   compound(LEFT),functor(LEFT,left,_),
   compound(RIGHT),functor(RIGHT,right,_),!,
   generate_term(imply,[LEFT,RIGHT],IMPLY).

% generate_imply(LEFT,RIGHT,IMPLY)
%  LEFTとRIGHTからimply(LEFT,RIGHT)を作る
% （左辺は空）
generate_imply(LEFT,RIGHT,IMPLY):-
   LEFT=left,
   compound(RIGHT),functor(RIGHT,right,_),!,
   generate_term(imply,[left,RIGHT],IMPLY). 

% generate_all(All,LEFT,RIGHT,ALL)
%  All,LEFT,RIGHTから
%  all(All,imply(LEFT,RIGHT))を作る
generate_all(All,LEFT,RIGHT,ALL):-
   fo_list(All),!,
   ( All==[],LEFT=left(\+(G)),RIGHT=right(fail),
          ALL = all([],imply(left,right(G)))
%%%   ; All=[V],LEFT=left(A=B),(V==A ; V==B),  %% Never ever
%%%          A=B,                              %% execute A=B
%%%          ALL = all([],imply(left,RIGHT))   %% during compilation!
   ; true,
       generate_imply(LEFT,RIGHT,IMPLY),
       generate_term(all,[All,IMPLY],ALL)
   ),!.

generate_all(All,IMPLY,ALL):-
   compound(IMPLY),
   functor(IMPLY,imply,2),
   fo_list(All),!,
   generate_term(all,[All,IMPLY],ALL).

% get_head_body(Clause,Head,BodyList)
%  ClauseからHeadとBodyのリストを得る
get_head_body(Clause,Head,BodyList):-
   Clause =.. [(:-),Head,Body],
   conjunction_to_list(Body,BodyList).

get_head_body(Clause,Clause,[]):-
   Clause =.. [Name|_],\+(Name=(:-)).

% conjunction_to_list(Term,List)
%  連言をリストにする
conjunction_to_list(Term,List):-
  x_to_list((','),Term,List).

% disjunction_to_list(Term,List)
%   選言をリストにする
disjunction_to_list(Term,List):-
 x_to_list((;),Term,List).

% list_to_conjunction(List,Conjunction)
%  Listを連言にする。
list_to_conjunction([],true):-!.
list_to_conjunction(List,Conjunction):-
  list_to_x((','),List,Conjunction).

% list_to_disjunction(List,Disjunction)
%  Listを選言にする
list_to_conjunction([],fail):-!.
list_to_disjunction(List,Disjunction):-
   list_to_x((;),List,Disjunction).

% x_to_list(Ope,X,List)
%  xのOpeを'.'にする
x_to_list(Ope,Term,List):-
   x_to_list_dl(Ope,Term,List-[]),!.
x_to_list_dl(_,Term,[Term|List]-List):-
   simple(Term),!.
x_to_list_dl(Ope,Term,[Term|List]-List):-
   compound(Term),
   functor(Term,Name,_),
   \+(Name=Ope).

x_to_list_dl(Ope,Term,List1-List3):-
   Term =.. [_|[Left|[Right]]],
   List1 = [Left|List2],
   x_to_list_dl(Ope,Right,List2-List3).

% list_to_x(Ope,List,X)
%  Listの'.'をOpeにする
list_to_x(_,NotList,NotList):-
   \+(fo_list(NotList)).
list_to_x(_,[],[]).
list_to_x(_,[L],L).

list_to_x(Ope,[L|List],X):-
   list_to_x(Ope,List,RightX),
   X =.. [Ope, L, RightX].

% list_to_new_variable_list(List,NewList)
%  リストの要素を全て新しい変数にする。
list_to_new_variable_list([],[]).

list_to_new_variable_list([_|List],[_|NewList]):-
   list_to_new_variable_list(List,NewList). 

% get_last(List,Last)
%  Listの最後の要素がLast
get_last([],[]).
get_last([L|[]],L).
get_last([_|List],Last):-
   get_last(List,Last).

% get_last(List,Last,Rest)
%  Listの最後の要素がLast、
%  それ以外の要素のリストがRest
get_last([],[],[]).
get_last(List,Last,Rest):-
   get_last_dl(List,Last,Rest-[]).
get_last_dl([L|[]],L,Rest-Rest).
get_last_dl([L|List],Last,Rest1-Rest3):-
   Rest1 = [L|Rest2],!,
   get_last_dl(List,Last,Rest2-Rest3).

% some_variables_to_new_variable(VarList,List1,List2)
%  VarListのメンバーで、List1に含まれている変数を
%  新変数に換える

% 全ての変数を新変数にしたら終了
some_variables_to_new_variable([V|Varlist],List1,List3):-
    % 変数を新変数に換える
   variable_to_term(V,_,List1,List2),!,
    % 次の変数について同じ操作をする
   some_variables_to_new_variable(Varlist,List2,List3).
some_variables_to_new_variable([],List2,List2).

% 不要
% copy_file(INPUT,OUTPUT)
% INPUTから節をよんで、
% numbervarsで変数に名前を付けて B に書く
%%%%%%%%%%%%%%  19/05/08 削除 %%%%%%%%%%
% どこからも呼ばれていない
%copy_file(INPUT,OUTPUT):-
%   read(INPUT,Clause),
%   ( Clause == end_of_file,!
%   ; numbervars(Clause,0,_),
%       write_clause(OUTPUT,Clause),fail ).
%copy_file(INPUT,OUTPUT):-
% copy_file(INPUT,OUTPUT).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% append_name(AName,BName,NewName)
%  AとBを繋げたNameを作る
append_name(AName,BName,NewName):-
   name(AName,ANameList),  % nameをリストにする
   name(BName,BNameList),  % 繋げる
   fo_append(ANameList,BNameList,NewNameList), % nameにする
   name(NewName,NewNameList).

% append_name(AName,BName,CName,NewName)
%  AとBとCを繋げたNameを作る
%  例えば append_name('closure_',ModeName,ClosureNumber)から
% closure_ModeNameClosureNumber(closure_append0など)を作るのに使う
append_name(AName,BName,CName,NewName):-
   name(AName,ANameList),  % nameをリストにする
   name(BName,BNameList),
   name(CName,CNameList),  % 繋げる
   fo_append(ANameList,BNameList,CNameList,NewNameList),  % nameにする
   name(NewName,NewNameList).

%%%%<<<<<<<<<< 置き換え by T.Sato, Oct. 2003 <<<<<<<<<<<
% unifiable(A,B)
%  AとBはユニファイ可能である
unifiable(A,B):-
   copy_term(A,C), % Aの変数を新変数にした項がC
   copy_term(B,D), % Bの変数を新変数にした項がD
   ( C=D, R1=ok
   ; R1=no ),!,
   copy_term([A,B],[E,F]),  % A,Bの共通変数を考慮
   ( unifiable2(E,F), R2=ok % unifiable/2 はoccur check 入り
   ; R2=no ),
   ( R1 ==R2, R2==ok
   ; R1\==R2,
	    format(" WRONG unification routine!~n",[]),
        R2==ok ),!.

unifiable2(A,B):-
   var(A),var(B),!,
   A=B.
unifiable2(A,B):-
   var(A),nonvar(B),!,
   get_variable(B,VarB),
   \+variable_member(A,VarB),!, % occur check
   A=B.
unifiable2(A,B):-
   var(B),nonvar(A),!,
   unifiable2(B,A).

unifiable2(A,B):-
   functor(A,F,N),
   functor(B,F,N),
   A =.. [_|ArgA],
   B =.. [_|ArgB],
   unifiable3(ArgA,ArgB).

unifiable3([A|X],[B|Y]):-!,
   unifiable2(A,B),
   unifiable3(X,Y).
unifiable3([],[]):-!.

%%%%>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


% singleton_variable_to_under_score(ClauseList3,ClauseVar3,ClauseList2)
%  一度しか出てこない変数をunder score('_')にする
singleton_variable_to_under_score(ClauseList2,[],ClauseList2).

singleton_variable_to_under_score(ClauseList3,
                                  [Var|ClauseVar3],ClauseList2):-
   variable_member(Var,ClauseVar3),!,
   variable_delete(Var,ClauseVar3,ClauseVar4),!,
   singleton_variable_to_under_score(ClauseList3,
                                   ClauseVar4,ClauseList2).

singleton_variable_to_under_score(ClauseList3,
                                  [Var|ClauseVar3],ClauseList2):-
%%%>>>>>>> 修正 yuizumi 05/Nov/2008 >>>>>>>
%  variable_to_term(Var,'_',ClauseList3,ClauseList4),!,
   variable_to_term(Var,'$VAR'('_'),ClauseList3,ClauseList4),!,
%%%<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
   singleton_variable_to_under_score(ClauseList4,
                                   ClauseVar3,ClauseList2).

%%%>>>>>>> 削除 yuizumi 05/Nov/2008 >>>>>>>
%%% 未使用につき削除．Prism 1.12 の新述語との混乱防止．
/*
% varlist(List) Listの要素は全て変数
varlist([]).
varlist([A|B]):- var(A),!,varlist(B).
*/
%%%<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

%%%>>>>>>> 修正 yuizumi 01/May/2008 >>>>>>>
%%% B-Prolog 6.9+ では delete/3 が標準述語として提供されており，
%%% 名前の衝突が生じたことから述語名を変更した．ちなみに，両者
%%% (delete/3 と fo_delete/3) の機能は同等であるが，引数の順序
%%% に違いがある．詳細は B-Prolog のマニュアルを参照されたい．

% fo_delete(A,List1,List2) List1からAを取除いたのがList2
fo_delete(A,List1,List2):-
 fo_delete_dl(A,List1,List2-[]).

fo_delete_dl(_,[],List2-List2).

fo_delete_dl(A,[L1|List1],List21-List23):-
 fo_delete_dl_sub(A,L1,List21-List22),!,
 fo_delete_dl(A,List1,List22-List23).

fo_delete_dl_sub(A,A,List22-List22).
fo_delete_dl_sub(_,L1,[L1|List22]-List22).
%%%<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

%unify(X,Y,List)
unify(X,Y,List):-
 unifiable(X,Y),
 unify2(X,Y,List).

unify2(X,Y,List):-
 var(X),
 var(Y),
 List = [(X=Y)|[]].

unify2(X,Y,List):-
 var(X),
 nonvar(Y),
 List = [(X=Y)|[]].

unify2(X,Y,List):-
 var(Y),
 nonvar(X),
 List = [(Y=X)|[]].

unify2(X,Y,List):-
 nonvar(X),
 nonvar(Y),
 constant(X),
 constant(Y),
 List = [(X=Y)|[]].

unify2(X,Y,List):-
   nonvar(X),
   nonvar(Y),
   compound(X),
   compound(Y),
   term_unify2(X,Y,List).

term_unify2(X,Y,List):-
   functor(X,F,N),
   functor(Y,F,N),
   unify2_args(N,X,Y,List-[]).

unify2_args(N,X,Y,List1-List3):-
   N>0,
   unify2_arg(N,X,Y,List1-List2),
   N1 is N-1,
   unify2_args(N1,X,Y,List2-List3).
unify2_args(0,_,_,List-List).

unify2_arg(N,X,Y,List1-List2):-
   arg(N,X,ArgX),
   arg(N,Y,ArgY),
   unify2(ArgX,ArgY,List),
   fo_append(List,List2,List1).

fo_trace:-
  ( retract(fo_trace(_)) ; true ),!,
  assertz(fo_trace(yes)).
fo_notrace:-
  ( retract(fo_trace(_)) ; true ),!,
  assertz(fo_trace(no)).
