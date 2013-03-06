% file = fo_prismn.psm
% default file for the compiled program is "temp"

prismn(File)             :- prismn(File,temp).
prismn(File,File2)       :- prismn([],File,File2).  % for backward compatibility
prismn_opt(Options,File) :- prismn(Options,File,temp).

prismn(Options,File,File2):-
    $pp_search_file(File,File1,[".psm",""]),!,
    call($pp_foc(File1,File2)),!, % call/1 for the parallel version
    prism(Options,File2).
prismn(_,File,_):-
    $pp_raise_existence_error($msg(3001),[File],
                              [prism_file,File],existence_error).

$pp_foc(File1,File2):-
    fo(File1,File2),format("Compilation done by FOC~n~n",[]).

foc(File):- fo(File).
foc(File,File2):- fo(File,File2).
