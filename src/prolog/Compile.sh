#! /bin/sh -x

BINDIR=../../bin

PAREA=4000000	# size of program area
STACK=2000000	# size of control stack and heap
TRAIL=2000000	# size of trail stack
TABLE=1000000	# size of table area

case `uname -m` in
    x86_64)
        PROC=64
        ;;
    *)
        PROC=32
        ;;
esac

case `uname -s` in
    Linux)
        BINARY=$BINDIR/prism_up_linux${PROC}.bin
        ;;
    Darwin)
        DARWIN_MAJOR=`uname -r | cut -d. -f 1`
        BINARY=$BINDIR/prism_up_darwin${DARWIN_MAJOR}.bin
        ;;
    CYGWIN*)
        BINARY=$BINDIR/prism_up_cygwin.exe
        ;;
esac

if [ ! -x "$BINARY" ]; then
    echo "`basename $0`: Can't execute \`${BINARY}'." 1>&2
    exit 1
fi

source=`basename $1 .pl`
target=`basename $1 .pl`.out

exec $BINARY -p $PAREA -s $STACK -b $TRAIL -t $TABLE $BINDIR/bp.out -g "set_prolog_flag(redefine_builtin,on),set_prolog_flag(stratified_warning,off),compile($source),halt"

## For profiling, use below instead of above
#exec $BINARY -p $PAREA -s $STACK -b $TRAIL -t $TABLE $BINDIR/bp.out -g "set_prolog_flag(redefine_builtin,on),set_prolog_flag(stratified_warning,off),profile_compile($source),halt"
