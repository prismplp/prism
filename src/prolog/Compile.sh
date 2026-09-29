#! /bin/sh -x

BINDIR=../../bin

PAREA=4000000	# size of program area
STACK=2000000	# size of control stack and heap
TRAIL=2000000	# size of trail stack
TABLE=1000000	# size of table area

# Leave the original compiler path unchanged unless MinGW is enabled.
if [ "${PRISM_MINGW:-0}" = 1 ]; then
    case `uname -s` in
        MINGW*|MSYS*|Windows_NT)
            case ${PRISM_MINGW_BITS:-64} in
                32|64) ;;
                *) echo 'PRISM_MINGW_BITS must be 32 or 64' >&2; exit 1 ;;
            esac
            BINARY=$BINDIR/prism_up_mingw${PRISM_MINGW_BITS:-64}.exe
            source=$(basename "$1" .pl)
            exec "$BINARY" -l -p $PAREA -s $STACK -b $TRAIL -t $TABLE "$BINDIR/bp.out" \
                -g "set_prolog_flag(redefine_builtin,on),set_prolog_flag(stratified_warning,off),compile($source),halt"
            ;;
    esac
fi

case `uname -s` in
    Linux)
        BINARY=$BINDIR/prism_up_linux.bin
        ;;
    Darwin)
        #DARWIN_MAJOR=`uname -r | cut -d. -f 1`
        BINARY=$BINDIR/prism_up_darwin.bin
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

echo "$BINARY -p $PAREA -s $STACK -b $TRAIL -t $TABLE $BINDIR/bp.out -g 'set_prolog_flag(redefine_builtin,on),set_prolog_flag(stratified_warning,off),compile($source),halt'"
exec $BINARY -p $PAREA -s $STACK -b $TRAIL -t $TABLE $BINDIR/bp.out -g "set_prolog_flag(redefine_builtin,on),set_prolog_flag(stratified_warning,off),compile($source),halt"

## For profiling, use below instead of above
#exec $BINARY -p $PAREA -s $STACK -b $TRAIL -t $TABLE $BINDIR/bp.out -g "set_prolog_flag(redefine_builtin,on),set_prolog_flag(stratified_warning,off),profile_compile($source),halt"
