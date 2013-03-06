#! /bin/bash

############################################################################
##  Configuration                                                         ##
############################################################################

##  locations of the SVN repositories
SYSTEM_REPOS_URI=file:///home/main/pub/prism/repos/prism2
MANUAL_REPOS_URI=file:///home/main/pub/prism/repos/manual
DISTRIB_REPOS_URI=file:///home/main/pub/prism/repos/distrib

##  host for building the 32-bit version(s)
COMPILE_HOST_32=elm

##  host for building the 64-bit version(s)
COMPILE_HOST_64=jasmine

##  host for building the MacOSX version(s)
COMPILE_HOST_D9=cyclamen
COMPILE_HOST_D10=prunus

##  host for installation
INSTALL_HOST=willow

##  destination directory on installation
##  Note: A version number will be appended automatically.
INSTALL_ROOT=/srv/opt/prism



############################################################################
##  Common Routines                                                       ##
############################################################################

Error() {
    echo "`basename $0`: $*" 1>&2
    exit 1
}

GetVersion() {
    grep "^VERSION *=" prism/src/prolog/Makefile | cut -d = -f 2 | tr -d " \r"
}

GetBinName() {
    #echo prism`GetVersion | cut -d - -f 1 | cut -d . -f 1-2 | tr -d .`
    echo prism
}

GetPkgName() {
    echo prism`GetVersion | cut -d - -f 1 | tr -d .`
}



############################################################################
##  Compile the System                                                    ##
############################################################################

CompilePrism() {
    set -e

    if [ -e prism ]; then
        Error "the directory \`prism' already exists. (please remove)"
    fi

    rm -f build.cmd build.sh
    svn export ${SYSTEM_REPOS_URI}/${1-"trunk"} prism

    cat - <<'EOF1' > build.cmd
@echo off

rem - set PATH for nmake
call "%VS100COMNTOOLS%\vsvars32.bat"

rem - compile the system
cd prism\src\c && nmake -f Makefile.nmake install clean
EOF1

    echo "`basename $0`: please run build.cmd on a Windows environment."
    echo "`basename $0`: waiting for the Windows binary..."
    while [ ! -f prism/bin/prism_win32.exe ] ; do sleep 1 ; done

    cat - <<'EOF2' > build.sh
#!/bin/sh
set -e

export PLATFORM=$1
export PROCTYPE=$2

PATH="/opt/mpich/ch-p4/bin:${PATH}"
cd `dirname $0`/prism

make -f Makefile.gmake install clean -C src/c
EOF2

    ssh $COMPILE_HOST_32  sh $PWD/build.sh linux32  up
    ssh $COMPILE_HOST_64  sh $PWD/build.sh linux64  up
    ssh $COMPILE_HOST_D9  sh $PWD/build.sh darwin9  up
    ssh $COMPILE_HOST_D10 sh $PWD/build.sh darwin10 up
    ssh $COMPILE_HOST_32  sh $PWD/build.sh linux32  mp
    ssh $COMPILE_HOST_64  sh $PWD/build.sh linux64  mp

    ssh $COMPILE_HOST_32 make install clean -C $PWD/prism/src/prolog

    mv prism/bin/prism_win32.exe     prism/bin/`GetBinName`.exe
    mv prism/bin/prism_up_linux32.bin   prism/bin/`GetBinName`.32
    mv prism/bin/prism_up_linux64.bin   prism/bin/`GetBinName`.64
    mv prism/bin/prism_mp_linux32.bin   prism/bin/`GetBinName`mp.32
    mv prism/bin/prism_mp_linux64.bin   prism/bin/`GetBinName`mp.64
    mv prism/bin/prism_up_darwin9.bin   prism/bin/`GetBinName`.darwin9
    mv prism/bin/prism_up_darwin10.bin  prism/bin/`GetBinName`.darwin10
    mv prism/src/prolog/Makefile.nofoc prism/src/prolog/Makefile
    rm -rf prism/src/c/bp4prism/patch

    tar cfvz prism.tar.gz prism
}



############################################################################
##  Install the System                                                    ##
############################################################################

InstallPrism() {
    set -e

    if [ `hostname -s` != $INSTALL_HOST ]; then
        Error "this script needs to be run on $INSTALL_HOST."
    fi

    if [ ! -d $INSTALL_ROOT ]; then
        Error "config error -- \`$INSTALL_ROOT' is not a valid directory."
    fi

    if [ `id -u` -ne 0 ]; then
        Error "this script must be run as root."
    fi

    for sfx in .exe .32 mp.32 .64 mp.64 .darwin9 .darwin10 ; do
        binary=prism/bin/`GetBinName`${sfx}
        if [ ! -f $binary ]; then
            Error "\`$binary' is not found. (build not complete?)"
        fi
    done

    destdir=$INSTALL_ROOT/`GetVersion | tr -d -`

    if [ -e $destdir ]; then
        Error "\`$destdir' already exists. (incorrect version number?)"
    fi

    (
        set -x
        cp -pr prism ${destdir}
        chmod 0750 ${destdir}/tools
        chmod 0750 ${destdir}/src/prolog/foc
        chown -R root:root ${destdir}
        chgrp sato ${destdir}/tools
        chgrp sato ${destdir}/src/prolog/foc

        rm -rf ${destdir}/src/c/bp4prism/patch
    )
}



############################################################################
##  Package the System for the Binary Distribution                        ##
############################################################################

PackagePrism() {
    set -e

    if [ ! -d $DISTRIB_ROOT ]; then
        Error "config error -- \`$DISTRIB_ROOT' is not a valid directory."
    fi

    for sfx in .exe .32 mp.32 .64 mp.64 .darwin9 .darwin10 ; do
        binary=prism/bin/`GetBinName`${sfx}
        if [ ! -f $binary ]; then
            Error "\`$binary' is not found. (build not complete?)"
        fi
    done

    tmpdir=`mktemp -d /tmp/prism.XXXXXX`

    #svn export ${MANUAL_REPOS_URI}/trunk ${tmpdir}/manual
    #svn export ${DISTRIB_REPOS_URI}/trunk ${tmpdir}/distrib
    #
    #docfile=`GetPkgName`.pdf
    #
    #cd ${tmpdir}/manual
    #sh Make.sh > /dev/null
    #cp -p manual.pdf ../distrib/doc/${docfile}
    #cd -

    cp -pr prism/doc/manual ${tmpdir}/manual
    cd ${tmpdir}/manual
    sh Make.sh > /dev/null
    cd -
    cp -p ${tmpdir}/manual/manual.pdf prism/doc
    rm -rf ${tmpdir}/manual
    rm -rf prism/doc/manual

    for platform in win linux macx ; do
        #cp -pr ${tmpdir}/distrib ${tmpdir}/prism

        cp -pr prism ${tmpdir}/prism
        rm -rf ${tmpdir}/prism/bin
        rm -rf ${tmpdir}/prism/tools
        rm -rf ${tmpdir}/prism/testing
        rm -rf ${tmpdir}/prism/src/c/bp4prism/patch
        rm -rf ${tmpdir}/prism/src/prolog/foc

        case $platform in
	    win)
		rm -f ${tmpdir}/prism/src/c/bp4prism/lib/bp4prism-darwin*.a
		rm -f ${tmpdir}/prism/src/c/bp4prism/lib/bp4prism-linux*.a
		;;
	    linux)
		rm -f ${tmpdir}/prism/src/c/bp4prism/lib/bp4prism-darwin*.a
		rm -f ${tmpdir}/prism/src/c/bp4prism/lib/bp4prism-cygwin.a
		rm -f ${tmpdir}/prism/src/c/bp4prism/lib/bp4prism-win32.lib
		;;
	    macx)
		rm -f ${tmpdir}/prism/src/c/bp4prism/lib/bp4prism-linux*.a
		rm -f ${tmpdir}/prism/src/c/bp4prism/lib/bp4prism-cygwin.a
		rm -f ${tmpdir}/prism/src/c/bp4prism/lib/bp4prism-win32.lib
		;;
	esac

        files="bp.out prism.out foc.out batch.out"

        case $platform in
            win)
                files="*.exe *.bat ${files}"
                pkgcmd="zip -r"
                pkgsfx=.zip
                ;;
            linux)
                files="*.32 *.64 prism upprism mpprism ${files} mpprism.out"
                pkgcmd="tar cfvz"
                pkgsfx=.tar.gz
                ;;
            macx)
                files="*.darwin* prism upprism mpprism ${files} mpprism.out"
                pkgcmd="tar cfvz"
                pkgsfx=.tar.gz
                ;;
        esac

        pkgname=${PWD}/`GetPkgName`_${platform}${pkgsfx}

        mkdir ${tmpdir}/prism/bin

        cd prism/bin
        cp -p ${files} ${tmpdir}/prism/bin
        cd -

        cd ${tmpdir}
        if [ -f prism/bin/`GetBinName`.exe ]; then
          mv prism/bin/`GetBinName`.exe prism/bin/prism_win32.exe
        fi
        if [ -f prism/bin/`GetBinName`.32  ]; then
          mv prism/bin/`GetBinName`.32 prism/bin/prism_up_linux32.bin
        fi
        if [ -f prism/bin/`GetBinName`.64  ]; then
          mv prism/bin/`GetBinName`.64 prism/bin/prism_up_linux64.bin
        fi
        if [ -f prism/bin/`GetBinName`mp.32  ]; then
          mv prism/bin/`GetBinName`mp.32 prism/bin/prism_mp_linux32.bin
        fi
        if [ -f prism/bin/`GetBinName`mp.64  ]; then
          mv prism/bin/`GetBinName`mp.64 prism/bin/prism_mp_linux64.bin
        fi
        if [ -f prism/bin/`GetBinName`.darwin9 ]; then
          mv prism/bin/`GetBinName`.darwin9 prism/bin/prism_up_darwin9.bin
        fi
        if [ -f prism/bin/`GetBinName`.darwin10 ]; then
          mv prism/bin/`GetBinName`.darwin10 prism/bin/prism_up_darwin10.bin
        fi
        if [ -f prism/bin/prism ]; then
          chmod a+x prism/bin/prism
        fi
        if [ -f prism/bin/upprism ]; then
          chmod a+x prism/bin/upprism
        fi
        if [ -f prism/bin/mpprism ]; then
          chmod a+x prism/bin/mpprism
        fi
        cd -

        cd ${tmpdir}
        rm -f ${pkgname} ; ${pkgcmd} ${pkgname} prism ; rm -fr prism
        cd -
    done

    rm -fr ${tmpdir}
}



############################################################################
##  Main Routine                                                          ##
############################################################################

case $1 in
    compile)
        CompilePrism $2
        ;;
    install)
        InstallPrism $2
        ;;
    package)
        PackagePrism $2
        ;;
    *)
        Error "Usage: `basename $0` {compile|install|package}"
        ;;
esac
