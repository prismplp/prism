#! /bin/bash

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



PackagePrism() {
    sfxs=".exe .32 mp.32 .64 mp.64"
    platforms="win linux macx"
    #packages= .exe .32 mp.32 .64 mp.64 .darwin9 .darwin10
    #platforms="win linux macx"
    set -e

    if [ ! -d $DISTRIB_ROOT ]; then
        Error "config error -- \`$DISTRIB_ROOT' is not a valid directory."
    fi
    tmpdir=`mktemp -d /tmp/prism.XXXXXX`
    cp -pr prism/doc/manual ${tmpdir}/manual
    cd ${tmpdir}/manual
    sh Make.sh > /dev/null
    cd -
    cp -p ${tmpdir}/manual/manual.pdf prism/doc
    rm -rf ${tmpdir}/manual
    rm -rf prism/doc/manual

    for platform in $platforms ; do
        #cp -pr ${tmpdir}/distrib ${tmpdir}/prism
        echo "Platform = " ${platform}
        cp -pr prism ${tmpdir}/prism
        rm -rf ${tmpdir}/prism/bin
        rm -rf ${tmpdir}/prism/tools
        rm -rf ${tmpdir}/prism/testing
        rm -rf ${tmpdir}/prism/src/c/bp4prism/patch
        rm -rf ${tmpdir}/prism/src/prolog/foc
    	mv ${tmpdir}/prism/src/prolog/Makefile.nofoc ${tmpdir}/prism/src/prolog/Makefile

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
                files="*_linux.bin prism upprism mpprism ${files} mpprism.out"
                pkgcmd="tar cfvz"
                pkgsfx=.tar.gz
                ;;
            macx)
                files="*_darwin.bin prism upprism mpprism ${files} mpprism.out"
                pkgcmd="tar cfvz"
                pkgsfx=.tar.gz
                ;;
        esac

        pkgname=${PWD}/`GetPkgName`_${platform}${pkgsfx}

        mkdir ${tmpdir}/prism/bin

        cd ../bin
        cp -p ${files} ${tmpdir}/prism/bin
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
src=../bin
dest=./prism/bin
project=./prism

echo "... exporting git project"
git clone ssh://kameya_lab/root/prism.git ./prism

rm -rf $project/.git

echo "... copying *.bin/*.exe"
cp $src/prism_up_linux.bin $dest/
cp $src/prism_mp_linux.bin $dest/
cp $src/prism_up_darwin.bin $dest/
cp $src/prism_mp_darwin.bin $dest/
cp $src/prism_win32.exe $dest/
cp $src/prism_win64.exe $dest/

echo "... copying *.out "
cp $src/prism.out $dest/
cp $src/foc.out $dest/
cp $src/batch.out $dest/
cp $src/bp.out $dest/
cp $src/mpprism.out $dest/

PackagePrism $1

rm -fr ./prism

