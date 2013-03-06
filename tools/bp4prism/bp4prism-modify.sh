#! /bin/sh

set -e

########################################################################

Error()
{
    echo "`basename $0`: $@" 1>&2
    exit 1
}

########################################################################

dir=`dirname $0`

for file in \
    bp4prism.files bp4prism.patch
do
    if [ ! -f ${dir}/${file} ] ; then
        Error "system file not found in $dir -- $file"
    fi
done

if [ ! -d Emulator ] ; then
    Error "directory not found -- Emulator"
fi

rm -fr Emulator.prism

########################################################################

echo -------- Step 1: Create the target directory

mkdir Emulator.prism

for file in `cat ${dir}/bp4prism.files` ; do
    if [ ! -f Emulator/${file} ] ; then
        Error "source file not found in Emulator -- $file"
    fi

    echo copying file `basename $file`
    cp -p Emulator/${file} Emulator.prism/
done

########################################################################

echo -------- Step 2: Apply bp4prism.patch

patch -d Emulator.prism --no-backup-if-mismatch -p1 < ${dir}/bp4prism.patch

########################################################################

echo -------- Step 3: Copy Makefiles

echo copying Makefiles
cp -p ${dir}/Makefiles/* Emulator.prism/

########################################################################

echo -------- Completed.
