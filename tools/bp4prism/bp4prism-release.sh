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

for file in bp4prism.headers bp4prism.libs
do
    if [ ! -f ${dir}/${file} ] ; then
        Error "system file not found in $dir -- $file"
    fi
done

if [ ! -d Emulator ] ; then
    Error "directory not found -- Emulator"
fi

if [ ! -d Emulator.prism ] ; then
    Error "directory not found -- Emulator.prism"
fi

########################################################################

echo -------- Step 1: Create the target directory

rm -fr Emulator.prism.tmp

echo copying files in Emulator.prism into Emulator.prism.tmp
cp -pr Emulator.prism Emulator.prism.tmp

########################################################################

echo -------- Step 2: Apply bp4prism-release.patch

patch -d Emulator.prism.tmp   \
      --no-backup-if-mismatch -p1 < ${dir}/bp4prism-release.patch

########################################################################

echo -------- Step 3: Extract headers and libraries

rel_dir=Emulator.prism.release
rm -fr $rel_dir
mkdir $rel_dir
mkdir $rel_dir/include
mkdir $rel_dir/lib

for file in `cat ${dir}/bp4prism.headers`; do
    if [ ! -f Emulator.prism.tmp/${file} ] ; then
        Error "header file not found in Emulator -- $file"
    fi

    echo copying file `basename $file`
    nkf -Lu < Emulator.prism.tmp/${file} > ${rel_dir}/include/${file}
done

for file in `cat ${dir}/bp4prism.libs`; do
    if [ ! -f Emulator.prism.tmp/${file} ] ; then
        Error "library file not found in Emulator -- $file"
    fi

    echo copying file `basename $file`
    cp -p Emulator.prism.tmp/${file} ${rel_dir}/lib
done

rm -rf Emulator.prism.tmp

########################################################################

echo -------- Completed.
