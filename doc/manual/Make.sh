#! /bin/sh

main=manual
indexes="concept predicate example"

case ${1-"normal"} in
    normal)
        dvipdfmx="dvipdfmx"
        ;;
    w32tex)
        dvipdfmx="dvipdfmx -f msfonts.map"
        ;;
    *)
        echo "${0##*/}: unknown compile mode -- $1" 1>&2 ; exit 1
        ;;
esac

##--------------------------------
##  prepare

rmfiles=""

rmfiles="${rmfiles} ${main}.aux"
rmfiles="${rmfiles} ${main}.bbl"
rmfiles="${rmfiles} ${main}.blg"
rmfiles="${rmfiles} ${main}.dvi"
rmfiles="${rmfiles} ${main}.log"
rmfiles="${rmfiles} ${main}.out"
rmfiles="${rmfiles} ${main}.toc"

for index in $indexes
do
    rmfiles="${rmfiles} ${index}.idx ${index}.ilg ${index}.ind"
done

##--------------------------------
##  compile

set -ex

rm -f $rmfiles

latex $main
bibtex $main

for index in $indexes
do
    makeindex -t ${index}.ilg < ${index}.idx | sed 's/\\_ /\\_/g' > ${index}.ind
done
 
latex $main
latex $main
dvips -q $main 2> /dev/null
$dvipdfmx -q $main

rm -f $rmfiles
