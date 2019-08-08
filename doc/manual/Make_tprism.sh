#! /bin/sh

main=tprism_manual

pdflatex ${main}
bibtex ${main}
pdflatex ${main}
pdflatex ${main}

