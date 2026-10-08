#! /bin/sh
##
## Converts the T-PRISM User's Manual (tprism_manual.tex) into a single
## HTML page with pandoc; the formulas are rendered by MathJax.
##
##   sh Make_tprism_html.sh [OUTPUT_DIR]
##
## OUTPUT_DIR defaults to bin/tprism_docs/tprism_manual, which is published
## on GitHub Pages by .github/workflows/gh-pages.yml.  The page is written
## to OUTPUT_DIR/index.html, and the figures to OUTPUT_DIR/fig/*.svg.
##
## Requirements: pandoc >= 3.11 and pdftocairo (poppler-utils) for
## converting the PDF figures (see also Make_html.sh).
##

dir=`cd \`dirname $0\` && pwd`
out=${1-$dir/../../bin/tprism_docs/tprism_manual}

if ! pandoc --help 2>/dev/null | grep -q -- '--math-method'; then
    echo "${0##*/}: pandoc >= 3.11 is required (found: `pandoc --version 2>/dev/null | head -1`)" 1>&2
    exit 1
fi

set -e
mkdir -p "$out/fig"
out=`cd "$out" && pwd`
tmp=`mktemp -d`
trap 'rm -rf "$tmp"' EXIT

##  figures: PDF -> SVG
for fig in `sed -n 's/.*\\\\includegraphics\(\[[^]]*\]\)\{0,1\}{\([^}]*\)\.pdf}.*/\2/p' "$dir/tprism_manual.tex"`; do
    pdftocairo -svg "$dir/$fig.pdf" "$out/fig/`basename $fig`.svg"
done

##  MathJax 3, as in Make_html.sh
mathjax=https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml-full.js

##  LaTeX constructs that pandoc cannot handle (see pandoc/tprism_manual.sed);
##  the source has CRLF line endings
tr -d '\r' < "$dir/tprism_manual.tex" | sed -f "$dir/pandoc/tprism_manual.sed" > "$tmp/tprism_manual.tex"

pandoc "$tmp/tprism_manual.tex" \
    --from=latex --to=html5 --standalone --wrap=none \
    --toc --toc-depth=2 --number-sections \
    --math-method=mathjax:"$mathjax" \
    --lua-filter="$dir/pandoc/eqref.lua" \
    --lua-filter="$dir/pandoc/tprism_manual.lua" \
    --citeproc --bibliography="$dir/tprism.bib" \
    --metadata=pagetitle="T-PRISM User's Manual" \
    --metadata=toc-title=Contents \
    --metadata=reference-section-title=Bibliography \
    --variable=maxwidth=50em \
    --include-in-header="$dir/pandoc/header.html" \
    --include-before-body="$dir/pandoc/before-body.html" \
    --output="$out/index.html"

echo "[SAVE] $out/index.html"
