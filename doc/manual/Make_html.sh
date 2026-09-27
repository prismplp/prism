#! /bin/sh
##
## Converts the PRISM user's manual (manual.tex) into a single HTML page
## with pandoc; the formulas are rendered by MathJax.
##
##   sh Make_html.sh [OUTPUT_DIR]
##
## OUTPUT_DIR defaults to bin/tprism_docs/prism, which is published on
## GitHub Pages by .github/workflows/gh-pages.yml.  The page is written to
## OUTPUT_DIR/index.html, and the figures to OUTPUT_DIR/fig/*.svg.
##
## Requirements: pandoc >= 3.11 (the version used by the CI; older ones
## lack --math-method, and some embed polyfill.io), ghostscript and
## pdftocairo (poppler-utils) for converting the EPS figures.
##

dir=`cd \`dirname $0\` && pwd`
out=${1-$dir/../../bin/tprism_docs/prism}

if ! pandoc --help 2>/dev/null | grep -q -- '--math-method'; then
    echo "${0##*/}: pandoc >= 3.11 is required (found: `pandoc --version 2>/dev/null | head -1`)" 1>&2
    exit 1
fi

set -e
mkdir -p "$out/fig"
out=`cd "$out" && pwd`
tmp=`mktemp -d`
trap 'rm -rf "$tmp"' EXIT

##  figures: EPS -> PDF -> SVG
for eps in "$dir"/fig/*.eps; do
    name=`basename "$eps" .eps`
    gs -q -dSAFER -dBATCH -dNOPAUSE -dEPSCrop -sDEVICE=pdfwrite -sOutputFile="$tmp/$name.pdf" "$eps"
    pdftocairo -svg "$tmp/$name.pdf" "$out/fig/$name.svg"
done

##  MathJax 3 with the bundled fonts: MathJax 4 (pandoc's default) took
##  ~30 times longer to typeset the ~2200 formulas of the page, and its
##  lazy typesetting cannot resolve forward references to equations.
mathjax=https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml-full.js

##  LaTeX constructs that pandoc cannot handle (see pandoc/manual.sed)
sed -f "$dir/pandoc/manual.sed" "$dir/manual.tex" > "$tmp/manual.tex"

pandoc "$tmp/manual.tex" \
    --from=latex --to=html5 --standalone --wrap=none \
    --toc --toc-depth=2 --number-sections \
    --math-method=mathjax:"$mathjax" \
    --lua-filter="$dir/pandoc/eqref.lua" \
    --citeproc --bibliography="$dir/manual.bib" \
    --metadata=pagetitle="PRISM User's Manual" \
    --metadata=toc-title=Contents \
    --metadata=reference-section-title=Bibliography \
    --variable=maxwidth=50em \
    --include-in-header="$dir/pandoc/header.html" \
    --include-before-body="$dir/pandoc/before-body.html" \
    --output="$out/index.html"

echo "[SAVE] $out/index.html"
