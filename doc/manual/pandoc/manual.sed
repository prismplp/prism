# sed script applied to manual.tex before converting it with pandoc
# (used by Make_html.sh)
#
# - The hacks for multind/hyperref make pandoc's macro expansion loop
#   (\tt is redefined in terms of itself), so they are replaced by
#   definitions that drop the index entries (\index{<index>}{<entry>}),
#   the indexes and the bookmarks, which have no use in HTML.
#   \index expands to an empty group rather than nothing: otherwise a line
#   holding only index entries (e.g. \conindex{...}) becomes a blank line
#   for pandoc and splits the paragraph.
# - pandoc's \centerline accepts only inline contents, while the manual
#   also centers tables with it.
/Some hacks by yuizumi (BGN)/,/Some hacks by yuizumi (END)/c\
\\renewcommand{\\index}[2]{{}}\
\\renewcommand{\\printindex}[2]{}\
\\newcommand{\\startindexes}[1]{}\
\\newcommand{\\myaddcontentsline}[3]{}\
\\renewcommand{\\centerline}[1]{#1}
#
# MathJax does not support \hspace* (in HTML it is the same as \hspace).
s/\\hspace\*{/\\hspace{/g
#
# The figures are converted from EPS to SVG by Make_html.sh.
s/\(\\includegraphics\(\[[^]]*\]\)\{0,1\}{[^}]*\)\.eps}/\1.svg}/
