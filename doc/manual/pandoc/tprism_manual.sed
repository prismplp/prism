# sed script applied to tprism_manual.tex before converting it with pandoc
# (used by Make_tprism_html.sh)
#
# - The page layout of the printed manual (the \makeatletter blocks, the
#   column type P, the listing style and the multind indexes) is removed:
#   pandoc cannot parse some of it, and it has no use in HTML.
# - keeptogether only keeps listings on one page.
/^\\makeatletter/,/^\\makeatother/d
/^\\newcolumntype/d
/^\\lstdefinestyle/,/^}/d
/^\\lstnewenvironment/d
/^\\makeindex{/d
/^\\newrefformat/d
/^[[:space:]]*\\begin{keeptogether}/d
/^[[:space:]]*\\end{keeptogether}/d
/\\begin{tabular}/s/P{/p{/g
#
# The two figures side by side in minipages (Section 9.1) become two
# figures: pandoc keeps only the last caption and label of a figure.
/^[[:space:]]*\\begin{minipage}/d
/^[[:space:]]*\\end{minipage}/d
s/^[[:space:]]*\\hfill[[:space:]]*$/\\end{figure}\\begin{figure}/
#
# The listing environments become lstlisting, which pandoc converts into
# code blocks (numbers=left gives line numbers).
s/\\begin{prologcode}\[/\\begin{lstlisting}[language=Prolog,/
s/\\begin{prologcode}/\\begin{lstlisting}[language=Prolog]/
s/\\begin{pythoncode}\[/\\begin{lstlisting}[language=Python,/
s/\\begin{pythoncode}/\\begin{lstlisting}[language=Python]/
s/\\begin{shellcode}/\\begin{lstlisting}/
s/\\end{prologcode}/\\end{lstlisting}/
s/\\end{pythoncode}/\\end{lstlisting}/
s/\\end{shellcode}/\\end{lstlisting}/
#
# pandoc does not know prettyref (see \newrefformat in the preamble).
# References to equations are resolved by pandoc/eqref.lua.
s/\\prettyref{\(chap:[^}]*\)}/Chapter~\\ref{\1}/g
s/\\prettyref{\(sec:[^}]*\)}/Section~\\ref{\1}/g
s/\\prettyref{\(fig:[^}]*\)}/Figure~\\ref{\1}/g
s/\\prettyref{\(tab:[^}]*\)}/Table~\\ref{\1}/g
s/\\prettyref{\(eq:[^}]*\)}/Eq.~(\\ref{\1})/g
#
# The PDF figures are converted to SVG by Make_tprism_html.sh.
s/\(\\includegraphics\(\[[^]]*\]\)\{0,1\}{\)\([^}]*\)\.pdf}/\1fig\/\3.svg}/
