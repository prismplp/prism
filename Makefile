# Minimal makefile for pdoc documentation
#

# You can set these variables from the command line.
BINBUILD   = pdoc
PROJ     = ../tprism
BUILDDIR      = ./tprism/

# PRISM user's manual (HTML) converted from doc/manual/manual.tex into ./prism/;
# needs pandoc, ghostscript and pdftocairo (see doc/manual/Make_html.sh)
.PHONY: prism
prism:
	@sh ../../doc/manual/Make_html.sh prism

# Makefile must be phony: otherwise `%: Makefile` also matches the makefile
# itself, and pdoc runs three times (with a "Circular makefile" warning)
.PHONY: Makefile

%: Makefile
	@$(BINBUILD) --math --docformat google ${PROJ} -o "$(BUILDDIR)"
