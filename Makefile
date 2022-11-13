# Minimal makefile for pdoc documentation
#

# You can set these variables from the command line.
BINBUILD   = pdoc
PROJ     = ../tprism
BUILDDIR      = ./tprism/

.phony: makefile

%: makefile
	@$(BINBUILD) --math --docformat google ${PROJ} -o "$(BUILDDIR)"
