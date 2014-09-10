# -*- Makefile -*-

##----------------------------------------------------------------------

COMPILER    = Compile.sh
TARGETS     = prism.out mpprism.out batch.out
INSTALLDIR  = ../../bin

VERSION     = 2.2a1

PL_CORE  = core/message.pl      \
           core/error.pl        \
           core/random.pl       \
           core/format.pl

PL_TRANS = trans/trans.pl       \
           trans/dump.pl        \
           trans/verify.pl      \
           trans/bpif.pl

PL_UP    = up/dynamic.pl        \
           up/main.pl           \
           up/switch.pl         \
           up/learn.pl          \
           up/vlearn.pl         \
           up/mcmc.pl           \
           up/prob.pl           \
           up/viterbi.pl        \
           up/hindsight.pl      \
           up/expl.pl           \
           up/sample.pl         \
           up/dist.pl           \
           up/list.pl           \
           up/hash.pl           \
           up/flags.pl          \
           up/util.pl           \
           up/feature.pl        \
           up/crflearn.pl       \
           up/crfviterbi.pl     \
           up/crfhindsight.pl   \
           up/fprob.pl          \
           up/bigarray.pl

PL_BAT   = up/batch.pl

PL_MP    = mp/mp_main.pl        \
           mp/mp_learn.pl

PL_BP    = bp/eval.pl

PL_CYCLE = cycle/lin_prob.pl    \
           cycle/nonlin_prob.pl      \
           cycle/trans_graph.pl \
           cycle/find_scc.pl    \
           cycle/solve_graph.pl

PL_UP_ALL   = $(PL_CORE) $(PL_UP) $(PL_TRANS) $(PL_BP) $(PL_CYCLE)
PL_MP_ALL   = $(PL_MP)
PL_BAT_ALL  = $(PL_BAT)
PL_FOC_ALL  = $(PL_FOC)

##----------------------------------------------------------------------

all: $(TARGETS)

install: $(TARGETS)
	cp $(TARGETS) $(INSTALLDIR)

clean:
	rm -f $(TARGETS) prism.pl mpprism.pl batch.pl

prism.pl: $(PL_UP_ALL)
	cat $^ > $@

mpprism.pl: $(PL_MP_ALL)
	cat $^ > $@

batch.pl: $(PL_BAT_ALL)
	cat $^ > $@

%.out: %.pl $(COMPILER)
	sh $(COMPILER) $<

.PHONY: all install clean
