#ifndef MCMC_H
#define MCMC_H

#include <math.h>
#include <stdio.h>
#include "bprolog.h"
#include "up/em_aux.h"
#include "up/em_aux_ml.h"
#include "up/em_aux_vb.h"
#include "up/up.h"
#include "up/graph.h"
#include "core/idtable.h"
#include "core/idtable_preds.h"
#include "core/random.h"
#include "core/gamma.h"
#include "up/viterbi.h"
#include "up/flags.h"
#include "up/util.h"

/*------------------------------------------------------------------------*/

typedef struct SwitchCount *SW_COUNT_PTR;
struct SwitchCount {
	int sw_ins_id;
	int count;
};

typedef struct PredictAns *PR_ANS_PTR;
struct PredictAns {
	int rank;
	double logP;
	struct PredictAns *next;
};

/*------------------------------------------------------------------------*/

#define SHOW_MCMC_PROGRESS(n)                                           \
    do {                                                                \
        if(mcmc_message > 0 && (n) % mcmc_progress == 0) {              \
            if((n) % (mcmc_progress * 10) == 0)                         \
                prism_printf("%d", n);                                  \
            else                                                        \
                prism_printf(".");                                      \
        }                                                               \
    } while (0)

#define SHOW_MCMC_PROGRESS_HEAD(str)                                    \
    do {                                                                \
        if(mcmc_message > 0)                                            \
            prism_printf("%s: ", str);                                  \
    } while (0)

#define SHOW_MCMC_PROGRESS_TAIL(n)                                      \
    do {                                                                \
        if(mcmc_message > 0)                                            \
            prism_printf("(%d) (Stopped)\n", n);                        \
    } while (0)

#define SHOW_MCMC_PROGRESS_INTR()                                       \
    do {                                                                \
        if(mcmc_message > 0)                                            \
            prism_printf("(Interrupted)\n");                            \
    } while (0)

#endif
