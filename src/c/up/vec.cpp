#define CXX_COMPILE 

#ifdef _MSC_VER
#include <windows.h>
#endif

extern "C" {
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "up/up.h"
#include "up/flags.h"
#include "bprolog.h"
#include "core/random.h"
#include "core/gamma.h"
#include "up/graph.h"
#include "up/util.h"
#include "up/em.h"
#include "up/em_aux.h"
#include "up/em_aux_ml.h"
#include "up/viterbi.h"
#include "up/graph_aux.h"
#include "up/nonlinear_eq.h"
#include "up/scc.h"
}

#include <iostream>
#include <set>
#include <cmath>



int run_cyc_vec(struct EM_Engine* em_ptr) {
	int	 r, iterate, old_valid, converged, saved = 0;
	double  likelihood, log_prior=0;
	double  lambda, old_lambda = 0.0;

	//config_em(em_ptr);
	double start_time=getCPUTime();
	init_scc();
	double scc_time=getCPUTime();
	//start EM
	double itemp = 1.0;
	initialize_params();
	ompute_inside_linear();
	print_eq();
	//compute_expectation_linear();
	save_params();
	if (saved) {
		restore_params();
	}
	//END EM
	print_sccs_statistics();
	double solution_time=getCPUTime();
	//free data
	free_scc();
	if(scc_debug_level>=1) {
		printf("CPU time (scc,solution,all)\n");
		printf("# %f,%f,%f\n",scc_time-start_time,solution_time-scc_time,solution_time - start_time);
	}

	em_ptr->lambda     = lambda;
	em_ptr->likelihood = likelihood;
	em_ptr->iterate    = iterate;
	em_ptr->bic = compute_bic(em_ptr->likelihood);
	em_ptr->cs  = em_ptr->smooth ? compute_cs(em_ptr->likelihood) : 0.0;
	return BP_TRUE;
}

extern "C"
int pc_prism_vec_7(void) {
	struct EM_Engine em_eng;
	RET_ON_ERR(check_smooth(&em_eng.smooth));
	//scc_debug_level = bpx_get_integer(bpx_get_call_arg(7,7));
	run_cyc_vec(&em_eng);
	return
	    bpx_unify(bpx_get_call_arg(1,7), bpx_build_integer(em_eng.iterate   )) &&
	    bpx_unify(bpx_get_call_arg(2,7), bpx_build_float  (em_eng.lambda    )) &&
	    bpx_unify(bpx_get_call_arg(3,7), bpx_build_float  (em_eng.likelihood)) &&
	    bpx_unify(bpx_get_call_arg(4,7), bpx_build_float  (em_eng.bic       )) &&
	    bpx_unify(bpx_get_call_arg(5,7), bpx_build_float  (em_eng.cs        )) &&
	    bpx_unify(bpx_get_call_arg(6,7), bpx_build_integer(em_eng.smooth    )) ;
}

