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

#include "eigen/Core"
#include "eigen/LU"

#include <iostream>
#include <set>
#include <cmath>


using namespace Eigen;


int run_cyc_em(struct EM_Engine* em_ptr) {
	int	 r, iterate, old_valid, converged, saved = 0;
	double  likelihood, log_prior=0;
	double  lambda, old_lambda = 0.0;

	//config_em(em_ptr);
	double start_time=getCPUTime();
	init_scc();
	double scc_time=getCPUTime();
	//start EM
	double itemp = 1.0;
	for (r = 0; r < num_restart; r++) {
		SHOW_PROGRESS_HEAD("#cyc-em-iters", r);
		initialize_params();
		iterate = 0;
		while (1) {
			old_valid = 0;
			while (1) {
				if (CTRLC_PRESSED) {
					SHOW_PROGRESS_INTR();
					RET_ERR(err_ctrl_c_pressed);
				}

				
				//RET_ON_ERR(em_ptr->compute_inside());
				compute_inside_linear();
				//RET_ON_ERR(em_ptr->examine_inside());
				//examine_inside_linear_cycle();
				//likelihood = em_ptr->compute_likelihood();
				likelihood=compute_likelihood_scaling_none();
				log_prior  = em_ptr->smooth ? em_ptr->compute_log_prior() : 0.0;
				lambda = likelihood + log_prior;
				if (verb_em) {
					if (em_ptr->smooth) {
						prism_printf("iteration #%d:\tlog_likelihood=%.9f\tlog_prior=%.9f\tlog_post=%.9f\n", iterate, likelihood, log_prior, lambda);
					}else {
						prism_printf("iteration #%d:\tlog_likelihood=%.9f\n", iterate, likelihood);
						if(scc_debug_level>=4) {
							print_eq();
						}
					}
				}

				if (!std::isfinite(lambda)) {
					emit_internal_error("invalid log likelihood or log post: %s (at iteration #%d)",
							std::isnan(lambda) ? "NaN" : "infinity", iterate);
					RET_ERR(ierr_invalid_likelihood);
				}
				if (old_valid && old_lambda - lambda > prism_epsilon) {
					emit_error("log likelihood or log post decreased [old: %.9f, new: %.9f] (at iteration #%d)",
							old_lambda, lambda, iterate);
					RET_ERR(err_invalid_likelihood);
				}

				converged = (old_valid && lambda - old_lambda <= prism_epsilon);
				if (converged || REACHED_MAX_ITERATE(iterate)) {
					break;
				}

				old_lambda = lambda;
				old_valid  = 1;

				//RET_ON_ERR(em_ptr->compute_expectation());
				compute_expectation_linear();

				SHOW_PROGRESS(iterate);
				RET_ON_ERR(em_ptr->update_params());
				//update_params();
				iterate++;
			}

			/* [21 Aug 2007, by yuizumi]
			 * Note that 1.0 can be represented exactly in IEEE 754.
			 */
			if (itemp == 1.0) {
				break;
			}
			itemp *= itemp_rate;
			if (itemp >= 1.0) {
				itemp = 1.0;
			}
	
		}

		SHOW_PROGRESS_TAIL(converged, iterate, lambda);

		if (r == 0 || lambda > em_ptr->lambda) {
			em_ptr->lambda     = lambda;
			em_ptr->likelihood = likelihood;
			em_ptr->iterate    = iterate;

			saved = (r < num_restart - 1);
			if (saved) {
				save_params();
			}
		}
	}
	if (saved) {
		restore_params();
	}
	//END EM

	if(scc_debug_level>=1) {
		print_sccs_statistics();
	}
	double solution_time=getCPUTime();
	//free data
	free_scc();
	if(scc_debug_level>=1) {
		printf("CPU time (scc,solution,all)\n");
		printf("# %f,%f,%f\n",scc_time-start_time,solution_time-scc_time,solution_time - start_time);
	}

	em_ptr->bic = compute_bic(em_ptr->likelihood);
	em_ptr->cs  = em_ptr->smooth ? compute_cs(em_ptr->likelihood) : 0.0;
	return BP_TRUE;
}

void config_cyc_em(EM_ENG_PTR em_ptr) {
	if (log_scale) {
		em_ptr->compute_inside      = daem ? compute_daem_inside_scaling_log_exp : compute_inside_scaling_log_exp;
		em_ptr->examine_inside      = examine_inside_scaling_log_exp;
		em_ptr->compute_expectation = compute_expectation_scaling_log_exp;
		em_ptr->compute_likelihood  = compute_likelihood_scaling_log_exp;
		em_ptr->compute_log_prior   = daem ? compute_daem_log_prior : compute_log_prior;
		em_ptr->update_params       = em_ptr->smooth ? update_params_smooth : update_params;
	} else {
		em_ptr->compute_inside      = daem ? compute_daem_inside_scaling_none : compute_inside_scaling_none;
		em_ptr->examine_inside      = examine_inside_scaling_none;
		em_ptr->compute_expectation = compute_expectation_scaling_none;
		em_ptr->compute_likelihood  = compute_likelihood_scaling_none;
		em_ptr->compute_log_prior   = daem ? compute_daem_log_prior : compute_log_prior;
		em_ptr->update_params       = em_ptr->smooth ? update_params_smooth : update_params;
	}
}

extern "C"
int pc_linear_eq_2(void) {
	int i;
	scc_debug_level = bpx_get_integer(bpx_get_call_arg(1,2));
	double start_time=getCPUTime();
	init_scc();
	double scc_time=getCPUTime();
	//solution
	if(scc_debug_level>=2) {
		printf("linear solver: LU decomposition\n");
	}
	for(i=0; i<scc_num; i++) {
		if(sccs[i].size==1&&sccs[i].order==0){
			update_inside_scc(i);
		}else{
			solve_linear_scc(i);
		}
		if(scc_debug_level>=2) {
			int n=sccs[i].size;
			int j;
			for(j=0; j<n; j++) {
				int w=sccs[i].el[j];
				printf("%d:%f\n",w,sorted_expl_graph[w]->inside);
			}
		}
	}
	double solution_time=getCPUTime();
	//free data
	free_scc();
	double prob=sorted_expl_graph[sorted_egraph_size-1]->inside;
	if(scc_debug_level>=1) {
		printf("CPU time (scc,solution,all)\n");
		printf("# %f,%f,%f\n",scc_time-start_time,solution_time-scc_time,solution_time - start_time);
	}
	return bpx_unify(bpx_get_call_arg(2,2),
			bpx_build_float(prob));

}


extern "C"
int pc_cyc_em_7(void) {
	struct EM_Engine em_eng;
	RET_ON_ERR(check_smooth(&em_eng.smooth));
	config_cyc_em(&em_eng);
	//scc_debug_level = bpx_get_integer(bpx_get_call_arg(7,7));
	run_cyc_em(&em_eng);
	return
	    bpx_unify(bpx_get_call_arg(1,7), bpx_build_integer(em_eng.iterate   )) &&
	    bpx_unify(bpx_get_call_arg(2,7), bpx_build_float  (em_eng.lambda    )) &&
	    bpx_unify(bpx_get_call_arg(3,7), bpx_build_float  (em_eng.likelihood)) &&
	    bpx_unify(bpx_get_call_arg(4,7), bpx_build_float  (em_eng.bic       )) &&
	    bpx_unify(bpx_get_call_arg(5,7), bpx_build_float  (em_eng.cs        )) &&
	    bpx_unify(bpx_get_call_arg(6,7), bpx_build_integer(em_eng.smooth    )) ;
}

