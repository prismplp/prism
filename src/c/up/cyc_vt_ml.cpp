#define CXX_COMPILE 
extern "C" {
#include "bprolog.h"
#include "up/up.h"
#include "up/em_aux.h"
#include "up/em_aux_ml.h"
#include "up/vt.h"
#include "up/vt_aux_ml.h"
#include "up/vt_ml.h"
#include "up/flags.h"
#include "up/util.h"
#include "up/viterbi.h"
#include "up/nonlinear_eq.h"


#include "bprolog.h"
#include "core/random.h"
#include "core/gamma.h"
#include "up/graph.h"
#include "up/scc.h"
}

extern "C"
int run_cyc_vt(VT_ENG_PTR vt_ptr) {
	int     r, iterate, old_valid, converged, saved = 0;
	double  likelihood, log_prior;
	double  lambda, old_lambda = 0.0;

	config_vt(vt_ptr);

	double start_time=getCPUTime();
    init_scc();
	double scc_time=getCPUTime();
	
	for (r = 0; r < num_restart; r++) {
		SHOW_PROGRESS_HEAD("#vt-iters", r);

		initialize_params();
		itemp = 1.0;
		iterate = 0;

		old_valid = 0;

		while (1) {
			/*compute_max();*/
			compute_nonlinear_viterbi(0);
			count_occ_sws();

			RET_ON_ERR(examine_likelihood());
			likelihood = compute_vt_likelihood();
			log_prior  = vt_ptr->smooth ? vt_ptr->compute_log_prior() : 0.0;
			lambda = likelihood + log_prior;

			if (!isfinite(lambda)) {
				emit_internal_error("invalid log likelihood or log post: %s (at iteration #%d)",
				                    isnan(lambda) ? "NaN" : "infinity", iterate);
				RET_ERR(ierr_invalid_likelihood);
			}
			if (old_valid && old_lambda - lambda > prism_epsilon) {
				emit_error("log likelihood or log post decreased [old: %.9f, new: %.9f] (at iteration #%d)",
				           old_lambda, lambda, iterate);
				RET_ERR(err_invalid_likelihood);
			}
			if (itemp == 1.0 && likelihood > 0.0) {
				emit_error("log likelihood greater than zero [value: %.9f] (at iteration #%d)",
				           likelihood, iterate);
				RET_ERR(err_invalid_likelihood);
			}

			converged = (old_valid && lambda - old_lambda <= prism_epsilon);
			if (converged || REACHED_MAX_ITERATE(iterate)) {
				break;
			}

			old_lambda = lambda;
			old_valid  = 1;

			SHOW_PROGRESS(iterate);
			RET_ON_ERR(vt_ptr->update_params());
			iterate++;
		}

		SHOW_PROGRESS_TAIL(converged, iterate, lambda);

		if (r == 0 || lambda > vt_ptr->lambda) {
			vt_ptr->lambda     = lambda;
			vt_ptr->likelihood = likelihood;
			vt_ptr->iterate    = iterate;

			saved = (r < num_restart - 1);
			if (saved) {
				save_params();
			}
		}
	}


	double solution_time=getCPUTime();
	//free data
	free_scc();
	if (saved) {
		restore_params();
	}

	if(scc_debug_level>=1) {
		printf("CPU time (scc,solution,all)\n");
		printf("# %f,%f,%f\n",scc_time-start_time,solution_time-scc_time,solution_time - start_time);
	}
	return BP_TRUE;
}
