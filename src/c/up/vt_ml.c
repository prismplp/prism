/* -*- c-basic-offset: 4 ; tab-width: 4 -*- */

/*------------------------------------------------------------------------*/

#include "bprolog.h"
#include "up/up.h"
#include "up/em_aux.h"
#include "up/em_aux_ml.h"
#include "up/vt.h"
#include "up/vt_aux_ml.h"
#include "up/flags.h"
#include "up/util.h"
#include "up/viterbi.h"

/*------------------------------------------------------------------------*/

void config_vt(VT_ENG_PTR vt_ptr)
{
    if (log_scale) {
        vt_ptr->compute_log_prior   = compute_log_prior;
        vt_ptr->update_params       = vt_ptr->smooth ? update_vt_params_smooth : update_vt_params;
    }
    else {
        vt_ptr->compute_log_prior   = compute_log_prior;
        vt_ptr->update_params       = vt_ptr->smooth ? update_vt_params_smooth : update_vt_params;
    }
}

/*------------------------------------------------------------------------*/

int run_vt(VT_ENG_PTR vt_ptr)
{
    int     r, iterate, old_valid, converged, saved = 0;
    double  likelihood, log_prior;
    double  lambda, old_lambda = 0.0;

    config_vt(vt_ptr);

    for (r = 0; r < num_restart; r++) {
        SHOW_PROGRESS_HEAD("#vt-iters", r);

        initialize_params();
        itemp = 1.0;
        iterate = 0;

        old_valid = 0;

        while (1) {
            compute_max();
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

    if (saved) {
        restore_params();
    }

    return BP_TRUE;
}
