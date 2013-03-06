/* -*- c-basic-offset: 4 ; tab-width: 4 -*- */

/*------------------------------------------------------------------------*/

#include "bprolog.h"
#include "up/up.h"
#include "up/em_aux.h"
#include "up/em_aux_vb.h"
#include "up/vt.h"
#include "up/vt_aux_ml.h"
#include "up/vt_aux_vb.h"
#include "up/flags.h"
#include "up/util.h"

/*------------------------------------------------------------------------*/

void config_vbvt(VBVT_ENG_PTR vbvt_ptr)
{
    if(log_scale) {
        vbvt_ptr->compute_pi = compute_pi_scaling_log_exp;
        vbvt_ptr->compute_free_energy_l1 = compute_vbvt_free_energy_l1_scaling_log_exp;
    }
    else {
        vbvt_ptr->compute_pi = compute_pi_scaling_none;
        vbvt_ptr->compute_free_energy_l1 = compute_vbvt_free_energy_l1_scaling_none;
    }
}

/*------------------------------------------------------------------------*/

int run_vbvt(VBVT_ENG_PTR vbvt_ptr)
{
    int r,iterate,old_valid,converged,saved = 0;
    double l0,l1,free_energy,old_free_energy = 0.0;;

    config_vbvt(vbvt_ptr);

    for(r = 0; r < num_restart; r++) {
        SHOW_PROGRESS_HEAD("#vbvt-iters",r);

        initialize_hyperparams();
        itemp = 1.0;
        iterate = 0;

        old_valid = 0;

        while(1) {
            RET_ON_ERR(vbvt_ptr->compute_pi());
            compute_max_pi();
            count_occ_sws();

            /* compute free energy */
            l0 = compute_free_energy_l0();
            l1 = vbvt_ptr->compute_free_energy_l1();
            free_energy = l0 - l1;

            if (!isfinite(free_energy)) {
                emit_internal_error("invalid variational free energy: %s (at iteration #%d)",
                                    isnan(free_energy) ? "NaN" : "infinity", iterate);
                RET_ERR(err_invalid_free_energy);
            }
            if (old_valid && old_free_energy - free_energy > prism_epsilon) {
                emit_error("variational free energy decreased [old: %.9f, new: %.9f] (at iteration #%d)",
                           old_free_energy, free_energy, iterate);
                RET_ERR(err_invalid_free_energy);
            }
            if (itemp == 1.0 && free_energy > 0.0) {
                emit_error("variational free energy exceeds zero [value: %.9f] (at iteration #%d)",
                           free_energy, iterate);
                RET_ERR(err_invalid_free_energy);
            }

            converged = (old_valid && free_energy - old_free_energy <= prism_epsilon);
            if(converged || REACHED_MAX_ITERATE(iterate)) {
                break;
            }

            old_free_energy = free_energy;
            old_valid = 1;

            SHOW_PROGRESS(iterate);
            RET_ON_ERR(update_vbvt_hyperparams());

            iterate++;
        }
        SHOW_PROGRESS_TAIL(converged,iterate,free_energy);

        if (r == 0 || free_energy > vbvt_ptr->free_energy) {
            vbvt_ptr->free_energy = free_energy;
            vbvt_ptr->iterate     = iterate;

            saved = (r < num_restart - 1);
            if (saved) {
                save_hyperparams();
            }
        }
    }

    if(saved) {
        restore_hyperparams();
    }
    transfer_hyperparams();

    return BP_TRUE;
}
