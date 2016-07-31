
#include <stdio.h>
#include "bprolog.h"
#include "up/up.h"
#include "up/util.h"
#include "up/graph.h"
#include "up/graph_aux.h"
#include "up/flags.h"
#include "up/viterbi.h"
#include "up/crf.h"
#include "up/em_aux.h"
#include "up/em_aux_ml.h"
#include "up/hindsight.h"
#include "core/fputil.h"
#include "core/random.h"
#include "up/crf_learn.h"
#include "up/crf_learn_aux.h"
#include "up/crf_rank.h"

int num_rank_lists;
RankList* rank_lists;

/* main loop */
int run_grd(CRF_ENG_PTR crf_ptr) {
	int r,iterate,old_valid,converged,saved = 0;
	double likelihood,old_likelihood = 0.0;
	double tmp_epsilon,alpha0,gf_sd,old_gf_sd = 0.0;

	config_crf(crf_ptr);

	initialize_weights();

	if (crf_learn_mode == 1) {
		initialize_LBFGS();
		printf("L-BFGS mode\n");
	}

	if (crf_learning_rate==1) {
		printf("learning rate:annealing\n");
	} else if (crf_learning_rate==2) {
		printf("learning rate:backtrack\n");
	} else if (crf_learning_rate==3) {
		printf("learning rate:golden section\n");
	}

	for (r = 0; r < num_restart; r++) {
		SHOW_PROGRESS_HEAD("#crf-iters", r);

		initialize_crf_count();
		initialize_lambdas();
		initialize_visited_flags();

		old_valid = 0;
		iterate = 0;
		tmp_epsilon = crf_epsilon;

		restart_LBFGS();

		while (1) {
			if (CTRLC_PRESSED) {
				SHOW_PROGRESS_INTR();
				RET_ERR(err_ctrl_c_pressed);
			}

			RET_ON_ERR(crf_ptr->compute_feature());

			crf_ptr->compute_crf_probs();

			likelihood = crf_ptr->compute_likelihood();

			if (verb_em) {
				prism_printf("Iteration #%d:\tlog_likelihood=%.9f\n", iterate, likelihood);
			}

			if (debug_level) {
				prism_printf("After I-step[%d]:\n", iterate);
				prism_printf("likelihood = %.9f\n", likelihood);
				print_egraph(debug_level, PRINT_EM);
			}

			if (!isfinite(likelihood)) {
				emit_internal_error("invalid log likelihood: %s (at iteration #%d)",
				                    isnan(likelihood) ? "NaN" : "infinity", iterate);
				RET_ERR(ierr_invalid_likelihood);
			}
			/*        if (old_valid && old_likelihood - likelihood > prism_epsilon) {
					  emit_error("log likelihood decreased [old: %.9f, new: %.9f] (at iteration #%d)",
					  old_likelihood, likelihood, iterate);
					  RET_ERR(err_invalid_likelihood);
					  }*/
			if (likelihood > 0.0) {
				emit_error("log likelihood greater than zero [value: %.9f] (at iteration #%d)",
				           likelihood, iterate);
				RET_ERR(err_invalid_likelihood);
			}

			if (crf_learn_mode == 1 && iterate > 0) restore_old_gradient();

			RET_ON_ERR(crf_ptr->compute_gradient());

			if (crf_learn_mode == 1 && iterate > 0) {
				compute_LBFGS_y_rho();
				compute_hessian(iterate);
			} else if (crf_learn_mode == 1 && iterate == 0) {
				initialize_LBFGS_q();
			}

			converged = (old_valid && fabs(likelihood - old_likelihood) <= prism_epsilon);

			if (converged || REACHED_MAX_ITERATE(iterate)) {
				break;
			}

			old_likelihood = likelihood;
			old_valid = 1;

			if (debug_level) {
				prism_printf("After O-step[%d]:\n", iterate);
				print_egraph(debug_level, PRINT_EM);
			}

			SHOW_PROGRESS(iterate);
			
			// computing learning rate
			if (crf_learning_rate == 1) { // annealing
				tmp_epsilon = (annealing_weight / (annealing_weight + iterate)) * crf_epsilon;
			} else if (crf_learning_rate == 2) { // line-search(backtrack)
				// gf_sd = grad f^T dot d (search direction)
				if (crf_learn_mode == 1) {
					gf_sd = compute_gf_sd_LBFGS();
				} else {
					gf_sd = compute_gf_sd();
				}
				if (iterate==0) {
					alpha0 = 1;
				} else {
					alpha0 = tmp_epsilon * old_gf_sd / gf_sd;
				}
				if (crf_learn_mode == 1) {
					tmp_epsilon = line_search_LBFGS(crf_ptr,alpha0,crf_ls_rho,crf_ls_c1,likelihood,gf_sd);
				} else {
					tmp_epsilon = line_search(crf_ptr,alpha0,crf_ls_rho,crf_ls_c1,likelihood,gf_sd);
				}

				if (tmp_epsilon < EPS) {
					emit_error("invalid alpha in line search(=0.0) (at iteration #%d)",iterate);
					RET_ERR(err_line_search);
				}
				old_gf_sd = gf_sd;
			} else if (crf_learning_rate == 3) { // line-search(golden section)
				if (crf_learn_mode == 1) {
					tmp_epsilon = golden_section_LBFGS(crf_ptr,0,crf_golden_b);
				} else {
					tmp_epsilon = golden_section(crf_ptr,0,crf_golden_b);
				}
			}
			// updating with learning rate 
			crf_ptr->update_lambdas(tmp_epsilon);

			iterate++;
		}

		SHOW_PROGRESS_TAIL(converged, iterate, likelihood);

		if (r == 0 || likelihood > crf_ptr->likelihood) {
			crf_ptr->likelihood = likelihood;
			crf_ptr->iterate    = iterate;

			saved = (r < num_restart - 1);
			if (saved) {
				save_params();
			}
		}
	}

	if (crf_learn_mode == 1) clean_LBFGS();
	INIT_VISITED_FLAGS;
	return BP_TRUE;
}

int pc_crf_rank_prepare_5(void) {
	TERM  p_fact_list,p_rank_lists;
	int   size;

	p_fact_list        = bpx_get_call_arg(1,5);
	size               = bpx_get_integer(bpx_get_call_arg(2,5));
	num_goals          = bpx_get_integer(bpx_get_call_arg(3,5));
	failure_root_index = bpx_get_integer(bpx_get_call_arg(4,5));
	p_rank_lists         = bpx_get_call_arg(5,5);

	failure_observed = (failure_root_index != -1);

	if (failure_root_index != -1) {
		failure_subgoal_id = prism_goal_id_get(failure_atom);
		if (failure_subgoal_id == -1) {
			emit_internal_error("no subgoal ID allocated to `failure'");
			RET_INTERNAL_ERR;
		}
	}

	initialize_egraph_index();
	alloc_sorted_egraph(size);
	RET_ON_ERR(sort_crf_egraphs(p_fact_list));
#ifndef MPI
	if (verb_graph) {
		print_egraph(0, PRINT_NEUTRAL);
	}
#endif /* !(MPI) */

	alloc_occ_switches();
	alloc_num_sw_vals();


	// count num_rank_list
	num_rank_lists=0;
	TERM  p_first_rank_list=p_rank_lists;
	while (bpx_is_list(p_rank_lists)) {
		p_rank_lists = bpx_get_cdr(p_rank_lists);
		num_rank_lists++;
	}
	p_rank_lists=p_first_rank_list;
	
	// build rank lists
	rank_lists=(RankList*)MALLOC(num_rank_lists*sizeof(RankList));
	int i,j;
	for(i=0;i<num_rank_lists;i++){
		TERM p_ranks = bpx_get_car(p_rank_lists);
		p_rank_lists = bpx_get_cdr(p_rank_lists);
		int num_ranks=0;
		// count ranks
		TERM  p_first_ranks=p_ranks;
		while (bpx_is_list(p_ranks)) {
			p_ranks = bpx_get_cdr(p_ranks);
			num_ranks++;
		}
		p_ranks=p_first_ranks;
		// build ranks
		RankData* ranks=MALLOC(num_ranks*sizeof(RankData));
		for(j=0;j<num_ranks;j++){
			TERM p_gid = bpx_get_car(p_ranks);
			p_ranks = bpx_get_cdr(p_ranks);
			int goal_id = bpx_get_integer(p_gid);
			ranks[j].goal_id=goal_id;
			printf("%d,",goal_id);
		}
		rank_lists[i].num_ranks=num_ranks;
		rank_lists[i].ranks=ranks;
		printf("\n");
	}
	return BP_TRUE;
}

int pc_crf_rank_learn_2(void) {
	struct CRF_Engine crf_eng;

	RET_ON_ERR(run_grd(&crf_eng));

	return
	    bpx_unify(bpx_get_call_arg(1,2), bpx_build_integer(crf_eng.iterate)) &&
	    bpx_unify(bpx_get_call_arg(2,2), bpx_build_float(crf_eng.likelihood));
}


