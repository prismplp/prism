#include "up/mcmc.h"

#include "up/mcmc_sample.h"
#include "up/mcmc_predict.h"
#include "up/mcmc_eml.h"
#include "up/mcmc_exact.h"

static void clean_mcmc_samples(void) {
	FREE(stored_params);
	FREE(stored_hparams);
	num_stored_params = 0;
	clean_mh_state_sw_count();
	clean_sampled_values();
	release_mh_occ_switches();
}

int pc_mcmc_prepare_3(void) {
	TERM  p_fact_list;
	int   size;

	p_fact_list = bpx_get_call_arg(1,3);
	size        = bpx_get_integer(bpx_get_call_arg(2,3));
	num_goals   = bpx_get_integer(bpx_get_call_arg(3,3));

	failure_observed = 0;
	failure_root_index = -1;

	initialize_egraph_index();
	alloc_sorted_egraph(size);
	RET_ON_ERR(sort_egraphs(p_fact_list));

	alloc_occ_switches();
	alloc_num_sw_vals();

	return BP_TRUE;
}

int pc_clean_mcmc_samples_0(void) {
	clean_mcmc_samples();
	return BP_TRUE;
}

int pc_mcmc_sample_7(void) {
	TERM p_goal_list,p_goal;
	TERM p_trace,p_postparam;
	int burn_in,end,skip;
	int i;

	clean_mcmc_samples(); /* clean up the previous sampling results */

	p_goal_list        = bpx_get_call_arg(1,7);
	num_observed_goals = bpx_get_integer(bpx_get_call_arg(2,7));
	end                = bpx_get_integer(bpx_get_call_arg(3,7));
	burn_in            = bpx_get_integer(bpx_get_call_arg(4,7));
	skip               = bpx_get_integer(bpx_get_call_arg(5,7));
	p_trace            = bpx_get_call_arg(6,7);
	p_postparam        = bpx_get_call_arg(7,7);

	if (bpx_is_list(p_trace)) {
		trace_sw_id = bpx_get_integer(bpx_get_car(p_trace));
		trace_step = bpx_get_integer(bpx_get_car(bpx_get_cdr(p_trace)));
	}
	if (bpx_is_list(p_postparam)) {
		postparam_sw_id = bpx_get_integer(bpx_get_car(p_postparam));
		postparam_step = bpx_get_integer(bpx_get_car(bpx_get_cdr(p_postparam)));
	}

	observed_goals = (int *)MALLOC(num_observed_goals * sizeof(int));

	for (i=0; i<num_observed_goals; i++) {
		p_goal = bpx_get_car(p_goal_list);
		observed_goals[i] = prism_goal_id_get(p_goal);
		p_goal_list = bpx_get_cdr(p_goal_list);
	}

	RET_ON_ERR(loop_mh(end,burn_in,skip));

	FREE(observed_goals);

	return BP_TRUE;
}

int pc_mcmc_marginal_1(void) {
	double eml;

	if (mh_state_sw_count == NULL) {
		RET_ERR(err_mcmc_unfinished);
	}

	eml = mh_marginal();

	return bpx_unify(bpx_get_call_arg(1,1),bpx_build_float(eml));
}

int pc_mcmc_predict_4(void) {
	TERM p_goal_list,p_goal,p_ans,p_ans_list,p_ans_list1;
	TERM p_n_viterbi_list,p_rank_list,p_temp;
	int n;
	int *prediction_goals;
	PR_ANS_PTR *ans;
	PR_ANS_PTR ans_ptr,prev_ptr;
	int i;

	if (mh_state_sw_count == NULL) {
		RET_ERR(err_mcmc_unfinished);
	}

	p_goal_list = bpx_get_call_arg(1,4);
	num_predict_goals = bpx_get_integer(bpx_get_call_arg(2,4));
	n = bpx_get_integer(bpx_get_call_arg(3,4));

	prediction_goals = (int *)MALLOC(num_predict_goals * sizeof(int));

	for (i = 0; i < num_predict_goals; i++) {
		p_goal = bpx_get_car(p_goal_list);
		prediction_goals[i] = prism_goal_id_get(p_goal);
		p_goal_list = bpx_get_cdr(p_goal_list);
	}

	ans = (PR_ANS_PTR *)MALLOC(num_predict_goals * sizeof(PR_ANS_PTR));

	mh_predict(prediction_goals,n,ans);

	p_ans_list = bpx_build_nil();
	for (i=num_predict_goals-1; i>-1; i--) {
		p_ans = bpx_build_structure("ans",3);
		get_only_nth_most_likely_path(ans[i]->next->rank + 1,
		                              prediction_goals[i],
		                              &p_n_viterbi_list);
		bpx_unify(bpx_get_arg(1,p_ans),p_n_viterbi_list);
		bpx_unify(bpx_get_arg(2,p_ans),bpx_build_float(ans[i]->next->logP));

		ans_ptr = ans[i]->next;
		p_rank_list = bpx_build_list();
		p_temp = p_rank_list;
		while (ans_ptr != NULL) {
			bpx_unify(bpx_get_car(p_temp),bpx_build_integer(ans_ptr->rank+1));
			ans_ptr = ans_ptr->next;
			if (ans_ptr != NULL) {
				bpx_unify(bpx_get_cdr(p_temp),bpx_build_list());
				p_temp = bpx_get_cdr(p_temp);
			} else {
				bpx_unify(bpx_get_cdr(p_temp),bpx_build_nil());
			}
		}
		bpx_unify(bpx_get_arg(3,p_ans),p_rank_list);

		p_ans_list1 = bpx_build_list();
		bpx_unify(bpx_get_car(p_ans_list1),p_ans);
		bpx_unify(bpx_get_cdr(p_ans_list1),p_ans_list);
		p_ans_list = p_ans_list1;
	}

	for (i=0; i<num_predict_goals; i++) {
		ans_ptr = ans[i];
		while (ans_ptr != NULL) {
			prev_ptr = ans_ptr;
			ans_ptr = ans_ptr->next;
			FREE(prev_ptr);
		}
	}
	FREE(ans);
	FREE(prediction_goals);

	return bpx_unify(bpx_get_call_arg(4,4),p_ans_list);
}

int pc_exact_marginal_3(void) {
	int i;
	TERM p_goal_list,p_goal;
	double lml;

	p_goal_list = bpx_get_call_arg(1,3);
	num_EMGs = bpx_get_integer(bpx_get_call_arg(2,3));

	exact_marg_goals = (int *)MALLOC(num_EMGs * sizeof(int));

	for (i=0; i<num_EMGs; i++) {
		p_goal = bpx_get_car(p_goal_list);
		exact_marg_goals[i] = prism_goal_id_get(p_goal);
		p_goal_list = bpx_get_cdr(p_goal_list);
	}

	lml = exact_marginal();

	FREE(exact_marg_goals);

	return bpx_unify(bpx_get_call_arg(3,3),bpx_build_float(lml));
}
