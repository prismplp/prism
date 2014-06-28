#include "up/mcmc.h"
#include "up/mcmc_sample.h"

typedef struct SwitchCountList *SWC_LIST_PTR;
struct SwitchCountList {
	SW_COUNT_PTR *swcs; /* an array of pointers to switch counts */
	int swcs_len;
	struct SwitchCountList *next;
};

typedef struct CombStack *COMB_STK_PTR;
struct CombStack {
	int goal_index;
	SWC_LIST_PTR scl_ptr;
	struct CombStack *prev;
};

int *exact_marg_goals = NULL; /* goals to compute the exact marginal likelihood */
int num_EMGs = 0;

static SWC_LIST_PTR *exact_marginal_sw_count = NULL;

static int *flat_expl = NULL;
static int flat_expl_index = 0;
static int max_flat_expl_size = INIT_MAX_EGRAPH_SIZE;

static SWC_LIST_PTR *expl_combination = NULL;

static COMB_STK_PTR comb_stack = NULL;

static int *and_stack = NULL;
static int *or_stack = NULL;
static int max_and_stack_size = INIT_MAX_EGRAPH_SIZE;;
static int max_or_stack_size = INIT_MAX_EGRAPH_SIZE;;
static int and_index = 0;
static int or_index = 0;

/*-Compute exact marginal log-likelihood of Gs by given hyper parameters As-*/
/*                                                                          */
/* P(G1,..,GN|As)                                                           */
/*         = sum_{E: E|-Gs} P(E|As)                                         */
/*               where E = <E1,..,EN>, Ei |- Gi (i=1,..,N)                  */
/*                                                                          */
/* P(E|As) = int_{thetas} P(E|thetas)*Dir(theta|As) dthetas                 */
/*               where E = msw(i,v) &...                                    */
/*                                                                          */
/*                   |            Gamma(sum_{v}(A_{i,v}))                   */
/*         = prod_{i}| ---------------------------------------              */
/*                   | Gamma(sum_{v}(sigma_{i,v}(E)+A_{i,v}))               */
/*                                                                          */
/*                       | Gamma(sigma_{i,v}(E)+A_{i,v})  ||                */
/*             * prod_{v}| ------------------------------ ||                */
/*                       |         Gamma(A_{i,v})         ||                */
/*--------------------------------------------------------------------------*/

static void push_comb_stack(int goal_index, SWC_LIST_PTR scl_ptr) {
	COMB_STK_PTR p = (COMB_STK_PTR)MALLOC(sizeof(struct CombStack));

	p->goal_index = goal_index;
	p->scl_ptr = scl_ptr;

	p->prev = comb_stack;
	comb_stack = p;
}

static void alloc_exact_marginal_cl_aux(int goal_index) {
	int i,j,sw_ins_id;
	int num_occ_sw_ins = 0;
	SWC_LIST_PTR scl_ptr;
	SW_COUNT_PTR *swcs;

	init_mh_sw_tab();

	for(i=0; i<flat_expl_index; i++) {
		sw_ins_id = flat_expl[i];
		mh_switch_table[sw_ins_id]++;
	}

	for (i=0; i<sw_ins_tab_size; i++) {
		if (mh_switch_table[i]!=0) {
			num_occ_sw_ins++;
		}
	}

	scl_ptr = (SWC_LIST_PTR)MALLOC(sizeof(struct SwitchCountList));

	if (num_occ_sw_ins==0) {
		scl_ptr->swcs_len = 0;
		scl_ptr->swcs = NULL;
	} else {
		scl_ptr->swcs_len = num_occ_sw_ins;
		swcs = (SW_COUNT_PTR *)MALLOC(num_occ_sw_ins * sizeof(SW_COUNT_PTR));
		j = 0;
		for (i=0; i<sw_ins_tab_size; i++) {
			if (mh_switch_table[i]!=0) {
				swcs[j] = (SW_COUNT_PTR)MALLOC(sizeof(struct SwitchCount));
				swcs[j]->sw_ins_id = i;
				swcs[j]->count = mh_switch_table[i];
				j++;
			}
		}
		scl_ptr->swcs = swcs;
	}

	scl_ptr->next = exact_marginal_sw_count[goal_index];
	exact_marginal_sw_count[goal_index] = scl_ptr;
}

static void expand_and_stack(int req_and_stack_size) {
	while (req_and_stack_size > max_and_stack_size) {
		if (max_and_stack_size > MAX_EGRAPH_SIZE_EXPAND_LIMIT) {
			max_and_stack_size += MAX_EGRAPH_SIZE_EXPAND_LIMIT;
		} else {
			max_and_stack_size *= 2;
		}
	}

	and_stack = (int *)REALLOC(and_stack,max_and_stack_size * sizeof(int));
}

static void expand_or_stack(int req_or_stack_size) {
	while (req_or_stack_size > max_or_stack_size) {
		if (max_or_stack_size > MAX_EGRAPH_SIZE_EXPAND_LIMIT) {
			max_or_stack_size += MAX_EGRAPH_SIZE_EXPAND_LIMIT;
		} else {
			max_or_stack_size *= 2;
		}
	}

	or_stack = (int *)REALLOC(or_stack,max_or_stack_size * sizeof(int));
}

static void trans_and_term(void) {
	int i;
	int goal_id,sws_len,goals_len,req_or_stack_size,req_and_stack_size;
	EG_PATH_PTR path_ptr,next_ptr;

	while(and_stack[--and_index]!=-1) { //switch term
		flat_expl[flat_expl_index++] = and_stack[and_index];
	}
	if(and_index>0) { //goal term
		goal_id = and_stack[--and_index];
		path_ptr = expl_graph[goal_id]->path_ptr;
		if(path_ptr!=NULL) {
			next_ptr = path_ptr->next;
			while(next_ptr!=NULL) { //OR
				goals_len = and_index + next_ptr->children_len;
				sws_len = next_ptr->sws_len;

				req_or_stack_size = or_index + goals_len + sws_len + 4;
				if(req_or_stack_size>=max_or_stack_size) {
					expand_or_stack(req_or_stack_size);
				}

				for(i=and_index-1; i>-1; i--) {
					or_stack[or_index++] = and_stack[i];
				}
				for(i=0; i<next_ptr->children_len; i++) {
					or_stack[or_index++] = next_ptr->children[i]->id;
				}
				or_stack[or_index++] = goals_len;

				for(i=0; i<next_ptr->sws_len; i++) {
					or_stack[or_index++] = next_ptr->sws[i]->id;
				}
				or_stack[or_index++] = sws_len;
				or_stack[or_index++] = flat_expl_index;
				or_stack[or_index++] = goals_len + sws_len + 4; //one term length

				next_ptr = next_ptr->next;
			}
			req_and_stack_size = and_index + path_ptr->children_len + path_ptr->sws_len + 1;
			if(req_and_stack_size>=max_and_stack_size) {
				expand_and_stack(req_and_stack_size);
			}
			for(i=0; i<path_ptr->children_len; i++) {
				and_stack[and_index++] = path_ptr->children[i]->id;
			}
			and_stack[and_index++] = -1;
			for(i=0; i<path_ptr->sws_len; i++) {
				and_stack[and_index++] = path_ptr->sws[i]->id;
			}
		}
	}
}

static void trans_or_term(void) {
	int i;
	int term_len,sws_len,goals_len,req_and_stack_size;
	int first_sw_index;

	term_len = or_stack[--or_index];

	req_and_stack_size = and_index + term_len - 3;
	if(req_and_stack_size>=max_and_stack_size) {
		expand_and_stack(req_and_stack_size);
	}

	flat_expl_index = or_stack[--or_index];
	sws_len = or_stack[--or_index];
	first_sw_index = or_index - sws_len;
	goals_len = or_stack[or_index-sws_len-1];
	or_index -= (sws_len + goals_len + 1);
	for(i=0; i<goals_len; i++) {
		and_stack[and_index++] = or_stack[or_index+i];
	}
	and_stack[and_index++] = -1;
	for(i=0; i<sws_len; i++) {
		and_stack[and_index++] = or_stack[first_sw_index+i];
	}
}

static void alloc_exact_marginal_cl(void) {
	int i,goal_id;

	exact_marginal_sw_count = (SWC_LIST_PTR *)MALLOC(num_EMGs * sizeof(SWC_LIST_PTR));
	flat_expl = (int *)MALLOC(max_flat_expl_size * sizeof(int));
	and_stack = (int *)MALLOC(max_and_stack_size * sizeof(int));
	or_stack = (int *)MALLOC(max_or_stack_size * sizeof(int));

	alloc_mh_switch_table();

	for(i=0; i<num_EMGs; i++) {
		exact_marginal_sw_count[i] = NULL;
		flat_expl_index = 0;
		or_index = 0;
		and_index = 0;

		goal_id = exact_marg_goals[i];
		and_stack[and_index++] = goal_id;
		and_stack[and_index++] = -1;

		while(or_index>0 || and_index>0) {
			if(and_index>0) { //AND exist
				trans_and_term();
			} else {
				alloc_exact_marginal_cl_aux(i);
				trans_or_term();
			}
		}
		alloc_exact_marginal_cl_aux(i);
	}

	clean_mh_switch_table();
	FREE(flat_expl);
	FREE(and_stack);
	FREE(or_stack);
}

static void clean_exact_marginal_cl(void) {
	int i,j;
	SWC_LIST_PTR scl_ptr,tmp;

	for(i=0; i<num_EMGs; i++) {
		scl_ptr = exact_marginal_sw_count[i];
		while(scl_ptr!=NULL) {
			for(j=0; j<scl_ptr->swcs_len; j++) {
				FREE(scl_ptr->swcs[j]);
			}
			tmp = scl_ptr;
			scl_ptr = scl_ptr->next;
			FREE(tmp);
		}
	}
	FREE(exact_marginal_sw_count);
}

static void combine_expl(int goal_index) {
	int i;
	SWC_LIST_PTR scl_ptr;

	for(i=goal_index; i<num_EMGs; i++) {
		scl_ptr = exact_marginal_sw_count[i];
		expl_combination[i] = scl_ptr;
		if(scl_ptr->next != NULL) {
			push_comb_stack(i,scl_ptr->next);
		}
	}
}

static double log_marginal_expl(void) {
	int i;
	SW_INS_PTR sw_ptr;
	double alpha,sumAs,sumBs,logR2_Ei;
	double logPexpl = 0.0;

	for (i=0; i<occ_switch_tab_size; i++) {
		sw_ptr = occ_switches[i];
		sumAs = 0.0;
		sumBs = 0.0;
		logR2_Ei = 0.0;
		while (sw_ptr != NULL) {
			alpha = sw_ptr->smooth_prolog + 1.0;
			sumAs += alpha;
			sumBs += sw_ptr->count;
			logR2_Ei += (lngamma(sw_ptr->count + alpha) - lngamma(alpha));
			sw_ptr = sw_ptr->next;
		}
		sumBs += sumAs;
		logPexpl += (lngamma(sumAs) - lngamma(sumBs) + logR2_Ei);;
	}
	return logPexpl;
}

static double log_marginal_goals_aux(void) {
	int i,j;
	SWC_LIST_PTR scl_ptr;
	SW_COUNT_PTR swc_ptr;

	for (i=0; i<sw_ins_tab_size; i++) {
		switch_instances[i]->count = 0;
	}

	for(i=0; i<num_EMGs; i++) {
		scl_ptr = expl_combination[i];
		for(j=0; j<scl_ptr->swcs_len; j++) {
			swc_ptr = scl_ptr->swcs[j];
			switch_instances[swc_ptr->sw_ins_id]->count += swc_ptr->count;
		}
	}

	return log_marginal_expl();
}

static double log_marginal_goals(void) {
	int goal_index;
	double this_logP, first_logP = 0.0, sum_rest;
	SWC_LIST_PTR scl_ptr;
	COMB_STK_PTR tmp;

	expl_combination = (SWC_LIST_PTR *)MALLOC(num_EMGs * sizeof(SWC_LIST_PTR));

	comb_stack = NULL;

	combine_expl(0);
	first_logP = log_marginal_goals_aux();
	sum_rest = 1.0;

	while(comb_stack!=NULL) {
		goal_index = comb_stack->goal_index;
		scl_ptr = comb_stack->scl_ptr;
		tmp = comb_stack;
		comb_stack = comb_stack->prev;
		FREE(tmp);
		expl_combination[goal_index] = scl_ptr;
		if(scl_ptr->next != NULL) {
			push_comb_stack(goal_index,scl_ptr->next);
		}
		combine_expl(goal_index+1);
		this_logP = log_marginal_goals_aux();
		if (this_logP - first_logP >= log(HUGE_PROB)) {
			sum_rest *= exp(first_logP - this_logP);
			first_logP = this_logP;
			sum_rest += 1.0; /* maybe sum_rest gets 1.0 */
		} else {
			sum_rest += exp(this_logP - first_logP);
		}
	}

	FREE(expl_combination);

	return first_logP + log(sum_rest);
}

double exact_marginal(void) {
	double lml=0.0;

	alloc_exact_marginal_cl();

	lml = log_marginal_goals();

	clean_exact_marginal_cl();
	release_occ_switches();
	release_num_sw_vals();

	return lml;
}
