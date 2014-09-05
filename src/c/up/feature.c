#include <stdio.h>
#include "bprolog.h"
#include "up/up.h"
#include "up/util.h"
#include "up/graph.h"
#include "up/graph_aux.h"
#include "up/flags.h"
#include "up/viterbi.h"
#include "up/crf_grd.h"
#include "up/em_aux.h"
#include "up/em_aux_ml.h"
#include "up/hindsight.h"
#include "core/fputil.h"
#include "core/random.h"

static int LBFGS_index = 0;
static double *LBFGS_a = NULL;
static double *LBFGS_b = NULL;
static double *LBFGS_rho = NULL;

/*------------------------------------------------------------------------*/

/* initialize methods */

static void initialize_lambdas_noisy_uniform(void) {
	int i;
	SW_INS_PTR ptr;

	for (i = 0; i < occ_switch_tab_size; i++) {
		ptr = occ_switches[i];

		if (ptr->fixed > 0) continue;

		while (ptr != NULL) {
			ptr->inside = random_gaussian(0, std_ratio);
			ptr = ptr->next;
		}
	}
}

static void initialize_lambdas_random(void) {
	int i;
	SW_INS_PTR ptr;

	for (i = 0; i < occ_switch_tab_size; i++) {
		ptr = occ_switches[i];

		if (ptr->fixed > 0) continue;

		while (ptr != NULL) {
			ptr->inside = random_float();
			ptr = ptr->next;
		}
	}
}

static void initialize_lambdas_zero(void) {
	int i;
	SW_INS_PTR ptr;

	for (i = 0; i < occ_switch_tab_size; i++) {
		ptr = occ_switches[i];

		if (ptr->fixed > 0) continue;

		while (ptr != NULL) {
			ptr->inside = 0;
			ptr = ptr->next;
		}
	}
}

static void initialize_lambdas(void) {
	if (crf_init_method == 1)
		initialize_lambdas_noisy_uniform();
	if (crf_init_method == 2)
		initialize_lambdas_random();
	if (crf_init_method == 3)
		initialize_lambdas_zero();
}

void initialize_weights(void) {
	int i;
	SW_INS_PTR ptr;

	for (i = 0; i < sw_tab_size; i++) {
		ptr = switches[i];

		while (ptr != NULL) {
			ptr->inside_h = ptr->smooth_prolog + 1.0;
			ptr = ptr->next;
		}
	}
}

static void set_visited_flags(EG_NODE_PTR node_ptr) {
	int i;
	EG_PATH_PTR path_ptr;

	if (node_ptr->visited == 1) return;

	node_ptr->visited = 1;

	path_ptr = node_ptr->path_ptr;

	while (path_ptr!=NULL) {
		for (i=0; i<path_ptr->children_len; i++) {
			set_visited_flags(path_ptr->children[i]);
		}
		path_ptr = path_ptr->next;
	}
}

static void initialize_visited_flags(void) {
	int i;
	EG_NODE_PTR node_ptr;

	for (i=0; i<sorted_egraph_size; i++) {
		sorted_expl_graph[i]->visited = 0;
	}

	for (i=0; i<num_roots; i++) {
		if (roots[i]->pid == -1) {
			node_ptr = expl_graph[roots[i]->id];
			set_visited_flags(node_ptr);
		}
	}
}

/*------------------------------------------------------------------------*/

/* compute node->inside (CRF inside) */
int compute_feature_scaling_none(void) {
	int i,k;
	double sum, this_path_inside, this_sws_inside;
	EG_NODE_PTR eg_ptr;
	EG_PATH_PTR path_ptr;

	for (i = 0; i < sorted_egraph_size; i++) {
		eg_ptr = sorted_expl_graph[i];
		sum = 0.0;
		path_ptr = eg_ptr->path_ptr;
		if (path_ptr == NULL) {
			sum = 1.0; /* path_ptr should not be NULL; but it happens */
		} else {
			while (path_ptr != NULL) {
				this_path_inside = 1.0;
				for (k = 0; k < path_ptr->children_len; k++) {
					this_path_inside *= path_ptr->children[k]->inside;
				}
				this_sws_inside = 0.0;
				for (k = 0; k < path_ptr->sws_len; k++) {
					this_sws_inside += (path_ptr->sws[k]->inside * path_ptr->sws[k]->inside_h);
				}
				this_path_inside *= exp(this_sws_inside);
				path_ptr->inside = this_path_inside;  //path_ptr->inside = exp(?)
				sum += this_path_inside;
				path_ptr = path_ptr->next;
			}
		}
		if (!isfinite(sum)) {
			emit_error("overflow CRF-inside");
			RET_ERR(err_overflow);
		}
		eg_ptr->inside = sum;  //eg_ptr->inside = exp(?)
	}

	return BP_TRUE;
}

int compute_feature_scaling_log_exp(void) {
	int i,k,t;
	double sum,sum_rest,this_path_inside,first_path_inside = 0.0;
	EG_NODE_PTR eg_ptr;
	EG_PATH_PTR path_ptr;

	for (i = 0; i < sorted_egraph_size; i++) {
		eg_ptr = sorted_expl_graph[i];
		sum = 0.0;
		path_ptr = eg_ptr->path_ptr;
		if (path_ptr == NULL) {
			sum = 0.0; /* path_ptr should not be NULL; but it happens */
		} else {
			sum_rest = 0.0;
			t = 0;
			while (path_ptr != NULL) {
				this_path_inside = 0.0;
				for (k = 0; k < path_ptr->children_len; k++) {
					this_path_inside += path_ptr->children[k]->inside;
				}
				for (k = 0; k < path_ptr->sws_len; k++) {
					this_path_inside += (path_ptr->sws[k]->inside * path_ptr->sws[k]->inside_h);
				}
				path_ptr->inside = this_path_inside;  //path_ptr->inside = log(exp(?))
				if (t == 0) {
					first_path_inside = this_path_inside;
					sum_rest += 1.0;
				} else if (this_path_inside - first_path_inside >= log(HUGE_PROB)) {
					sum_rest *= exp(first_path_inside - this_path_inside);
					first_path_inside = this_path_inside;
					sum_rest += 1.0; /* maybe sum_rest gets 1.0 */
				} else {
					sum_rest += exp(this_path_inside - first_path_inside);
				}
				path_ptr = path_ptr->next;
				t++;
			}
			sum = first_path_inside + log(sum_rest);
		}
		eg_ptr->inside = sum;
	}

	return BP_TRUE;
}

/*------------------------------------------------------------------------*/

/* compute gradient */

static void count_complete_features(EG_NODE_PTR node_ptr, int count) {
	int k;
	EG_PATH_PTR path_ptr;

	path_ptr = node_ptr->path_ptr;

	while (path_ptr!=NULL) {
		for (k=0; k<path_ptr->children_len; k++) {
			count_complete_features(path_ptr->children[k],count);
		}
		for (k=0; k<path_ptr->sws_len; k++) {
			path_ptr->sws[k]->count += count;
		}
		path_ptr = path_ptr->next;
	}
}

static void initialize_crf_count(void) {
	int i;

	for (i=0; i<sw_ins_tab_size; i++) {
		switch_instances[i]->count = 0;
	}

	for (i=0; i<num_roots; i++) {
		roots[i]->sgd_count = roots[i]->count;
		expl_graph[roots[i]->id]->root_id = i;
		if (roots[i]->pid != -1) {
			count_complete_features(expl_graph[roots[i]->id],roots[i]->count);
		}
	}
}

/* compute gradient for each switch (gradient is always no-scale)*/
/* sw->gradient = ( sw->count - sw->total_expect ) * sw->inside_h */
/* use node->inside */
static int compute_gradient_scaling_none(void) {
	int i,k;
	EG_PATH_PTR path_ptr;
	EG_NODE_PTR eg_ptr,node_ptr;
	SW_INS_PTR sw_ptr;
	double q;

	for (i = 0; i < sw_ins_tab_size; i++) {
		switch_instances[i]->total_expect = 0.0;
	}

	for (i = 0; i < sorted_egraph_size; i++) {
		sorted_expl_graph[i]->outside = 0.0;
	}

	for (i = 0; i < num_roots; i++) {
		if (roots[i]->pid == -1) {
			eg_ptr = expl_graph[roots[i]->id];
			if (i == failure_root_index) {
				eg_ptr->outside = num_goals / (1.0 - inside_failure);
			} else {
				eg_ptr->outside = roots[i]->sgd_count / eg_ptr->inside;
			}
		}
	}

	for (i = sorted_egraph_size - 1; i >= 0; i--) {
		eg_ptr = sorted_expl_graph[i];
		if (eg_ptr->visited == 0) continue;
		path_ptr = eg_ptr->path_ptr;
		while (path_ptr != NULL) {
			q = eg_ptr->outside * path_ptr->inside;
			if (q > 0.0) {
				for (k = 0; k < path_ptr->children_len; k++) {
					node_ptr = path_ptr->children[k];
					node_ptr->outside += q / node_ptr->inside;
				}
				for (k = 0; k < path_ptr->sws_len; k++) {
					sw_ptr = path_ptr->sws[k];
					sw_ptr->total_expect += q;
				}
			}
			path_ptr = path_ptr->next;
		}
	}

	for (i=0; i<occ_switch_tab_size; i++) {
		sw_ptr = occ_switches[i];
		while (sw_ptr!=NULL) {
			sw_ptr->gradient = (sw_ptr->count - sw_ptr->total_expect) * sw_ptr->inside_h;
			if (crf_penalty != 0.0) {
				sw_ptr->gradient -= sw_ptr->inside * crf_penalty;
			}
			sw_ptr = sw_ptr->next;
		}
	}

	return BP_TRUE;
}

static int compute_gradient_scaling_log_exp(void) {
	int i,k;
	EG_PATH_PTR path_ptr;
	EG_NODE_PTR eg_ptr,node_ptr;
	SW_INS_PTR sw_ptr;
	double q,r;

	for (i = 0; i < sw_ins_tab_size; i++) {
		switch_instances[i]->total_expect = 0.0;
		switch_instances[i]->has_first_expectation = 0;
		switch_instances[i]->first_expectation = 0.0;
	}

	for (i = 0; i < sorted_egraph_size; i++) {
		sorted_expl_graph[i]->outside = 0.0;
		sorted_expl_graph[i]->has_first_outside = 0;
		sorted_expl_graph[i]->first_outside = 0.0;
	}

	for (i = 0; i < num_roots; i++) {
		if (roots[i]->pid == -1) {
			eg_ptr = expl_graph[roots[i]->id];
			if (i == failure_root_index) {
				eg_ptr->first_outside =
				    log(num_goals / (1.0 - exp(inside_failure)));
			} else {
				eg_ptr->first_outside =
				    log((double)(roots[i]->sgd_count)) - eg_ptr->inside;
			}
			eg_ptr->has_first_outside = 1;
			eg_ptr->outside = 1.0;
		}
	}

	/* sorted_expl_graph[to] must be a root node */
	for (i = sorted_egraph_size - 1; i >= 0; i--) {
		eg_ptr = sorted_expl_graph[i];

		if (eg_ptr->visited == 0) continue;

		/* First accumulate log-scale outside probabilities: */
		if (!eg_ptr->has_first_outside) {
			emit_internal_error("unexpected has_first_outside[%s]",
			                    prism_goal_string(eg_ptr->id));
			RET_INTERNAL_ERR;
		} else if (!(eg_ptr->outside > 0.0)) {
			emit_internal_error("unexpected outside[%s]",
			                    prism_goal_string(eg_ptr->id));
			RET_INTERNAL_ERR;
		} else {
			eg_ptr->outside = eg_ptr->first_outside + log(eg_ptr->outside);
		}

		path_ptr = sorted_expl_graph[i]->path_ptr;
		while (path_ptr != NULL) {
			q = sorted_expl_graph[i]->outside + path_ptr->inside;
			for (k = 0; k < path_ptr->children_len; k++) {
				node_ptr = path_ptr->children[k];
				r = q - node_ptr->inside;
				if (!node_ptr->has_first_outside) {
					node_ptr->first_outside = r;
					node_ptr->outside += 1.0;
					node_ptr->has_first_outside = 1;
				} else if (r - node_ptr->first_outside >= log(HUGE_PROB)) {
					node_ptr->outside *= exp(node_ptr->first_outside - r);
					node_ptr->first_outside = r;
					node_ptr->outside += 1.0;
				} else {
					node_ptr->outside += exp(r - node_ptr->first_outside);
				}
			}
			for (k = 0; k < path_ptr->sws_len; k++) {
				sw_ptr = path_ptr->sws[k];
				if (!sw_ptr->has_first_expectation) {
					sw_ptr->first_expectation = q;
					sw_ptr->total_expect += 1.0;
					sw_ptr->has_first_expectation = 1;
				} else if (q - sw_ptr->first_expectation >= log(HUGE_PROB)) {
					sw_ptr->total_expect *= exp(sw_ptr->first_expectation - q);
					sw_ptr->first_expectation = q;
					sw_ptr->total_expect += 1.0;
				} else {
					sw_ptr->total_expect += exp(q - sw_ptr->first_expectation);
				}
			}
			path_ptr = path_ptr->next;
		}
	}

	/* unscale total_expect */
	for (i = 0; i < sw_ins_tab_size; i++) {
		sw_ptr = switch_instances[i];
		if (!sw_ptr->has_first_expectation) continue;
		if (!(sw_ptr->total_expect > 0.0)) {
			emit_error("unexpected expectation for %s",prism_sw_ins_string(i));
			RET_ERR(err_invalid_numeric_value);
		}
		sw_ptr->total_expect =
		    exp(sw_ptr->first_expectation + log(sw_ptr->total_expect));
	}

	for (i=0; i<occ_switch_tab_size; i++) {
		sw_ptr = occ_switches[i];
		while (sw_ptr!=NULL) {
			sw_ptr->gradient = (sw_ptr->count - sw_ptr->total_expect) * sw_ptr->inside_h;
			if (crf_penalty != 0.0) {
				sw_ptr->gradient -= sw_ptr->inside * crf_penalty;
			}
			sw_ptr = sw_ptr->next;
		}
	}

	return BP_TRUE;
}

/*------------------------------------------------------------------------*/

/* compute log-likelihood */
/* use node->crfprob */
static double compute_log_likelihood_scaling_none(void) {
	int i;
	double log_likelihood,penalty;
	SW_INS_PTR sw_ptr;

	log_likelihood = 0.0;

	for (i=0; i<num_roots; i++) {
		if (roots[i]->pid != -1) {
			log_likelihood += log(expl_graph[roots[i]->id]->crfprob) * roots[i]->count;
		}
	}

	if (crf_penalty != 0.0) {
		penalty = 0.0;
		for (i=0; i<occ_switch_tab_size; i++) {
			sw_ptr = occ_switches[i];
			while (sw_ptr!=NULL) {
				penalty += sw_ptr->inside * sw_ptr->inside;
				sw_ptr = sw_ptr->next;
			}
		}
		penalty *= (crf_penalty / 2);
		log_likelihood -= penalty;
	}

	return log_likelihood;

}

static double compute_log_likelihood_scaling_log_exp(void) {
	int i;
	double log_likelihood,penalty;
	SW_INS_PTR sw_ptr;

	log_likelihood = 0.0;

	for (i=0; i<num_roots; i++) {
		if (roots[i]->pid != -1) {
			log_likelihood += expl_graph[roots[i]->id]->crfprob * roots[i]->count;
		}
	}

	if (crf_penalty != 0.0) {
		penalty = 0.0;
		for (i=0; i<occ_switch_tab_size; i++) {
			sw_ptr = occ_switches[i];
			while (sw_ptr!=NULL) {
				penalty += sw_ptr->inside * sw_ptr->inside;
				sw_ptr = sw_ptr->next;
			}
		}
		penalty *= (crf_penalty / 2);
		log_likelihood -= penalty;
	}


	return log_likelihood;

}

/*------------------------------------------------------------------------*/

/* compute crfprob P(E|G) */
/* use node->inside */
static int compute_crf_probs_scaling_none(void) {
	int i;
	EG_NODE_PTR eg_ptr;
	EG_NODE_PTR peg_ptr;

	for (i=0; i<num_roots; i++) {
		if (roots[i]->pid != -1) {
			eg_ptr = expl_graph[roots[i]->id];
			peg_ptr = expl_graph[roots[i]->pid];
			eg_ptr->crfprob = eg_ptr->inside / peg_ptr->inside;
		}
	}

	return BP_TRUE;
}

static int compute_crf_probs_scaling_log_exp(void) {
	int i;
	EG_NODE_PTR eg_ptr;
	EG_NODE_PTR peg_ptr;

	for (i=0; i<num_roots; i++) {
		if (roots[i]->pid != -1) {
			eg_ptr = expl_graph[roots[i]->id];
			peg_ptr = expl_graph[roots[i]->pid];
			eg_ptr->crfprob = eg_ptr->inside - peg_ptr->inside;
		}
	}

	return BP_TRUE;
}

/*------------------------------------------------------------------------*/

/* update params */
static int update_lambdas(double tmp_epsilon) {
	int i;
	SW_INS_PTR sw_ptr;

	for (i=0; i<occ_switch_tab_size; i++) {
		sw_ptr = occ_switches[i];
		if (sw_ptr->fixed > 0) continue;
		while (sw_ptr!=NULL) {
			if (sw_ptr->fixed == 0) {
				sw_ptr->inside += tmp_epsilon * sw_ptr->gradient;
			}
			sw_ptr = sw_ptr->next;
		}
	}

	return BP_TRUE;
}

/*------------------------------------------------------------------------*/

/* learning rate */

static void save_current_params(void) {
	int i;
	SW_INS_PTR ptr;

	for (i = 0; i < occ_switch_tab_size; i++) {
		ptr = occ_switches[i];
		if (ptr->fixed > 0) continue;
		while (ptr != NULL) {
			ptr->current_inside = ptr->inside;
			ptr = ptr->next;
		}
	}
}

static void restore_current_params(void) {
	int i;
	SW_INS_PTR ptr;

	for (i = 0; i < occ_switch_tab_size; i++) {
		ptr = occ_switches[i];
		if (ptr->fixed > 0) continue;
		while (ptr != NULL) {
			ptr->inside = ptr->current_inside;
			ptr = ptr->next;
		}
	}
}

static double compute_gf_sd(void) {
	int i;
	SW_INS_PTR sw_ptr;
	double gf_sd;

	gf_sd = 0.0;
	for (i=0; i<occ_switch_tab_size; i++) {
		sw_ptr = occ_switches[i];
		if (sw_ptr->fixed > 0) continue;
		while (sw_ptr!=NULL) {
			if (sw_ptr->fixed == 0) {
				gf_sd -= sw_ptr->gradient * sw_ptr->gradient;
			}
			sw_ptr = sw_ptr->next;
		}
	}

	return gf_sd;
}

static double compute_gf_sd_LBFGS(void) {
	int i;
	SW_INS_PTR sw_ptr;
	double gf_sd;

	gf_sd = 0.0;
	for (i=0; i<occ_switch_tab_size; i++) {
		sw_ptr = occ_switches[i];
		if (sw_ptr->fixed > 0) continue;
		while (sw_ptr!=NULL) {
			if (sw_ptr->fixed == 0) {
				gf_sd += sw_ptr->LBFGS_q * sw_ptr->gradient;
			}
			sw_ptr = sw_ptr->next;
		}
	}

	return gf_sd;
}

/* line search(backtrack) */
static double compute_phi_alpha(CRF_ENG_PTR crf_ptr, double alpha) {
	int i;
	SW_INS_PTR sw_ptr;

	for (i=0; i<occ_switch_tab_size; i++) {
		sw_ptr = occ_switches[i];
		if (sw_ptr->fixed > 0) continue;
		while (sw_ptr!=NULL) {
			if (sw_ptr->fixed == 0) {
				sw_ptr->inside = sw_ptr->current_inside + ( alpha * sw_ptr->gradient );
			}
			sw_ptr = sw_ptr->next;
		}
	}

	RET_ON_ERR(crf_ptr->compute_feature());
	crf_ptr->compute_crf_probs();
	return crf_ptr->compute_likelihood() * (-1);
}

static double compute_phi_alpha_LBFGS(CRF_ENG_PTR crf_ptr, double alpha) {
	int i;
	SW_INS_PTR sw_ptr;

	for (i=0; i<occ_switch_tab_size; i++) {
		sw_ptr = occ_switches[i];
		if (sw_ptr->fixed > 0) continue;
		while (sw_ptr!=NULL) {
			if (sw_ptr->fixed == 0) {
				sw_ptr->inside = sw_ptr->current_inside - ( alpha * sw_ptr->LBFGS_q );
			}
			sw_ptr = sw_ptr->next;
		}
	}

	RET_ON_ERR(crf_ptr->compute_feature());
	crf_ptr->compute_crf_probs();
	return crf_ptr->compute_likelihood() * (-1);
}

static double line_search(CRF_ENG_PTR crf_ptr, double alpha0, double rho, double c1, double likelihood, double gf_sd) {
	double c_gf_sd,alpha,l_k,l_k2;

	l_k = (-1) * likelihood;
	c_gf_sd = gf_sd * c1;

	save_current_params();

	alpha = alpha0;
	while (1) {
		l_k2 = compute_phi_alpha(crf_ptr,alpha);
		if ( l_k2 <= ( l_k + alpha * c_gf_sd ) ) break;
		alpha *= rho;
	}

	restore_current_params();

	return alpha;
}

static double line_search_LBFGS(CRF_ENG_PTR crf_ptr, double alpha0, double rho, double c1, double likelihood, double gf_sd) {
	double c_gf_sd,alpha,l_k,l_k2;

	l_k = (-1) * likelihood;
	c_gf_sd = gf_sd * c1;

	save_current_params();

	alpha = alpha0;
	while (1) {
		l_k2 = compute_phi_alpha_LBFGS(crf_ptr,alpha);
		if ( l_k2 <= ( l_k + alpha * c_gf_sd ) ) break;
		alpha *= rho;
	}

	restore_current_params();

	return alpha;
}

/* golden section */
static double golden_section(CRF_ENG_PTR crf_ptr,double a, double b) {
	double p,q,f_p,f_q;
	static double tau;

	save_current_params();

	tau = (sqrt(5) - 1)/2;

	p = b - tau * ( b - a );
	q = a + tau * ( b - a );

	f_p = compute_phi_alpha(crf_ptr,p);

	f_q = compute_phi_alpha(crf_ptr,q);

	while ( b - a >= prism_epsilon ) {
		if ( f_p < f_q ) {
			b = q;
			q = p;
			p = b - tau * (b - a);
			f_q = f_p;
			f_p = compute_phi_alpha(crf_ptr,p);
		} else {
			a = p;
			p = q;
			q = a + tau * (b - a);
			f_p = f_q;
			f_q = compute_phi_alpha(crf_ptr,q);
		}
	}

	restore_current_params();

	return (a + b)/2.0;
}

static double golden_section_LBFGS(CRF_ENG_PTR crf_ptr,double a, double b) {
	double p,q,f_p,f_q;
	static double tau;

	save_current_params();

	tau = (sqrt(5) - 1)/2;

	p = b - tau * ( b - a );
	q = a + tau * ( b - a );

	f_p = compute_phi_alpha_LBFGS(crf_ptr,p);

	f_q = compute_phi_alpha_LBFGS(crf_ptr,q);

	while ( b - a >= prism_epsilon ) {
		if ( f_p < f_q ) {
			b = q;
			q = p;
			p = b - tau * (b - a);
			f_q = f_p;
			f_p = compute_phi_alpha_LBFGS(crf_ptr,p);
		} else {
			a = p;
			p = q;
			q = a + tau * (b - a);
			f_p = f_q;
			f_q = compute_phi_alpha_LBFGS(crf_ptr,q);
		}
	}

	restore_current_params();

	return (a + b)/2.0;
}

/*-----[L-BFGS]-----------------------------------------------------------*/

static void initialize_LBFGS(void) {
	int i;
	SW_INS_PTR sw_ptr;

	for (i=0; i<occ_switch_tab_size; i++) {
		sw_ptr = occ_switches[i];
		while (sw_ptr!=NULL) {
			sw_ptr->LBFGS_s = (double *)MALLOC(10 * sizeof(double));
			sw_ptr->LBFGS_y = (double *)MALLOC(10 * sizeof(double));
			sw_ptr = sw_ptr->next;
		}
	}

	LBFGS_a = (double *)MALLOC(10 * sizeof(double));
	LBFGS_b = (double *)MALLOC(10 * sizeof(double));
	LBFGS_rho = (double *)MALLOC(10 * sizeof(double));

	LBFGS_index = 0;
}

static void clean_LBFGS(void) {
	int i;
	SW_INS_PTR sw_ptr;

	for (i=0; i<occ_switch_tab_size; i++) {
		sw_ptr = occ_switches[i];
		while (sw_ptr!=NULL) {
			FREE(sw_ptr->LBFGS_s);
			FREE(sw_ptr->LBFGS_y);
			sw_ptr = sw_ptr->next;
		}
	}

	FREE(LBFGS_a);
	FREE(LBFGS_b);
	FREE(LBFGS_rho);
}

static void compute_hessian(int iterate) {
	int i,j,m,index;
	SW_INS_PTR sw_ptr;
	double a,b;

	if (iterate<10) {
		m = iterate;
	} else {
		m = 10;
	}

	for (j=0; j<m; j++) {
		index = (LBFGS_index - j + 10) % 10;
		a = 0;
		for (i=0; i<occ_switch_tab_size; i++) {
			sw_ptr = occ_switches[i];
			while (sw_ptr!=NULL) {
				a += sw_ptr->LBFGS_s[index] * sw_ptr->LBFGS_q;
				sw_ptr = sw_ptr->next;
			}
		}
		LBFGS_a[index] = LBFGS_rho[index] * a;

		for (i=0; i<occ_switch_tab_size; i++) {
			sw_ptr = occ_switches[i];
			while (sw_ptr!=NULL) {
				sw_ptr->LBFGS_q -= LBFGS_a[index] * sw_ptr->LBFGS_y[index];
				sw_ptr = sw_ptr->next;
			}
		}
	}

	for (j=m-1; j>-1; j--) {
		index = (LBFGS_index - j + 10) % 10;
		b = 0;
		for (i=0; i<occ_switch_tab_size; i++) {
			sw_ptr = occ_switches[i];
			while (sw_ptr!=NULL) {
				b += sw_ptr->LBFGS_y[index] * sw_ptr->LBFGS_q;
				sw_ptr = sw_ptr->next;
			}
		}
		LBFGS_b[index] = LBFGS_rho[index] * b;

		for (i=0; i<occ_switch_tab_size; i++) {
			sw_ptr = occ_switches[i];
			while (sw_ptr!=NULL) {
				sw_ptr->LBFGS_q += (LBFGS_a[index] - LBFGS_b[index]) * sw_ptr->LBFGS_s[index];
				sw_ptr = sw_ptr->next;
			}
		}
	}

	LBFGS_index = ( LBFGS_index + 11) % 10;
}

static void restore_old_gradient(void) {
	int i;
	SW_INS_PTR sw_ptr;

	for (i=0; i<occ_switch_tab_size; i++) {
		sw_ptr = occ_switches[i];
		while (sw_ptr!=NULL) {
			sw_ptr->LBFGS_y[LBFGS_index] = (-1) * sw_ptr->gradient;
			sw_ptr = sw_ptr->next;
		}
	}
}

static void initialize_LBFGS_q(void) {
	int i;
	SW_INS_PTR sw_ptr;

	for (i=0; i<occ_switch_tab_size; i++) {
		sw_ptr = occ_switches[i];
		while (sw_ptr!=NULL) {
			sw_ptr->LBFGS_q = (-1) * sw_ptr->gradient;
			sw_ptr = sw_ptr->next;
		}
	}
}

/* compute LBFGS_y, LBFGS_rho, and LBFGS_a */
static void compute_LBFGS_y_rho(void) {
	int i;
	SW_INS_PTR sw_ptr;
	double rho,a;

	rho = 0;
	a = 0;

	for (i=0; i<occ_switch_tab_size; i++) {
		sw_ptr = occ_switches[i];
		while (sw_ptr!=NULL) {
			sw_ptr->LBFGS_q = (-1) * sw_ptr->gradient;
			sw_ptr->LBFGS_y[LBFGS_index] = sw_ptr->LBFGS_q - sw_ptr->LBFGS_y[LBFGS_index];
			rho += sw_ptr->LBFGS_s[LBFGS_index] * sw_ptr->LBFGS_y[LBFGS_index];
			sw_ptr = sw_ptr->next;
		}
	}

	LBFGS_rho[LBFGS_index] = 1 / rho;
}

static int update_lambdas_LBFGS(double tmp_epsilon) {
	int i;
	SW_INS_PTR sw_ptr;

	for (i=0; i<occ_switch_tab_size; i++) {
		sw_ptr = occ_switches[i];
		if (sw_ptr->fixed > 0) continue;
		while (sw_ptr!=NULL) {
			if (sw_ptr->fixed == 0) {
				sw_ptr->LBFGS_s[LBFGS_index] = tmp_epsilon * (-1) * sw_ptr->LBFGS_q;
				sw_ptr->inside += sw_ptr->LBFGS_s[LBFGS_index];
			}
			sw_ptr = sw_ptr->next;
		}
	}

	return BP_TRUE;
}

/*------------------------------------------------------------------------*/

void config_crf(CRF_ENG_PTR crf_ptr) {
	if (log_scale) {
		crf_ptr->compute_feature      = compute_feature_scaling_log_exp;
		crf_ptr->compute_crf_probs     = compute_crf_probs_scaling_log_exp;
		crf_ptr->compute_likelihood   = compute_log_likelihood_scaling_log_exp;
		crf_ptr->compute_gradient     = compute_gradient_scaling_log_exp;
	} else {
		crf_ptr->compute_feature      = compute_feature_scaling_none;
		crf_ptr->compute_crf_probs     = compute_crf_probs_scaling_none;
		crf_ptr->compute_likelihood   = compute_log_likelihood_scaling_none;
		crf_ptr->compute_gradient     = compute_gradient_scaling_none;
	}

	if (crf_learn_mode == 1) {
		crf_ptr->update_lambdas = update_lambdas_LBFGS;
	} else {
		crf_ptr->update_lambdas = update_lambdas;
	}
}

/*------------------------------------------------------------------------*/

/* main loop */
static int run_grd(CRF_ENG_PTR crf_ptr) {
	int r,iterate,old_valid,converged,conv_time,saved = 0;
	double likelihood,old_likelihood = 0.0;
	double crf_max_iterate = 0.0;
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

	if (max_iterate == -1) {
		crf_max_iterate = DEFAULT_MAX_ITERATE;
	} else if (max_iterate >= +1) {
		crf_max_iterate = max_iterate;
	}

	for (r = 0; r < num_restart; r++) {
		SHOW_PROGRESS_HEAD("#crf-iters", r);

		initialize_crf_count();
		initialize_lambdas();
		initialize_visited_flags();

		old_valid = 0;
		iterate = 0;
		tmp_epsilon = crf_epsilon;

		LBFGS_index = 0;
		conv_time = 0;

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

			if (crf_learning_rate == 1) { // annealing
				tmp_epsilon = (annealing_weight / (annealing_weight + iterate)) * crf_epsilon;
			} else if (crf_learning_rate == 2) { // line-search(backtrack)
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

int pc_crf_prepare_4(void) {
	TERM  p_fact_list;
	int   size;

	p_fact_list        = bpx_get_call_arg(1,4);
	size               = bpx_get_integer(bpx_get_call_arg(2,4));
	num_goals          = bpx_get_integer(bpx_get_call_arg(3,4));
	failure_root_index = bpx_get_integer(bpx_get_call_arg(4,4));

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

	return BP_TRUE;
}

int pc_prism_grd_2(void) {
	struct CRF_Engine crf_eng;

	RET_ON_ERR(run_grd(&crf_eng));

	return
	    bpx_unify(bpx_get_call_arg(1,2), bpx_build_integer(crf_eng.iterate)) &&
	    bpx_unify(bpx_get_call_arg(2,2), bpx_build_float(crf_eng.likelihood));
}

/*---[fprob]--------------------------------------------------------------*/

int pc_compute_feature_2(void) {
	int gid;
	double prob;
	EG_NODE_PTR eg_ptr;

	gid = bpx_get_integer(bpx_get_call_arg(1,2));

	initialize_egraph_index();
	alloc_sorted_egraph(1);
	RET_ON_ERR(sort_one_egraph(gid, 0, 1));

	if (verb_graph) {
		print_egraph(0, PRINT_NEUTRAL);
	}

	eg_ptr = expl_graph[gid];

	initialize_weights();

	if (log_scale) {
		RET_ON_ERR(compute_feature_scaling_log_exp());
		prob = eg_ptr->inside;
	} else {
		RET_ON_ERR(compute_feature_scaling_none());
		prob = eg_ptr->inside;
	}

	return bpx_unify(bpx_get_call_arg(2,2), bpx_build_float(prob));
}

int pc_compute_fprobf_1(void) {
	int prmode;

	prmode = bpx_get_integer(bpx_get_call_arg(1,1));

	failure_root_index = -1;

	initialize_weights();

	/* [31 Mar 2008, by yuizumi]
	 * compute_outside_scaling_*() needs to be called because
	 * eg_ptr->outside computed by compute_expectation_scaling_*()
	 * is different from the outside probability.
	 */
	if (log_scale) {
		RET_ON_ERR(compute_feature_scaling_log_exp());
		if (prmode != 1) {
			RET_ON_ERR(compute_expectation_scaling_log_exp());
			RET_ON_ERR(compute_outside_scaling_log_exp());
		}
	} else {
		RET_ON_ERR(compute_feature_scaling_none());
		if (prmode != 1) {
			RET_ON_ERR(compute_expectation_scaling_none());
			RET_ON_ERR(compute_outside_scaling_none());
		}
	}

	return BP_TRUE;
}

int pc_get_snode_feature_3(void) {
	int idx = bpx_get_integer(bpx_get_call_arg(1,3));
	double val = switch_instances[idx]->inside_h;
	double lambda = switch_instances[idx]->inside;

	return bpx_unify(bpx_get_call_arg(2,3),bpx_build_float(val))
	       && bpx_unify(bpx_get_call_arg(3,3),bpx_build_float(lambda));
}
