#include "up/mcmc.h"

SW_COUNT_PTR **mh_state_sw_count = NULL;
SW_COUNT_PTR *prop_state_sw_count = NULL;
SW_COUNT_PTR **smp_val_sw_count = NULL;
int *mh_state_sw_count_size = NULL;
int prop_state_sw_count_size = 0;
int *smp_val_sw_count_size = NULL;

int *mh_switch_table = NULL;

int num_observed_goals = 0;
int *observed_goals = NULL;
int num_samples = 0;

double *stored_params = NULL;
double *stored_hparams = NULL;
int num_stored_params = 0;

double logV0 = 0.0; /* logV0 = log(P(G1,...GN|theta*)P(theta*|As)) */

int trace_sw_id = -1;
int trace_step = -1;
int postparam_sw_id = -1;
int postparam_step = -1;

static int *diff_sw_ins_ids = NULL;
static int diff_sw_ins_ids_size = 0;

static int *diff_switch_table = NULL;

static EG_NODE_PTR **mh_sorted_egraphs = NULL;
static int *mh_sorted_egraph_sizes = NULL;
static int max_mh_sorted_egraph_size = INIT_MAX_EGRAPH_SIZE;
static int mh_index_to_sort = 0;

static SW_INS_PTR *mh_occ_switches = NULL;
static int mh_occ_switch_tab_size = 0;

/*--------------------<< Metropolis-Hastings sampler for PRISM >>--------------------*/
/*                                                                                   */
/* P(E1,..,EN,G1,..,GN,Theta)                                                        */
/*   = P(Theta)P(E1,..,EN|Theta)P(G1,..,GN|E1,..,EN) (<= generative model)           */
/* P(E1,..,EN,Theta|G1,..,GN) (<= posterior distribution)                            */
/*   = P(Theta|E1,..,EN)P(E1,..,EN|G1,..,GN)                                         */
/* Given G1,...,GN, sample expls <E1,..,EN> ~ P(E1,..,EN|G1,..,GN)                   */
/* by M-H sampling with propsal transition at each E_k                               */
/* Sample E_k ~ P(E_k| E_{all-k},G1,..,GN) for MH sampling                           */
/* (0) We are in state E_all = <E_1,..,E_N>                                          */
/* (1) Randomly choose one of k in [1,N]                                             */
/* (2) Compute for every i of msw(i,v),                                              */
/*      X_{i,v} = sigma_{i,v}(E_all)- sigma_{i,v}(E_k) + Alpha_{i,v}                 */
/*      Theta*_{i,v} = X_{i,v}/sum_{v}X_{i,v} (mean of Dirichlet pdf)                */
/*      Set Theta*_{i,v} to msw(i,v)'s hyper parameter                               */
/* (3) Sample a proposal E_k' ~ P(Expl | G_k, Theta*)                                */
/* (4) Compute Accept(E_k,E_k') =                                                    */
/*                                                                                   */
/*               P(E_k'| G_k,E_{all-k})  P(E_k | G_k,Theta*)                         */
/*     min{ 1,  ----------------------- --------------------- }                      */
/*               P(E_k | G_k,E_{all-k})  P(E_k' | G_k,Theta*)                        */
/*                                                                                   */
/* (5) Accept E_k' with prob. Accept(E_k,E_k')                                       */
/*-----------------------------------------------------------------------------------*/

static void count_sw_ins_id(EG_PATH_PTR ptr, int index) {
	int i,len,sw_ins_id;

	if (ptr != NULL) {
		len = ptr->sws_len;
		for (i=0; i<len; i++) {
			sw_ins_id = ptr->sws[i]->id;
			mh_switch_table[sw_ins_id]++;
			if (mh_switch_table[sw_ins_id] == 1) {
				mh_state_sw_count_size[index]++;
			}
			ptr->sws[i]->count++;
		}
		len = ptr->children_len;
		for (i=0; i<len; i++) {
			count_sw_ins_id(ptr->children[i]->max_path,index);
		}
	}
}

static void alloc_mh_state_sw_count(void) {
	int i;

	mh_state_sw_count = (SW_COUNT_PTR **)MALLOC(num_observed_goals * sizeof(SW_COUNT_PTR*));

	for (i=0; i<num_observed_goals; i++) {
		mh_state_sw_count[i] = NULL;
	}
}

static void alloc_one_mh_state_sw_count(int index) {
	int i,j=0;

	mh_state_sw_count[index] = (SW_COUNT_PTR *)MALLOC(mh_state_sw_count_size[index] * sizeof(SW_COUNT_PTR));

	for (i=0; i<sw_ins_tab_size; i++) {
		if (mh_switch_table[i]!=0) {
			mh_state_sw_count[index][j] = (SW_COUNT_PTR)MALLOC(sizeof(struct SwitchCount));
			mh_state_sw_count[index][j]->sw_ins_id = i;
			mh_state_sw_count[index][j]->count = mh_switch_table[i];
			j++;
		}
	}
}

void init_mh_sw_tab(void) {
	int i;

	for (i=0; i<sw_ins_tab_size; i++) {
		mh_switch_table[i] = 0;
	}
}

static void initialize_diff_sw_tab(void) {
	int i;

	for (i=0; i<sw_ins_tab_size; i++) {
		diff_switch_table[i] = 0;
	}
}

void alloc_mh_switch_table(void) {
	mh_switch_table = (int *)MALLOC(sw_ins_tab_size * sizeof(int));
}

void clean_mh_switch_table(void) {
	FREE(mh_switch_table);
}

static void alloc_diff_switch_table(void) {
	diff_switch_table = (int *)MALLOC(sw_ins_tab_size * sizeof(int));
}

static void clean_diff_switch_table(void) {
	FREE(diff_switch_table);
}

static void initialize_all_count(void) {
	int i;
	SW_INS_PTR sw_ptr;

	for (i=0; i<sw_tab_size; i++) {
		sw_ptr = switches[i];
		while (sw_ptr != NULL) {
			sw_ptr->count = 0;
			sw_ptr = sw_ptr->next;
		}
	}
}

static void set_init_state(void) {
	int i;

	alloc_mh_state_sw_count();

	mh_state_sw_count_size = (int *)MALLOC(num_observed_goals * sizeof(int));

	for (i=0; i<num_observed_goals; i++) {
		init_mh_sw_tab();
		mh_state_sw_count_size[i] = 0;
		count_sw_ins_id(expl_graph[observed_goals[i]]->max_path,i);
		if (mh_state_sw_count_size[i]!=0) alloc_one_mh_state_sw_count(i);
	}
}

void clean_mh_state_sw_count(void) {
	int i,j;

	if (mh_state_sw_count != NULL) {
		for (i=0; i<num_observed_goals; i++) {
			for (j=0; j<mh_state_sw_count_size[i]; j++) {
				FREE(mh_state_sw_count[i][j]);
			}
			FREE(mh_state_sw_count[i]);
		}
		FREE(mh_state_sw_count);
		FREE(mh_state_sw_count_size);
	}
}

static void add_all_counts(int index) {
	int i,sw_ins_id;

	for (i=0; i<mh_state_sw_count_size[index]; i++) {
		sw_ins_id = mh_state_sw_count[index][i]->sw_ins_id;
		switch_instances[sw_ins_id]->count += mh_state_sw_count[index][i]->count;
	}
}

static void add_proposal_counts(void) {
	int i,sw_ins_id;

	for (i=0; i<prop_state_sw_count_size; i++) {
		sw_ins_id = prop_state_sw_count[i]->sw_ins_id;
		switch_instances[sw_ins_id]->count += prop_state_sw_count[i]->count;
	}
}

static void subtract_all_counts(int index) {
	int i,sw_ins_id;

	for (i=0; i<mh_state_sw_count_size[index]; i++) {
		sw_ins_id = mh_state_sw_count[index][i]->sw_ins_id;
		switch_instances[sw_ins_id]->count -= mh_state_sw_count[index][i]->count;
	}
}

static void subtract_proposal_counts(void) {
	int i,sw_ins_id;

	for (i=0; i<prop_state_sw_count_size; i++) {
		sw_ins_id = prop_state_sw_count[i]->sw_ins_id;
		switch_instances[sw_ins_id]->count -= prop_state_sw_count[i]->count;
	}
}

/* set Dirichlet's means to params */
static void set_means_to_params(void) {
	SW_INS_PTR sw_ptr;
	int i;
	double sum;

	for (i=0; i<occ_switch_tab_size; i++) {
		sw_ptr = occ_switches[i];
		sum = 0.0;
		while (sw_ptr != NULL) {
			sum += sw_ptr->smooth_prolog + 1.0;
			sum += sw_ptr->count;
			sw_ptr = sw_ptr->next;
		}
		sw_ptr = occ_switches[i];
		while (sw_ptr != NULL) {
			sw_ptr->inside = (sw_ptr->smooth_prolog + 1.0 + sw_ptr->count)/sum;
			sw_ptr = sw_ptr->next;
		}
	}
}

static void sample_node_aux(EG_NODE_PTR eg_ptr) {
	int len,i,j,sw_ins_id,num_path=0;
	double rand,sum = 0.0,maxP = 0.0;
	EG_PATH_PTR path_ptr,path_ptr2;
	double *norm_path_probs;

	path_ptr = eg_ptr->path_ptr;

	if (path_ptr != NULL) {
		if (path_ptr->next != NULL) {
			path_ptr2 = path_ptr;
			while (path_ptr2 != NULL) {
				num_path++;
				path_ptr2 = path_ptr2->next;
			}
			norm_path_probs = (double *)MALLOC(num_path * sizeof(double));

			if (log_scale) {
				path_ptr2 = path_ptr;
				while (path_ptr2 != NULL) {
					if (maxP<path_ptr2->inside) {
						maxP = path_ptr2->inside;
					}
					path_ptr2 = path_ptr2->next;
				}
				path_ptr2 = path_ptr;
				for (i=0; i<num_path; i++) {
					norm_path_probs[i] = exp(path_ptr2->inside - maxP);
					sum += norm_path_probs[i];
					path_ptr2 = path_ptr2->next;
				}
			} else {
				path_ptr2 = path_ptr;
				for (i=0; i<num_path; i++) {
					norm_path_probs[i] = path_ptr2->inside;
					sum += norm_path_probs[i];
					path_ptr2 = path_ptr2->next;
				}
			}
			for (i=0; i<num_path; i++) {
				norm_path_probs[i] /= sum;
			}
			rand = random_float();
			i = num_path - 1;
			while (norm_path_probs[i] <= rand && path_ptr->next != NULL) {
				rand -= norm_path_probs[i];
				i--;
			}
			for (j=0; j<i; j++) {
				path_ptr = path_ptr->next;
			}
			FREE(norm_path_probs);
		}
		len = path_ptr->sws_len;
		for (i=0; i<len; i++) {
			sw_ins_id = path_ptr->sws[i]->id;
			mh_switch_table[sw_ins_id]++;
			if (mh_switch_table[sw_ins_id] == 1) {
				prop_state_sw_count_size++;
			}
		}
		len = path_ptr->children_len;
		for (i=0; i<len; i++) {
			sample_node_aux(path_ptr->children[i]);
		}
	}
}

static void sample_node(int g_id) {
	int i,j = 0;
	EG_NODE_PTR eg_ptr;

	eg_ptr = expl_graph[g_id];

	prop_state_sw_count_size = 0;
	init_mh_sw_tab();

	sample_node_aux(eg_ptr);

	prop_state_sw_count = (SW_COUNT_PTR *)MALLOC(prop_state_sw_count_size * sizeof(SW_COUNT_PTR));

	for (i=0; i<sw_ins_tab_size; i++) {
		if (mh_switch_table[i]!=0) {
			prop_state_sw_count[j] = (SW_COUNT_PTR)MALLOC(sizeof(struct SwitchCount));
			prop_state_sw_count[j]->sw_ins_id = i;
			prop_state_sw_count[j]->count = mh_switch_table[i];
			j++;
		}
	}
}

static void clean_proposal_state(void) {
	int i;

	for (i=0; i<prop_state_sw_count_size; i++) {
		FREE(prop_state_sw_count[i]);
	}

	FREE(prop_state_sw_count);
}

static void clean_diff(void) {
	FREE(diff_sw_ins_ids);
}

static int set_diff(int index) {
	int i,sw_ins_id,j = 0;

	for (i=0; i<mh_state_sw_count_size[index]; i++) {
		sw_ins_id = mh_state_sw_count[index][i]->sw_ins_id;
		mh_switch_table[sw_ins_id] -= mh_state_sw_count[index][i]->count;
	}

	diff_sw_ins_ids_size = 0;

	for (i=0; i<sw_ins_tab_size; i++) {
		if (mh_switch_table[i]!=0) {
			diff_sw_ins_ids_size++;
		}
	}

	if (diff_sw_ins_ids_size==0) {
		return 0;
	} else {
		diff_sw_ins_ids = (int *)MALLOC(diff_sw_ins_ids_size * sizeof(int));

		for (i=0; i<sw_ins_tab_size; i++) {
			if (mh_switch_table[i]!=0) {
				diff_sw_ins_ids[j] = i;
				j++;
			}
		}
		return 1;
	}
}

static double log_ratio_R1(int index) {
	int i,sw_ins_id;
	double logR1,logP1 = 0.0,logP2 = 0.0;

	for (i=0; i<mh_state_sw_count_size[index]; i++) {
		sw_ins_id = mh_state_sw_count[index][i]->sw_ins_id;
		logP1 += log(switch_instances[sw_ins_id]->inside) * mh_state_sw_count[index][i]->count;
	}
	for (i=0; i<prop_state_sw_count_size; i++) {
		sw_ins_id = prop_state_sw_count[i]->sw_ins_id;
		logP2 += log(switch_instances[sw_ins_id]->inside) * prop_state_sw_count[i]->count;
	}

	logR1 = logP1 - logP2;

	return logR1;
}

double log_dirichlet_Z(int sw_id) {
	double logD = 0.0,logE = 0.0,logZ = 0.0;
	SW_INS_PTR sw_ptr;

	sw_ptr = switch_instances[sw_id];

	while (sw_ptr != NULL) {
		logE += lngamma(sw_ptr->smooth_prolog + 1.0 + sw_ptr->count);
		logD += (sw_ptr->smooth_prolog + 1.0 + sw_ptr->count);
		sw_ptr = sw_ptr->next;
	}

	logD = lngamma(logD);
	logZ = logE - logD;

	return logZ;
}

static double log_prod_i(void) {
	int i,j,sw_ins_id,occ_flag = 0;
	double log_prod_is = 0.0;
	SW_INS_PTR sw_ptr;

	for (i=0; i<occ_switch_tab_size; i++) {
		sw_ptr = occ_switches[i];
		while (sw_ptr != NULL) {
			sw_ins_id = sw_ptr->id;
			for (j=0; j<diff_sw_ins_ids_size; j++) {
				if (sw_ins_id == diff_sw_ins_ids[j]) {
					occ_flag = 1;
					break;
				}
			}
			if (occ_flag == 1) {
				log_prod_is += log_dirichlet_Z(occ_switches[i]->id);
				occ_flag = 0;
				break;
			}
			sw_ptr = sw_ptr->next;
		}
	}
	return log_prod_is;
}

static double log_prod_i_all(void) {
	int i;
	double log_prod_is = 0.0;

	for (i=0; i<occ_switch_tab_size; i++) {
		log_prod_is += log_dirichlet_Z(occ_switches[i]->id);
	}

	return log_prod_is;
}

static void change_state(int index) {
	int tmp;
	SW_COUNT_PTR *tmp_ptr;

	tmp = mh_state_sw_count_size[index];
	mh_state_sw_count_size[index] = prop_state_sw_count_size;
	prop_state_sw_count_size = tmp;

	tmp_ptr = mh_state_sw_count[index];
	mh_state_sw_count[index] = prop_state_sw_count;
	prop_state_sw_count = tmp_ptr;
}

static void alloc_sampled_values(void) {
	int i;

	smp_val_sw_count = (SW_COUNT_PTR **)MALLOC(num_samples * sizeof(SW_COUNT_PTR *));
	smp_val_sw_count_size = (int *)MALLOC(num_samples * sizeof(int));

	for (i=0; i<num_samples; i++) {
		smp_val_sw_count[i] = NULL;
		smp_val_sw_count_size[i] = 0;
	}
}

void clean_sampled_values(void) {
	int i,j;

	if (smp_val_sw_count != NULL) {
		for (i=0; i<num_samples; i++) {
			for (j=0; j<smp_val_sw_count_size[i]; j++) {
				FREE(smp_val_sw_count[i][j]);
			}
			FREE(smp_val_sw_count[i]);
		}
		FREE(smp_val_sw_count);
		FREE(smp_val_sw_count_size);
	}
}

static void store_diff_sw_tab(void) {
	int i;

	for (i=0; i<sw_ins_tab_size; i++) {
		diff_switch_table[i] += mh_switch_table[i];
	}
}

static void store_sampled_value(int index) {
	int i,j=0;
	int diff_num_occ_sw_ins=0;

	for (i=0; i<sw_ins_tab_size; i++) {
		if (diff_switch_table[i]!=0) {
			diff_num_occ_sw_ins++;
		}
	}

	smp_val_sw_count_size[index] = diff_num_occ_sw_ins;

	if (smp_val_sw_count_size[index]!=0) {
		smp_val_sw_count[index] = (SW_COUNT_PTR *)MALLOC(smp_val_sw_count_size[index] * sizeof(SW_COUNT_PTR));

		for (i=0; i<sw_ins_tab_size; i++) {
			if (diff_switch_table[i]!=0) {
				smp_val_sw_count[index][j] = (SW_COUNT_PTR)MALLOC(sizeof(struct SwitchCount));
				smp_val_sw_count[index][j]->sw_ins_id = i;
				smp_val_sw_count[index][j]->count = diff_switch_table[i];
				j++;
			}
		}
		initialize_diff_sw_tab();
	}
}

double log_dirichlet_pdf_value(void) {
	int i;
	double sumXs,sumAs,logE,logD;
	double val=0.0;
	SW_INS_PTR sw_ptr;

	for (i=0; i<mh_occ_switch_tab_size; i++) {
		sw_ptr = mh_occ_switches[i];
		if (sw_ptr->next != NULL) {
			sumXs = 0.0;
			sumAs = 0.0;
			logE = 0.0;
			while (sw_ptr != NULL) {
				sumXs += ((sw_ptr->smooth_prolog + sw_ptr->count) * log(sw_ptr->inside));
				sumAs += (sw_ptr->smooth_prolog + 1.0 + sw_ptr->count);
				logE += lngamma(sw_ptr->smooth_prolog + 1.0 + sw_ptr->count);
				sw_ptr = sw_ptr->next;
			}
			logD = lngamma(sumAs);
			val += (logD - logE + sumXs);
		}
	}
	return val;
}

static double ini_sampled_value_learn(void) {
	int i;
	double x=0.0;

	if (log_scale) {
		RET_ON_ERR(compute_inside_scaling_log_exp());
		for (i=0; i<num_observed_goals; i++) {
			x += expl_graph[observed_goals[i]]->inside;
		}
	} else {
		RET_ON_ERR(compute_inside_scaling_none());
		for (i=0; i<num_observed_goals; i++) {
			x += log(expl_graph[observed_goals[i]]->inside);
		}
	}
	return x + log_dirichlet_pdf_value();
}

static void preserve_current_params(void) {
	int i;

	num_stored_params = sw_ins_tab_size;
	stored_params = (double *)MALLOC(num_stored_params * sizeof(double));
	stored_hparams = (double *)MALLOC(num_stored_params * sizeof(double));

	for (i=0; i<num_stored_params; i++) {
		stored_params[i] = switch_instances[i]->inside;
		stored_hparams[i] = switch_instances[i]->smooth_prolog;
	}
}

static void clean_mh_sorted_egraphs(void) {
	int i;

	for (i=0; i<num_observed_goals; i++) {
		FREE(mh_sorted_egraphs[i]);
	}
	FREE(mh_sorted_egraphs);
	FREE(mh_sorted_egraph_sizes);
}

static void alloc_mh_sorted_egraphs(void) {
	int i;

	mh_sorted_egraphs = (EG_NODE_PTR **)MALLOC(num_observed_goals * sizeof(EG_NODE_PTR *));
	mh_sorted_egraph_sizes = (int *)MALLOC(num_observed_goals * sizeof(int));

	for (i=0; i<num_observed_goals; i++) {
		mh_sorted_egraphs[i] = NULL;
		mh_sorted_egraph_sizes[i] = 0;
	}
}

static void alloc_mh_sorted_egraph(int goal_index) {
	max_mh_sorted_egraph_size = INIT_MAX_EGRAPH_SIZE;
	mh_sorted_egraphs[goal_index] = (EG_NODE_PTR *)MALLOC(sizeof(EG_NODE_PTR) * max_mh_sorted_egraph_size);
}

static void expand_mh_sorted_egraph(int goal_index, int req_sorted_egraph_size) {
	if (req_sorted_egraph_size > max_mh_sorted_egraph_size) {
		while (req_sorted_egraph_size > max_mh_sorted_egraph_size) {
			if (max_mh_sorted_egraph_size > MAX_EGRAPH_SIZE_EXPAND_LIMIT)
				max_mh_sorted_egraph_size += MAX_EGRAPH_SIZE_EXPAND_LIMIT;
			else
				max_mh_sorted_egraph_size *= 2;
		}
		mh_sorted_egraphs[goal_index] =
		    (EG_NODE_PTR *)
		    REALLOC(mh_sorted_egraphs[goal_index],
		            max_mh_sorted_egraph_size * sizeof(EG_NODE_PTR));
	}
}

static void mh_topological_sort(int goal_index, int node_id) {
	EG_PATH_PTR path_ptr;
	EG_NODE_PTR *children;
	int k,len;
	EG_NODE_PTR child_ptr;

	expl_graph[node_id]->visited = 2;
	UPDATE_MIN_MAX_NODE_NOS(node_id);

	path_ptr = expl_graph[node_id]->path_ptr;
	while (path_ptr != NULL) {
		children = path_ptr->children;
		len = path_ptr->children_len;
		for (k = 0; k < len; k++) {
			child_ptr = children[k];

			if (child_ptr->visited == 0) {
				mh_topological_sort(goal_index,child_ptr->id);
				expand_mh_sorted_egraph(goal_index,mh_index_to_sort + 1);
				mh_sorted_egraphs[goal_index][mh_index_to_sort++] = child_ptr;
			}
		}
		path_ptr = path_ptr->next;
	}
	expl_graph[node_id]->visited = 1;
}

static void initialize_all_visited_flags(void) {
	int i;

	for (i=0; i<prism_goal_count(); i++) {
		expl_graph[i]->visited = 0;
	}
}

/* compute the inside probability for one goal */
static int mh_compute_inside(int goal_index) {
	int i,k,u;
	double sum,this_path_inside;
	double sum_rest,first_path_inside;
	EG_NODE_PTR eg_ptr;
	EG_PATH_PTR path_ptr;

	if (mh_sorted_egraphs[goal_index] == NULL) {
		alloc_mh_sorted_egraph(goal_index);
		INIT_MIN_MAX_NODE_NOS;
		mh_index_to_sort = 0;
		max_mh_sorted_egraph_size = INIT_MAX_EGRAPH_SIZE;

		mh_topological_sort(goal_index,observed_goals[goal_index]);

		expand_mh_sorted_egraph(goal_index,mh_index_to_sort + 1);
		mh_sorted_egraphs[goal_index][mh_index_to_sort] = expl_graph[observed_goals[goal_index]];

		mh_index_to_sort++;
		mh_sorted_egraph_sizes[goal_index] = mh_index_to_sort;
		INIT_VISITED_FLAGS;
	}

	if (log_scale) {
		for (i = 0; i < mh_sorted_egraph_sizes[goal_index]; i++) {
			eg_ptr = mh_sorted_egraphs[goal_index][i];
			path_ptr = eg_ptr->path_ptr;
			if (path_ptr == NULL) {
				sum = 0.0; /* path_ptr should not be NULL; but it happens */
			} else {
				sum_rest = 0.0;
				u = 0;
				while (path_ptr != NULL) {
					this_path_inside = 0.0;
					for (k = 0; k < path_ptr->children_len; k++) {
						this_path_inside += path_ptr->children[k]->inside;
					}
					for (k = 0; k < path_ptr->sws_len; k++) {
						this_path_inside += log(path_ptr->sws[k]->inside);
					}
					path_ptr->inside = this_path_inside;
					if (u == 0) {
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
					u++;
				}
				sum = first_path_inside + log(sum_rest);
			}

			eg_ptr->inside = sum;
		}
	} else {
		for (i = 0; i < mh_sorted_egraph_sizes[goal_index]; i++) {
			eg_ptr = mh_sorted_egraphs[goal_index][i];

			sum = 0.0;
			path_ptr = eg_ptr->path_ptr;
			if (path_ptr == NULL)
				sum = 1.0; /* path_ptr should not be NULL; but it happens */
			while (path_ptr != NULL) {
				this_path_inside = 1.0;
				for (k = 0; k < path_ptr->children_len; k++) {
					this_path_inside *= path_ptr->children[k]->inside;
				}
				for (k = 0; k < path_ptr->sws_len; k++) {
					this_path_inside *= path_ptr->sws[k]->inside;
				}
				path_ptr->inside = this_path_inside;
				sum += this_path_inside;
				path_ptr = path_ptr->next;
			}

			eg_ptr->inside = sum;
		}
	}

	return BP_TRUE;
}

/* print out the post_param information */
static void print_dirichlet_pdf(int iterate) {
	int i,size=0;
	double *as,*ps;
	SW_INS_PTR sw_ptr;

	if (iterate!=-1) {
		prism_printf("%6d  ",iterate);
	}

	sw_ptr = switches[postparam_sw_id];

	while (sw_ptr != NULL) {
		size++;
		sw_ptr = sw_ptr->next;
	}

	as = (double *)MALLOC(size * sizeof(double));
	ps = (double *)MALLOC(size * sizeof(double));

	sw_ptr = switches[postparam_sw_id];

	for (i=0; i<size; i++) {
		as[i] = sw_ptr->smooth_prolog + 1.0 + sw_ptr->count;
		sw_ptr = sw_ptr->next;
	}

	sample_dirichlet(as,size,ps);

	for (i=0; i<size; i++) {
		prism_printf(" %.3f",ps[i]);
	}

	FREE(as);
	FREE(ps);
}

/* print out the trace */
static void print_mh_info(int iterate, int goal_index, double ratio,
                          double u, int diff, int figure) {
	double logML,acceptP;
	SW_INS_PTR sw_ptr;

	logML = log_prod_i_all();

	prism_printf("%6d  %7.3f ",iterate,logML);

	sw_ptr = switches[trace_sw_id];
	while (sw_ptr != NULL) {
		prism_printf(" %d",sw_ptr->count);
		sw_ptr = sw_ptr->next;
	}

	prism_printf("  : #G = %*d, ",figure,goal_index+1);

	acceptP = (ratio > 1.0) ? 1.0 : ratio;

	if (diff == 0) { /* state unchanged */
		prism_printf(" -             ");
	} else {         /* state changed */
		if (acceptP >= u) {  /* accepted */
			prism_printf(" * ");
		} else {             /* rejected */
			prism_printf(" - ");
		}
		prism_printf(" R = %.3f  ",ratio);
	}
}

void release_mh_occ_switches(void) {
	FREE(mh_occ_switches);
	mh_occ_switch_tab_size = 0;
}

static void alloc_mh_occ_switches(void) {
	int i;

	mh_occ_switch_tab_size = occ_switch_tab_size;
	mh_occ_switches = (SW_INS_PTR *)MALLOC(mh_occ_switch_tab_size * sizeof(SW_INS_PTR));

	for (i=0; i<mh_occ_switch_tab_size; i++) {
		mh_occ_switches[i] = occ_switches[i];
	}
}

/* Metroporis-Hastings sampler */
int loop_mh(int end, int burn_in, int skip) {
	int iterate,iterate_max,k,diff,sample_index=0;
	double logR1,logR2,logR2D,logR2E,ratio,acceptP,u;
	int figure=0;

	alloc_occ_switches();

	alloc_mh_switch_table();
	initialize_all_count();

	/* save the parameters learned by VB */
	preserve_current_params();

	/* copy occ_switches for estimation of log marginal likelihood */
	alloc_mh_occ_switches();

	/* logV0: used in estimated marginal likelihood part */
	logV0 = ini_sampled_value_learn();

	/* we use the viterbi answer by VB as the initial state */
	compute_max();
	set_init_state();

	num_samples = (int)((end - burn_in) / skip);
	iterate_max = end - (end - burn_in) % skip;

	alloc_sampled_values();
	alloc_diff_switch_table();
	initialize_diff_sw_tab();

	alloc_mh_sorted_egraphs();
	initialize_all_visited_flags();

	if (trace_sw_id != 1) figure = log10(num_observed_goals) + 1;

	SHOW_MCMC_PROGRESS_HEAD("#visited");

	/* main loop of M-H sampler*/
	for (iterate = 0; iterate < iterate_max; iterate++) {
		if (CTRLC_PRESSED) {
			SHOW_MCMC_PROGRESS_INTR();
			RET_ERR(err_ctrl_c_pressed);
		}

		/* pick up the k-th goal randomly and sample a new explanation */
		k = random_int(num_observed_goals);
		subtract_all_counts(k);
		set_means_to_params();
		RET_ON_ERR(mh_compute_inside(k));
		sample_node(observed_goals[k]);

		/* computed the difference between the current and the new */
		diff = set_diff(k);

		/* compute acceptance_rate(acceptP) */
		if (diff == 0) {
			logR1 = 0.0;
		} else {
			logR1 = log_ratio_R1(k);
		}

		add_all_counts(k);

		if (diff == 0) {
			logR2 = 0.0;
		} else {
			logR2D = log_prod_i();
			subtract_all_counts(k);
			add_proposal_counts();
			logR2E = log_prod_i();
			logR2  = logR2E - logR2D;
		}

		ratio = exp(logR1 + logR2);
		acceptP = (ratio > 1.0) ? 1.0 : ratio;

		u = random_float();

		if (acceptP < u) {  /* rejected */
			subtract_proposal_counts();  /* restore the state */
			add_all_counts(k);
			if (iterate >= burn_in && iterate < iterate_max &&
			        (iterate - burn_in + 1) % skip == 0) {
				store_sampled_value(sample_index);  /* save the difference */
			}
		} else {
			if (diff == 1) { /* accepted */
				/* replace mh_state_sw_count[k] with the switch counts of
				 * the new state (explanation)
				 */
				change_state(k);
				if (iterate >= burn_in && iterate < iterate_max) {
					store_diff_sw_tab();
					if ((iterate - burn_in + 1) % skip == 0) {
						/* save the difference of switch counts */
						store_sampled_value(sample_index);
					}
				}
			} else {
				if (iterate >= burn_in && iterate < iterate_max &&
				        (iterate - burn_in + 1) % skip == 0) {
					/* save the difference of switch counts */
					store_sampled_value(sample_index);
				}
			}
		}
		if (iterate >= burn_in && iterate < iterate_max &&
		        (iterate - burn_in + 1) % skip == 0) {
			sample_index++;
		}

		clean_proposal_state();
		if (diff == 1) {
			clean_diff();
		}

		/* output the trace or the post_param information */
		if (trace_sw_id != -1 && trace_step <= iterate + 1) {
			print_mh_info(iterate,k,ratio,u,diff,figure);
			if (postparam_sw_id != -1 && postparam_step <= iterate) {
				print_dirichlet_pdf(-1);
			}
			prism_printf("\n");
		} else if (postparam_sw_id != -1 && postparam_step <= iterate) {
			print_dirichlet_pdf(iterate);
			prism_printf("\n");
		}

		SHOW_MCMC_PROGRESS(iterate);
	}

	SHOW_MCMC_PROGRESS_TAIL(iterate);

	clean_mh_sorted_egraphs();
	clean_mh_switch_table();
	clean_diff_switch_table();
	release_occ_switches();
	release_num_sw_vals();

	return BP_TRUE;
}
