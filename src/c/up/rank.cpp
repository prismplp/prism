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
#include "core/idtable_preds.h"
#include "up/graph.h"
#include "up/util.h"
#include "up/em.h"
#include "up/em_aux.h"
#include "up/em_aux_ml.h"
#include "up/viterbi.h"
#include "up/graph_aux.h"
#include "up/nonlinear_eq.h"
#include "up/scc.h"
#include "up/rank.h"
#include "up/crf.h"
#include "up/crf_learn.h"
#include "up/crf_learn_aux.h"
}
#include "up/sgd.h"

#include "eigen/Core"
#include "eigen/LU"

#include <iostream>
#include <set>
#include <cmath>


using namespace Eigen;

RNK_NODE_PTR rank_root;
RankMinibatchPtr rank_minibatches;
int rank_minibatch_id;
int num_rank_root;
void clean_rank_minibatch(void);
void initialize_rank_minibatch(void);

/*------------------------------------------------------------------------*/

void clean_rank_minibatch(void) {
	for (int i=0; i<num_minibatch; i++) {
		FREE(rank_minibatches[i].roots);
		FREE(rank_minibatches[i].egraph);
	}
	FREE(rank_minibatches);
	rank_minibatches=NULL;
}
void initialize_rank_minibatch(void) {

	if(num_rank_root<num_minibatch){
		prism_printf("The indicated rank_minibatch size (%d) is greater than the number of ranking lists (%d) \n",num_minibatch,num_rank_root);
		num_minibatch=num_rank_root;
	}
	rank_minibatches=(RankMinibatch*)MALLOC(num_minibatch*sizeof(RankMinibatch));

	for (int i=0; i<num_minibatch; i++) {
		rank_minibatches[i].egraph_size=0;
		rank_minibatches[i].batch_size=0;
		rank_minibatches[i].num_roots=0;
		rank_minibatches[i].count=0;
	}

	for (int i = 0; i < num_rank_root; i++) {
		int index = i%num_minibatch;
		if (i == failure_root_index) {
			continue;
		} else {
			rank_minibatches[index].num_roots++;
		}
	}
	for (int i=0; i<num_minibatch; i++) {
		rank_minibatches[i].roots=(RNK_NODE_PTR*)MALLOC(rank_minibatches[i].num_roots*sizeof(RNK_NODE_PTR));
	}
	int i=0;
	for (RNK_NODE_PTR itr = rank_root; itr != NULL; itr=itr->next) {
		int index = i%num_minibatch;
		rank_minibatches[index].roots[rank_minibatches[index].count]=itr;
		rank_minibatches[index].batch_size+=1;
		rank_minibatches[index].count++;
		i++;
	}
	for (int i=0; i<num_minibatch; i++) {
		initialize_visited_flags();
		for (int j=0;j<rank_minibatches[i].num_roots;j++){
			RNK_NODE_PTR itr=rank_minibatches[i].roots[j];
			for(int k=0;k<itr->goal_count;k++){
				EG_NODE_PTR eg_ptr = expl_graph[itr->goals[k]];
				set_visited_flags(eg_ptr);
			}
		}
		for (int j=0; j<sorted_egraph_size; j++) {
			if(sorted_expl_graph[j]->visited>0){
				rank_minibatches[i].egraph_size++;
			}
		}
		rank_minibatches[i].egraph=(EG_NODE_PTR*)MALLOC(rank_minibatches[i].egraph_size*sizeof(EG_NODE_PTR));
		rank_minibatches[i].count=0;
		for (int j=0; j<sorted_egraph_size; j++) {
			if(sorted_expl_graph[j]->visited>0){
				rank_minibatches[i].egraph[rank_minibatches[i].count]=sorted_expl_graph[j];
				rank_minibatches[i].count++;
			}
		}
	}
	

	if (verb_em) {
		for (int i=0; i<num_minibatch; i++) {
			prism_printf(" rank_minibatch(%d): Number of goals = %d, Graph size = %d\n",i,rank_minibatches[i].batch_size,rank_minibatches[i].egraph_size);
		}
	}
	INIT_VISITED_FLAGS;
}



/*------------------------------------------------------------------------*/


double compute_rank_loss(bool verb,int iterate) {
	int i;
	EG_NODE_PTR eg_ptr0,eg_ptr1;
	SW_INS_PTR sw_ptr;
	RNK_NODE_PTR rank_ptr=rank_root;
	int pair_count=0;
	int pair_grad_count=0;
	double ll_loss=0;
	double r_loss=0;
	double total_loss=0;
	for (RNK_NODE_PTR itr = rank_ptr; itr != NULL; itr=itr->next) {
		if(itr->goal_count>=2){
			for(i=0;i<itr->goal_count-1;i++){
				eg_ptr0 = expl_graph[itr->goals[i]];
				eg_ptr1 = expl_graph[itr->goals[i+1]];
				//if (i == failure_root_index) {
				//	eg_ptr->outside = num_goals / (1.0 - inside_failure);
				//}
				pair_count++;
				switch(rank_loss){
					case RANK_LOSS_HINGE:
					{
						// loss=h (h>0)
						double h = log(eg_ptr1->inside) - log(eg_ptr0->inside)+rank_loss_c;
						if(h>0){
							ll_loss+=h;
							pair_grad_count++;
						}
					}
					break;
					case RANK_LOSS_SQUARE:
					{
						// loss=h^2 (h>0)
						double h = log(eg_ptr1->inside) - log(eg_ptr0->inside)+rank_loss_c;
						if(h>0){
							ll_loss+=h*h;
							pair_grad_count++;
						}
					}
					break;
					case RANK_LOSS_EXP:
					{
						// loss=exp(z)
						double z = log(eg_ptr1->inside) - log(eg_ptr0->inside);
						ll_loss+=exp(z);
						pair_grad_count++;
					}
					break;
					case RANK_LOSS_LOG:
					{
						// loss=exp(z)
						double z = log(eg_ptr1->inside) - log(eg_ptr0->inside);
						ll_loss+=log(1.0+exp(z));
						pair_grad_count++;
					}
					break;
					default:
						emit_internal_error("unexpected loss function[%d]",
							rank_loss);
						RET_INTERNAL_ERR;
					break;
				}
			}
		}
	}

	for (i = 0; i < occ_switch_tab_size; i++) {
		sw_ptr = occ_switches[i];
		while (sw_ptr != NULL) {
			double r=sgd_penalty*sw_ptr->pi*sw_ptr->pi;
			r_loss+=r;
			sw_ptr = sw_ptr->next;
		}
	}
	total_loss=ll_loss+r_loss;
	//total_loss=ll_loss;
	if(verb){
		prism_printf("iteration #%d:\t total loss = %f (log likelihood loss = %f, regularization loss = %f) [positive loss pairs: %d/%d]\n",iterate,total_loss,ll_loss,r_loss,pair_grad_count,pair_count);
	}
	return total_loss;
}

int compute_rank_expectation_scaling_none(void) {
	int i,k;
	EG_PATH_PTR path_ptr;
	EG_NODE_PTR eg_ptr,node_ptr;
	EG_NODE_PTR eg_ptr0,eg_ptr1;
	SW_INS_PTR sw_ptr,sw_node_ptr;
	double q;
	
	for (i = 0; i < occ_switch_tab_size; i++) {
		sw_ptr = occ_switches[i];
		while (sw_ptr != NULL) {
			sw_ptr->total_expect = 0.0;
			sw_ptr = sw_ptr->next;
		}
	}

	for (i = 0; i < sorted_egraph_size; i++) {
		sorted_expl_graph[i]->outside = 0.0;
	}
	int pair_count=0;
	int pair_grad_count=0;
	RankMinibatchPtr mb_ptr=&rank_minibatches[rank_minibatch_id];
	for (i = 0; i < mb_ptr->num_roots; i++) {
		RNK_NODE_PTR itr =mb_ptr->roots[i];
		if(itr->goal_count>=2){
				for(k=0;k<itr->goal_count-1;k++){
					eg_ptr0 = expl_graph[itr->goals[k]];
					eg_ptr1 = expl_graph[itr->goals[k+1]];
					pair_count++;
					switch(rank_loss){
					case RANK_LOSS_SQUARE:
						{
							// loss=h^2 (h>0)
							double h = log(eg_ptr1->inside) - log(eg_ptr0->inside)+rank_loss_c;
							if(h>0){
								eg_ptr0->outside += 1.0 *2* h/ eg_ptr0->inside;
								eg_ptr1->outside += -1.0* 2* h/ eg_ptr1->inside;
								pair_grad_count++;
							}
						}
						break;
					case RANK_LOSS_HINGE:
						{
							// loss=h (h>0)
							double h = log(eg_ptr1->inside) - log(eg_ptr0->inside)+rank_loss_c;
							if(h>0){
								eg_ptr0->outside += 1.0 / eg_ptr0->inside;
								eg_ptr1->outside += -1.0 / eg_ptr1->inside;
								pair_grad_count++;
							}
						}
						break;
					case RANK_LOSS_EXP:
						{
							// loss=exp(z)
							double z = log(eg_ptr1->inside) - log(eg_ptr0->inside);
							eg_ptr0->outside += 1.0 * exp(z)/eg_ptr0->inside;
							eg_ptr1->outside += -1.0* exp(z)/eg_ptr1->inside;
						}
						break;
					case RANK_LOSS_LOG:
						{
							// loss=exp(z)
							double z = log(eg_ptr1->inside) - log(eg_ptr0->inside);
							eg_ptr0->outside += 1.0 * exp(z)/((1+exp(z))* eg_ptr0->inside);
							eg_ptr1->outside += -1.0* exp(z)/((1+exp(z))* eg_ptr1->inside);
						}
						break;

					default:
						emit_internal_error("unexpected loss function[%d]",
							rank_loss);
						RET_INTERNAL_ERR;
						break;
					}
					
				pair_grad_count++;
			}
		}
	}

	for (i = mb_ptr->egraph_size - 1; i >= 0; i--) {
		eg_ptr = mb_ptr->egraph[i];
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
					sw_node_ptr=sw_ptr->parent;
					while(sw_node_ptr!=NULL){
						if(sw_node_ptr!=sw_ptr){
							sw_node_ptr->total_expect -= q*sw_node_ptr->inside;
						}else{
							sw_node_ptr->total_expect += q*(1.0-sw_node_ptr->inside);
						}
						sw_node_ptr=sw_node_ptr->next;
					}
				}
			}
			path_ptr = path_ptr->next;
		}
	}

	return BP_TRUE;
}
// log scale is not supported
int compute_rank_expectation_scaling_log_exp(void) {
	int i,k;
	EG_PATH_PTR path_ptr;
	EG_NODE_PTR eg_ptr,node_ptr;
	EG_NODE_PTR eg_ptr0,eg_ptr1;
	SW_INS_PTR sw_ptr,sw_node_ptr;
	double q,r;
	RankMinibatchPtr mb_ptr=&rank_minibatches[rank_minibatch_id];
	RNK_NODE_PTR rank_ptr=rank_root;
	
	for (i = 0; i < occ_switch_tab_size; i++) {
		sw_ptr = occ_switches[i];
		while (sw_ptr != NULL) {
			sw_ptr->total_expect = 0.0;
			sw_ptr->has_first_expectation = 0;
			sw_ptr->first_expectation = 0.0;
			sw_ptr = sw_ptr->next;
		}
	}

	for (i = 0; i < sorted_egraph_size; i++) {
		sorted_expl_graph[i]->outside = 0.0;
		sorted_expl_graph[i]->has_first_outside = 0;
		sorted_expl_graph[i]->first_outside = 0.0;
	}

	for (RNK_NODE_PTR itr = rank_ptr; itr != NULL; itr=itr->next) {
		if(itr->goal_count>=2){
			for(i=0;i<itr->goal_count-1;i++){
				eg_ptr0 = expl_graph[itr->goals[i]];
				eg_ptr1 = expl_graph[itr->goals[i+1]];
				
				if(eg_ptr0->inside > eg_ptr1->inside){
					eg_ptr0->first_outside =
						log((double)(1.0)) - eg_ptr0->inside;
					eg_ptr0->has_first_outside = 1;
					eg_ptr0->outside = 1.0;
				}
			}
		}
	}

	for (i = mb_ptr->egraph_size - 1; i >= 0; i--) {
		eg_ptr = mb_ptr->egraph[i];
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

		path_ptr = eg_ptr->path_ptr;
		while (path_ptr != NULL) {
			q = eg_ptr->outside + path_ptr->inside;
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
				sw_node_ptr=sw_ptr->parent;
				//sw_node_ptr->inside : unscale
				//q,el : log scale
				//sw_node_ptr->first_expectation: log scale;
				//sw_node_ptr->total_expect: unscale;

				while(sw_node_ptr!=NULL){
					double el;
					if(sw_node_ptr!=sw_ptr){
						el=-(q+log(sw_node_ptr->inside));
					}else{
						el=q+log(1.0-sw_node_ptr->inside);
					}
					if (!sw_node_ptr->has_first_expectation) {
						sw_node_ptr->first_expectation = el;
						sw_node_ptr->total_expect += 1.0;
						sw_node_ptr->has_first_expectation = 1;
					} else if (el - sw_node_ptr->first_expectation >= log(HUGE_PROB)) {
						sw_node_ptr->total_expect *= exp(sw_node_ptr->first_expectation - el);
						sw_node_ptr->first_expectation = el;
						sw_node_ptr->total_expect += 1.0;
					} else {
						sw_node_ptr->total_expect += exp(el - sw_node_ptr->first_expectation);
					}
					sw_node_ptr=sw_node_ptr->next;
				}
			}

			path_ptr = path_ptr->next;
		}
	}

	/* unscale total_expect */
	for (i = 0; i < occ_switch_tab_size; i++) {
		sw_ptr = occ_switches[i];
		while (sw_ptr != NULL) {
			if (!sw_ptr->has_first_expectation){
				sw_ptr = sw_ptr->next;
				continue;
			}
			if (!(sw_ptr->total_expect > 0.0)) {
				emit_error("unexpected expectation for %s",prism_sw_ins_string(i));
				RET_ERR(err_invalid_numeric_value);
			}
			sw_ptr->total_expect =
				exp(sw_ptr->first_expectation + log(sw_ptr->total_expect));
			//printf("(%d,%f),",i,sw_ptr->total_expect);
			sw_ptr = sw_ptr->next;
		}
	}

	return BP_TRUE;
}

/*------------------------------------------------------------------------*/


int run_rank_learn(struct EM_Engine* em_ptr) {
	int	 r, iterate, old_valid, converged=0, saved = 0;
	double  loss,old_loss;

	//config_em(em_ptr);
	double start_time=getCPUTime();
	init_scc();
	initialize_parent_switch();
	initialize_rank_minibatch();
	double scc_time=getCPUTime();
	//start SGD
	double itemp = 1.0;
	int err=0;
	for (r = 0; r < num_restart; r++) {
		SHOW_PROGRESS_HEAD("#sgd-iters", r);
		initialize_params();
		initialize_sgd_weights();
		iterate = 0;
		while (!err) {
			old_valid = 0;
			while (1) {
				if (CTRLC_PRESSED) {
					SHOW_PROGRESS_INTR();
					SET_ERR(err_ctrl_c_pressed);
					err=1;
					break;
				}
				rank_minibatch_id=iterate%num_minibatch;
				
				compute_inside_linear();
				loss=compute_rank_loss(verb_em,iterate);
				if(scc_debug_level>=4) {
					print_eq();
				}
			
				if (!std::isfinite(loss)) {
					emit_internal_error("invalid loss: %s (at iteration #%d)",
							std::isnan(loss) ? "NaN" : "infinity", iterate);
					SET_ERR(ierr_invalid_likelihood);
					err=1;
					break;
				}
				if (num_minibatch==1){
					if (old_valid && loss - old_loss > prism_epsilon) {
						emit_error("loss increased [old: %.9f, new: %.9f] (at iteration #%d)",
								old_loss, loss, iterate);
						break;
					}
				}
				
				if ( REACHED_MAX_ITERATE(iterate)) {
					break;
				}

				old_loss = loss;
				old_valid  = 1;

				if(BP_ERROR==em_ptr->compute_expectation()){
					err=1;
					break;
				}

				SHOW_PROGRESS(iterate);
				update_sgd_weights(iterate);
				if(BP_ERROR==em_ptr->update_params()){
					err=1;
					break;
				}
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

		SHOW_PROGRESS_TAIL(converged, iterate, loss);

		if (r == 0 || loss > em_ptr->likelihood) {
			em_ptr->lambda     = loss;
			em_ptr->likelihood = loss;
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
	clean_rank_minibatch();
	if(scc_debug_level>=1) {
		prism_printf("CPU time (scc,solution,all)\n");
		prism_printf("# %f,%f,%f\n",scc_time-start_time,solution_time-scc_time,solution_time - start_time);
	}
	if (err) {
		return BP_ERROR;
	}
	em_ptr->bic = compute_bic(em_ptr->likelihood);
	em_ptr->cs  = em_ptr->smooth ? compute_cs(em_ptr->likelihood) : 0.0;
	return BP_TRUE;
}

void config_rank_learn(EM_ENG_PTR em_ptr) {
	if (log_scale) {
		em_ptr->compute_inside      = daem ? compute_daem_inside_scaling_log_exp : compute_inside_scaling_log_exp;
		em_ptr->examine_inside      = examine_inside_scaling_log_exp;
		em_ptr->compute_expectation = compute_rank_expectation_scaling_log_exp;
		em_ptr->compute_likelihood  = compute_likelihood_scaling_log_exp;
		em_ptr->compute_log_prior   = daem ? compute_daem_log_prior : compute_log_prior;
		em_ptr->update_params       = update_sgd_params;
	} else {
		em_ptr->compute_inside      = daem ? compute_daem_inside_scaling_none : compute_inside_scaling_none;
		em_ptr->examine_inside      = examine_inside_scaling_none;
		em_ptr->compute_expectation = compute_rank_expectation_scaling_none;
		em_ptr->compute_likelihood  = compute_likelihood_scaling_none;
		em_ptr->compute_log_prior   = daem ? compute_daem_log_prior : compute_log_prior;
		em_ptr->update_params       = update_sgd_params;
	}
}


extern "C"
int pc_rank_learn_7(void) {
	struct EM_Engine em_eng;
	RET_ON_ERR(check_smooth(&em_eng.smooth));
	config_rank_learn(&em_eng);
	RET_ON_ERR(run_rank_learn(&em_eng));
	return
	    bpx_unify(bpx_get_call_arg(1,7), bpx_build_integer(em_eng.iterate   )) &&
	    bpx_unify(bpx_get_call_arg(2,7), bpx_build_float  (em_eng.lambda    )) &&
	    bpx_unify(bpx_get_call_arg(3,7), bpx_build_float  (em_eng.likelihood)) &&
	    bpx_unify(bpx_get_call_arg(4,7), bpx_build_float  (em_eng.bic       )) &&
	    bpx_unify(bpx_get_call_arg(5,7), bpx_build_float  (em_eng.cs        )) &&
	    bpx_unify(bpx_get_call_arg(6,7), bpx_build_integer(em_eng.smooth    )) ;
}

extern "C"
int pc_set_goal_rank_1(void) {
	TERM goal_ranks;
	TERM goal_list;
	TERM goal;
	int gid;
	int goal_count;
	RNK_NODE_PTR rank_path;

	goal_ranks = bpx_get_call_arg(1,1);
	num_rank_root=0;

	while (bpx_is_list(goal_ranks)) {
		if(rank_root==NULL){
			rank_root=(RNK_NODE_PTR)MALLOC(sizeof(RankNode));
			rank_path=rank_root;
			rank_path->next=NULL;
		}else{
			rank_path->next=(RNK_NODE_PTR)MALLOC(sizeof(RankNode));
			rank_path=rank_path->next;
			rank_path->next=NULL;
		}
		num_rank_root++;
		
		goal_list = bpx_get_car(goal_ranks);
		// length of goal_list
		goal_count=0;
		while (bpx_is_list(goal_list)){
			goal_count++;
			goal_list = bpx_get_cdr(goal_list);
		}
		rank_path->goal_count=goal_count;
		rank_path->goals=(int*)MALLOC(goal_count*sizeof(int));
		// set gid
		goal_list = bpx_get_car(goal_ranks);
		goal_count=0;
		while (bpx_is_list(goal_list)){
			goal = bpx_get_car(goal_list);
			gid = prism_goal_id_get(goal);
			//char* s=bp_term_2_string(goal);
			//printf("[%d:%s]",gid,s);
			rank_path->goals[goal_count]=gid;
			goal_count++;
			goal_list = bpx_get_cdr(goal_list);
		}
		goal_ranks = bpx_get_cdr(goal_ranks);
	}

	return BP_TRUE;
}

extern "C"
int pc_clear_goal_rank_0(void) {
	RNK_NODE_PTR rank_path=rank_root;
	RNK_NODE_PTR temp;
	while(rank_path!=NULL){
		FREE(rank_path->goals);
		temp=rank_path;
		rank_path=rank_path->next;
		FREE(temp);
	}
	rank_root=NULL;
	return BP_TRUE;
}

// crf rank learn is not supported
int crf_rank_learn(CRF_ENG_PTR crf_ptr) {
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

			if (!std::isfinite(likelihood)) {
				emit_internal_error("invalid log likelihood: %s (at iteration #%d)",
				                    std::isnan(likelihood) ? "NaN" : "infinity", iterate);
				RET_ERR(ierr_invalid_likelihood);
			}
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


