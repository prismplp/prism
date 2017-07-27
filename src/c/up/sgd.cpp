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

#include "up/sgd.h"
#include <iostream>
#include <set>
#include <cmath>

MinibatchPtr minibatches;
int minibatch_id;

void initialize_parent_switch(void) {
	int i;
	SW_INS_PTR ptr;
	SW_INS_PTR parent_ptr;
	for (i = 0; i < sw_tab_size; i++) {
		parent_ptr=ptr = switches[i];
		while (ptr != NULL) {
			ptr->parent=parent_ptr;
			ptr = ptr->next;
		}
	}
}

void initialize_sgd_weights(void) {
	int i;
	SW_INS_PTR ptr;

	for (i = 0; i < sw_tab_size; i++) {
		ptr = switches[i];
		while (ptr != NULL) {
			ptr->pi=log(ptr->inside);
			int num_g_aux=2;
			ptr->gradient_aux=new double[num_g_aux];
			for(int j=0;j<num_g_aux;j++){
				ptr->gradient_aux[j]=0;
			}
			ptr = ptr->next;
		}
	}
}

int update_sgd_weights(int iterate) {
	int i;
	SW_INS_PTR ptr;

	for (i = 0; i < occ_switch_tab_size; i++) {
		ptr = occ_switches[i];
		
		if (ptr->fixed > 0) continue;
		while (ptr != NULL) {
			if (ptr->fixed == 0){
				double g=ptr->total_expect-sgd_penalty*2*ptr->pi;
				double dw;
				double t= iterate;
				switch(sgd_optimizer){
					case OPTIMIZER_SGD:
					{//SGD
						dw=sgd_learning_rate*g;
						break;
					}
					case OPTIMIZER_ADADELTA:
					{//AdaDelta
						//alpha is about 1
						double alpha=sgd_learning_rate;
						double gamma=sgd_adadelta_gamma;
						double epsilon=sgd_adadelta_epsilon;
						double r,v,s;
						r=ptr->gradient_aux[0];
						s=ptr->gradient_aux[1];
						//
						r=gamma*r+(1-gamma)*g*g;
						v=sqrt(s+epsilon)/sqrt(r+epsilon)*g;
						s=gamma*s+(1-gamma)*v*v;
						//
						ptr->gradient_aux[0]=r;
						ptr->gradient_aux[1]=s;
						dw=alpha*v;
						break;
					}
					case OPTIMIZER_ADAM:
					{//Adam
						double alpha=sgd_learning_rate;
						double beta=sgd_adam_beta;
						double gamma=sgd_adam_gamma;
						double epsilon=sgd_adam_epsilon;
						double r,v;
						r=ptr->gradient_aux[0];
						v=ptr->gradient_aux[1];
						//
						r=gamma*r+(1-gamma)*g*g;
						v=beta*v+(1-beta)*g;
						//
						ptr->gradient_aux[0]=r;
						ptr->gradient_aux[1]=v;
						dw=alpha/(sqrt(r/(1-pow(gamma,t+1))+epsilon))*v/(1-pow(beta,t+1));
						break;
					}
				}
				ptr->pi += dw;
			}
			if (log_scale && ptr->inside < TINY_PROB) {
				emit_error("Parameter being zero (-inf in log scale) -- %s",
						   prism_sw_ins_string(ptr->id));
				RET_ERR(err_underflow);
			}
			ptr = ptr->next;
		}
	}

	return BP_TRUE;
}



int update_sgd_params(void) {
	int i;
	SW_INS_PTR ptr;
	double sum;

	for (i = 0; i < occ_switch_tab_size; i++) {
		ptr = occ_switches[i];
		sum = 0.0;
		while (ptr != NULL) {
			sum += exp(ptr->pi);
			ptr = ptr->next;
		}
		if (sum != 0.0) {
			ptr = occ_switches[i];
			if (ptr->fixed > 0) continue;
			while (ptr != NULL) {
				if (ptr->fixed == 0) ptr->inside = exp(ptr->pi) / sum;
				if (log_scale && ptr->inside < TINY_PROB) {
					emit_error("Parameter being zero (-inf in log scale) -- %s",
					           prism_sw_ins_string(ptr->id));
					RET_ERR(err_underflow);
				}
				ptr = ptr->next;
			}
		}
	}

	return BP_TRUE;
}


void set_visited_flags(EG_NODE_PTR node_ptr) {
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

void initialize_visited_flags(void) {
	int i;
	for (i=0; i<sorted_egraph_size; i++) {
		sorted_expl_graph[i]->visited = 0;
	}
}


void clean_minibatch(void) {
	for (int i=0; i<num_minibatch; i++) {
		FREE(minibatches[i].roots);
		FREE(minibatches[i].egraph);
	}
	FREE(minibatches);
	minibatches=NULL;
}
void initialize_minibatch(void) {
	
	if(num_roots<num_minibatch){
		prism_printf("The indicated minibatch size (%d) is greater than the number of goals (%d) \n",num_minibatch,num_roots);
		num_minibatch=num_roots;
	}
	minibatches=(Minibatch*)MALLOC(num_minibatch*sizeof(Minibatch));

	for (int i=0; i<num_minibatch; i++) {
		minibatches[i].egraph_size=0;
		minibatches[i].batch_size=0;
		minibatches[i].num_roots=0;
		minibatches[i].count=0;
	}

	for (int i = 0; i < num_roots; i++) {
		int index = i%num_minibatch;
		if (i == failure_root_index) {
			continue;
		} else {
			minibatches[index].num_roots++;
		}
	}
	for (int i=0; i<num_minibatch; i++) {
		minibatches[i].roots=(ROOT*)MALLOC(minibatches[i].num_roots*sizeof(ROOT));
	}
	for (int i = 0; i < num_roots; i++) {
		int index = i%num_minibatch;
		if (i == failure_root_index) {
			continue;
		} else {
			minibatches[index].roots[minibatches[index].count]=roots[i];
			minibatches[index].batch_size+=roots[i]->count;
			minibatches[index].count++;
		}
	}
	for (int i=0; i<num_minibatch; i++) {
		initialize_visited_flags();
		for (int j=0;j<minibatches[i].num_roots;j++){
			EG_NODE_PTR eg_ptr = expl_graph[minibatches[i].roots[j]->id];
			set_visited_flags(eg_ptr);
		}
		for (int j=0; j<sorted_egraph_size; j++) {
			if(sorted_expl_graph[j]->visited>0){
				minibatches[i].egraph_size++;
			}
		}
		minibatches[i].egraph=(EG_NODE_PTR*)MALLOC(minibatches[i].egraph_size*sizeof(EG_NODE_PTR));
		minibatches[i].count=0;
		for (int j=0; j<sorted_egraph_size; j++) {
			if(sorted_expl_graph[j]->visited>0){
				minibatches[i].egraph[minibatches[i].count]=sorted_expl_graph[j];
				minibatches[i].count++;
			}
		}
	}
	

	if (verb_em) {
		for (int i=0; i<num_minibatch; i++) {
			printf(" Minibatch(%d): Number of goals = %d, Graph size = %d\n",i,minibatches[i].batch_size,minibatches[i].egraph_size);
		}
	}
}



/*------------------------------------------------------------------------*/

int compute_sgd_expectation_scaling_none(void) {
	int i,k;
	EG_PATH_PTR path_ptr;
	EG_NODE_PTR eg_ptr,node_ptr;
	SW_INS_PTR sw_ptr,sw_node_ptr;
	double q;
	MinibatchPtr mb_ptr=&minibatches[minibatch_id];
	
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
	for (i = 0; i < mb_ptr->num_roots; i++) {
		eg_ptr = expl_graph[roots[i]->id];
		if (i == failure_root_index) {
			eg_ptr->outside = num_goals / (1.0 - inside_failure);
		} else {
			eg_ptr->outside = roots[i]->count / eg_ptr->inside;
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

int compute_sgd_expectation_scaling_log_exp(void) {
	int i,k;
	EG_PATH_PTR path_ptr;
	EG_NODE_PTR eg_ptr,node_ptr;
	SW_INS_PTR sw_ptr,sw_node_ptr;
	double q,r;
	MinibatchPtr mb_ptr=&minibatches[minibatch_id];
	
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

	for (i = 0; i < mb_ptr->num_roots; i++) {
		eg_ptr = expl_graph[mb_ptr->roots[i]->id];
		if (i == failure_root_index) {
			eg_ptr->first_outside =
			    log(num_goals / (1.0 - exp(inside_failure)));
		} else {
			eg_ptr->first_outside =
			    log((double)(roots[i]->count)) - eg_ptr->inside;
		}
		eg_ptr->has_first_outside = 1;
		eg_ptr->outside = 1.0;
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
			sw_ptr = sw_ptr->next;
		}
	}

	return BP_TRUE;
}

/*------------------------------------------------------------------------*/




int run_sgd(struct EM_Engine* em_ptr) {
	int	r, iterate,old_valid=0, converged=0, saved = 0;
	double  likelihood, log_prior=0;
	double  old_lambda,lambda;

	//config_em(em_ptr);
	double start_time=getCPUTime();
	init_scc();
	initialize_parent_switch();
	initialize_minibatch();
	double scc_time=getCPUTime();
	//start EM
	double itemp = 1.0;
	for (r = 0; r < num_restart; r++) {
		SHOW_PROGRESS_HEAD("#sgd-iters", r);
		initialize_params();
		initialize_sgd_weights();
		iterate = 0;
		while (1) {
			old_valid = 0;
			while (1) {
				if (CTRLC_PRESSED) {
					SHOW_PROGRESS_INTR();
					RET_ERR(err_ctrl_c_pressed);
				}
				minibatch_id=iterate%num_minibatch;
				
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
				/*
				if (old_valid && old_lambda - lambda > prism_epsilon) {
					emit_error("log likelihood or log post decreased [old: %.9f, new: %.9f] (at iteration #%d)",
							old_lambda, lambda, iterate);
					RET_ERR(err_invalid_likelihood);
				}

				converged = (old_valid && lambda - old_lambda <= prism_epsilon);
				*/
				converged=false;
				if (converged || REACHED_MAX_ITERATE(iterate)) {
					break;
				}

				old_lambda = lambda;
				old_valid  = 1;

				RET_ON_ERR(em_ptr->compute_expectation());
				//compute_expectation_linear();

				SHOW_PROGRESS(iterate);
				update_sgd_weights(iterate);
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
	clean_minibatch();
	if(scc_debug_level>=1) {
		printf("CPU time (scc,solution,all)\n");
		printf("# %f,%f,%f\n",scc_time-start_time,solution_time-scc_time,solution_time - start_time);
	}

	em_ptr->bic = compute_bic(em_ptr->likelihood);
	em_ptr->cs  = em_ptr->smooth ? compute_cs(em_ptr->likelihood) : 0.0;
	return BP_TRUE;
}

void config_sgd(EM_ENG_PTR em_ptr) {
	if (log_scale) {
		em_ptr->compute_inside      = daem ? compute_daem_inside_scaling_log_exp : compute_inside_scaling_log_exp;
		em_ptr->examine_inside      = examine_inside_scaling_log_exp;
		em_ptr->compute_expectation = compute_sgd_expectation_scaling_log_exp;
		em_ptr->compute_likelihood  = compute_likelihood_scaling_log_exp;
		em_ptr->compute_log_prior   = daem ? compute_daem_log_prior : compute_log_prior;
		em_ptr->update_params       = update_sgd_params;
	} else {
		em_ptr->compute_inside      = daem ? compute_daem_inside_scaling_none : compute_inside_scaling_none;
		em_ptr->examine_inside      = examine_inside_scaling_none;
		em_ptr->compute_expectation = compute_sgd_expectation_scaling_none;
		em_ptr->compute_likelihood  = compute_likelihood_scaling_none;
		em_ptr->compute_log_prior   = daem ? compute_daem_log_prior : compute_log_prior;
		em_ptr->update_params       = update_sgd_params;
	}
}

extern "C"
int pc_sgd_learn_7(void) {
	struct EM_Engine em_eng;
	RET_ON_ERR(check_smooth(&em_eng.smooth));
	config_sgd(&em_eng);
	//scc_debug_level = bpx_get_integer(bpx_get_call_arg(7,7));
	run_sgd(&em_eng);
	/*
	printf("switches");
	SW_INS_PTR ptr;
	for (int i = 0; i < sw_tab_size; i++) {
		SW_INS_PTR parent_ptr=ptr = switches[i];
		printf("%d:",parent_ptr->id);
		while (ptr != NULL) {
			printf("%d,",ptr->id);
			ptr = ptr->next;
		}
		printf("\n");
	}
	printf("instances\n");
	for (int i = 0; i < sw_ins_tab_size; i++) {
		ptr = switch_instances[i];
		printf("%d:",ptr->id);
		printf("\n");
	}
	printf("occ\n");
	for (int i = 0; i < occ_switch_tab_size; i++) {
		ptr = occ_switches[i];
		printf("%d:",ptr->id);
		printf("\n");
	}*/
	return
	    bpx_unify(bpx_get_call_arg(1,7), bpx_build_integer(em_eng.iterate   )) &&
	    bpx_unify(bpx_get_call_arg(2,7), bpx_build_float  (em_eng.lambda    )) &&
	    bpx_unify(bpx_get_call_arg(3,7), bpx_build_float  (em_eng.likelihood)) &&
	    bpx_unify(bpx_get_call_arg(4,7), bpx_build_float  (em_eng.bic       )) &&
	    bpx_unify(bpx_get_call_arg(5,7), bpx_build_float  (em_eng.cs        )) &&
	    bpx_unify(bpx_get_call_arg(6,7), bpx_build_integer(em_eng.smooth    )) ;
}

