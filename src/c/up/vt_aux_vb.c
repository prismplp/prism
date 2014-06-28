/* -*- c-basic-offset: 2; tab-width: 8 -*- */

/*------------------------------------------------------------------------*/

#include "bprolog.h"
#include "up/up.h"
#include "up/graph.h"
#include "up/flags.h"
#include "up/util.h"

/*------------------------------------------------------------------------*/

double compute_vbvt_free_energy_l1_scaling_none(void) {
	double l1 = 0.0;
	SW_INS_PTR ptr;
	int i;

	for(i = 0; i < occ_switch_tab_size; i++) {
		ptr = occ_switches[i];
		while(ptr!=NULL) {
			l1 += ((ptr->inside_h - 1.0) - ptr->smooth - ptr->count) * log(ptr->pi);
			ptr = ptr->next;
		}
	}

	return l1;
}

double compute_vbvt_free_energy_l1_scaling_log_exp(void) {
	double l1 = 0.0;
	SW_INS_PTR ptr;
	int i;

	for(i = 0; i < occ_switch_tab_size; i++) {
		ptr = occ_switches[i];
		while(ptr!=NULL) {
			/* pi is in log-scale */
			l1 += ((ptr->inside_h - 1.0) - ptr->smooth - ptr->count) * ptr->pi;
			ptr = ptr->next;
		}
	}

	return l1;
}
/*------------------------------------------------------------------------*/

void compute_max_pi(void) {
	int i,k,u;
	double max_p = 0.0,this_path_max;
	EG_PATH_PTR max_path = NULL;
	EG_NODE_PTR eg_ptr;
	EG_PATH_PTR path_ptr;

	if(log_scale) {
		for(i=0; i<sorted_egraph_size; i++) {
			eg_ptr = sorted_expl_graph[i];
			path_ptr = eg_ptr->path_ptr;

			/* path_ptr should not be NULL; but it happens */
			if(path_ptr==NULL) {
				max_p = 0.0; /* log-scale */
				max_path = NULL;
			} else {
				u = 0;
				while(path_ptr!=NULL) {
					this_path_max = 0.0;
					for(k=0; k<path_ptr->children_len; k++) {
						this_path_max += path_ptr->children[k]->max;
					}
					for(k=0; k<path_ptr->sws_len; k++) {
						this_path_max += path_ptr->sws[k]->pi;
					}
					path_ptr->max = this_path_max;

					if(u==0 || max_p <= this_path_max) {
						max_p = this_path_max;
						max_path = path_ptr;
					}

					path_ptr = path_ptr->next;
					u++;
				}
			}

			sorted_expl_graph[i]->max = max_p;
			sorted_expl_graph[i]->max_path = max_path;
		}
	} else {
		for(i=0; i<sorted_egraph_size; i++) {
			max_p = 0.0;
			eg_ptr = sorted_expl_graph[i];
			path_ptr = eg_ptr->path_ptr;

			/* path_ptr should not be NULL; but it happens */
			if(path_ptr==NULL) {
				max_p = 1.0;
				max_path = NULL;
			}
			while(path_ptr!=NULL) {
				this_path_max = 1.0;
				for(k=0; k<path_ptr->children_len; k++) {
					this_path_max *= path_ptr->children[k]->max;
				}
				for(k=0; k<path_ptr->sws_len; k++) {
					this_path_max *= path_ptr->sws[k]->pi;
				}
				path_ptr->max = this_path_max;

				if(max_p < this_path_max) {
					max_p = this_path_max;
					max_path = path_ptr;
				}

				path_ptr = path_ptr->next;
			}

			sorted_expl_graph[i]->max = max_p;
			sorted_expl_graph[i]->max_path = max_path;
		}
	}
}

/*------------------------------------------------------------------------*/

int update_vbvt_hyperparams(void) {
	int i;
	SW_INS_PTR ptr;

	for(i=0; i<occ_switch_tab_size; i++) {
		ptr = occ_switches[i];
		if(ptr->fixed_h > 0) continue;

		while(ptr!=NULL) {
			ptr->inside_h = ptr->count + ptr->smooth + 1.0;
			ptr = ptr->next;
		}
	}

	return BP_TRUE;
}

/*------------------------------------------------------------------------*/
