/* -*- c-basic-offset: 2; tab-width: 8 -*- */

/*------------------------------------------------------------------------*/

#include "bprolog.h"
#include "up/up.h"
#include "up/graph.h"
#include "up/util.h"
#include "up/flags.h"

/*------------------------------------------------------------------------*/
static EG_PATH_PTR * max_paths = NULL;
static int max_max_path_size;
static int max_path_size;

static void expand_max_paths(int req_max_path_size) {
	int old_size,i;
	if (req_max_path_size > max_max_path_size) {
		old_size = max_max_path_size;
		while (req_max_path_size > max_max_path_size) {
			max_max_path_size *= 2;
		}
		max_paths =
		    (EG_PATH_PTR *)
		    REALLOC(max_paths,
		            max_max_path_size * sizeof(EG_PATH_PTR));

		for (i = old_size; i < max_max_path_size; i++) {
			max_paths[i] = NULL;
		}
	}
}

void count_occ_sws(void) {
	int i,k,t,root_id,count;
	EG_PATH_PTR path_ptr;
	SW_INS_PTR ptr;
	int vindex;

	for(i = 0; i<occ_switch_tab_size; i++) {
		ptr = occ_switches[i];
		while(ptr != NULL) {
			ptr->count = 0;
			ptr = ptr->next;
		}
	}
	max_max_path_size = sorted_egraph_size;
	max_paths = (EG_PATH_PTR *)MALLOC(max_max_path_size * sizeof(EG_PATH_PTR));
	for (i = 0; i < max_max_path_size; i++)
		max_paths[i] = NULL;

	for (t = 0; t < num_roots; t++) {
		root_id  = roots[t]->id;
		count    = roots[t]->count;

		max_paths[0]  = expl_graph[root_id]->max_path;
		max_path_size = 1;
		vindex = 0;

		while (vindex < max_path_size) {
			path_ptr = max_paths[vindex];
			for (i = 0; i < path_ptr->children_len; i++) {
				if (path_ptr->children[i]->max_path == NULL) continue;

				if (max_path_size >= max_max_path_size)
					expand_max_paths(max_path_size + 1);

				max_paths[max_path_size] = path_ptr->children[i]->max_path;
				max_path_size++;
			}
			vindex++;
		}

		for (i = 0; i < max_path_size; i++) {
			path_ptr = max_paths[i];
			for (k = 0; k < path_ptr->sws_len; k++)
				path_ptr->sws[k]->count += count;
			max_paths[i] = NULL;
		}
	}

	FREE(max_paths);
}

/*------------------------------------------------------------------------*/

int examine_likelihood(void) {
	SW_INS_PTR ptr;
	int i;

	for(i = 0; i < occ_switch_tab_size; i++) {
		ptr = occ_switches[i];
		while(ptr!=NULL) {
			if(ptr->count == 0 && ptr->inside < TINY_PROB) {
				emit_error("Parameter being zero -- %s",prism_sw_ins_string(ptr->id)); //FIXME:error message
				RET_ERR(err_underflow);
			}
			ptr = ptr->next;
		}
	}

	return BP_TRUE;
}

/*------------------------------------------------------------------------*/

double compute_vt_likelihood(void) {
	double likelihood = 0.0;
	SW_INS_PTR ptr;
	int i;

	for(i = 0; i < occ_switch_tab_size; i++) {
		ptr = occ_switches[i];
		while(ptr!=NULL) {
			likelihood += ptr->count * log(ptr->inside);
			ptr = ptr->next;
		}
	}

	return likelihood;
}

/*------------------------------------------------------------------------*/

int update_vt_params(void) {
	int i;
	SW_INS_PTR ptr;
	double sum;

	for (i = 0; i < occ_switch_tab_size; i++) {
		ptr = occ_switches[i];
		sum = 0.0;
		while (ptr != NULL) {
			ptr = ptr->next;
		}
		if (sum != 0.0) {
			ptr = occ_switches[i];
			if (ptr->fixed > 0) continue;
			while (ptr != NULL) {
				if (ptr->fixed == 0) ptr->inside = ptr->count / sum;
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

int update_vt_params_smooth(void) {
	int i;
	SW_INS_PTR ptr;
	double sum;

	for (i = 0; i < occ_switch_tab_size; i++) {
		ptr = occ_switches[i];
		sum = 0.0;
		while (ptr != NULL) {
			sum += ptr->count + ptr->smooth;
			ptr = ptr->next;
		}
		if (sum != 0.0) {
			ptr = occ_switches[i];
			if (ptr->fixed > 0) continue;
			while (ptr != NULL) {
				if (ptr->fixed == 0)
					ptr->inside = (ptr->count + ptr->smooth) / sum;
				ptr = ptr->next;
			}
		}
	}

	return BP_TRUE;
}

/*------------------------------------------------------------------------*/
