#include "up/up.h"
#include "up/graph.h"
#include "up/graph_aux.h"
#include "up/flags.h"
#include "up/viterbi.h"
#include "up/crf_learn.h"

/*------------------------------------------------------------------------*/

/* Viterbi works on only one explanation graph */
void compute_crf_max(void) {
	int i,k,u;
	double max_p,this_path_max;
	EG_PATH_PTR max_path = NULL;
	EG_NODE_PTR eg_ptr;
	EG_PATH_PTR path_ptr;

	for (i = 0; i < sorted_egraph_size; i++) {
		max_p = 1.0;          /* any positive value is possible */
		eg_ptr = sorted_expl_graph[i];
		path_ptr = eg_ptr->path_ptr;

		/* path_ptr should not be NULL; but it happens */
		if (path_ptr == NULL) {
			max_p = 0.0;
			max_path = NULL;
		}

		u = 0;

		/* [Note] we perform probability computations in log-scale */
		while (path_ptr != NULL) {
			this_path_max = 0.0;
			for (k = 0; k < path_ptr->children_len; k++) {
				this_path_max += path_ptr->children[k]->max;
			}
			for (k = 0; k < path_ptr->sws_len; k++) {
				this_path_max += path_ptr->sws[k]->inside * path_ptr->sws[k]->inside_h;
			}
			path_ptr->max = this_path_max;

			if (u == 0 || max_p <= this_path_max) {
				max_p = this_path_max;
				max_path = path_ptr;
			}

			path_ptr = path_ptr->next;
			u++;
		}
		sorted_expl_graph[i]->max = max_p;
		sorted_expl_graph[i]->max_path = max_path;
	}
}

/*------------------------------------------------------------------------*/

/* [Note] node copying is not required here even in computation without
 * inter-goal sharing, but we need to declare it explicitly.
 */
int pc_compute_crfviterbi_5(void) {
	TERM p_goal_path,p_subpath_goal,p_subpath_sw;
	int goal_id;
	double viterbi_prob;

	goal_id = bpx_get_integer(bpx_get_call_arg(1,5));

	initialize_egraph_index();
	alloc_sorted_egraph(1);
	/* INIT_MIN_MAX_NODE_NOS; */
	RET_ON_ERR(sort_one_egraph(goal_id,0,1));
	if (verb_graph) print_egraph(0,PRINT_NEUTRAL);

	initialize_weights();
	compute_crf_max();

	if (debug_level) print_egraph(1,PRINT_VITERBI);

	get_most_likely_path(goal_id,&p_goal_path,&p_subpath_goal,
	                     &p_subpath_sw,&viterbi_prob);

	return
	    bpx_unify(bpx_get_call_arg(2,5), p_goal_path)    &&
	    bpx_unify(bpx_get_call_arg(3,5), p_subpath_goal) &&
	    bpx_unify(bpx_get_call_arg(4,5), p_subpath_sw)   &&
	    bpx_unify(bpx_get_call_arg(5,5), bpx_build_float(viterbi_prob));
}

int pc_compute_n_crfviterbi_3(void) {
	TERM p_n_viterbi_list;
	int n,goal_id;

	n       = bpx_get_integer(bpx_get_call_arg(1,3));
	goal_id = bpx_get_integer(bpx_get_call_arg(2,3));

	initialize_egraph_index();
	alloc_sorted_egraph(1);
	/* INIT_MIN_MAX_NODE_NOS; */
	RET_ON_ERR(sort_one_egraph(goal_id,0,1));
	if (verb_graph) print_egraph(0,PRINT_NEUTRAL);

	initialize_weights();
	compute_n_crf_max(n);

	if (debug_level) print_egraph(1,PRINT_VITERBI);

	get_n_most_likely_path(n,goal_id,&p_n_viterbi_list);

	return bpx_unify(bpx_get_call_arg(3,3),p_n_viterbi_list);
}
