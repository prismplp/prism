#define CXX_COMPILE 

#ifdef _MSC_VER
#include <windows.h>
#endif
extern "C" {
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


#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "up/nonlinear_eq.h"
#include "up/scc.h"
}

extern "C"
int pc_compute_n_nonlinear_viterbi_rerank_4(void){
	emit_error("n_viterbi on cyclic explanation graphs is not supported");
	RET_ERR(ierr_function_not_implemented);
}
extern "C"
int pc_compute_n_nonlinear_viterbi_3(void){
	emit_error("n_viterbi on cyclic explanation graphs is not supported");
	RET_ERR(ierr_function_not_implemented);
}
extern "C"
int pc_nonlinear_eq_2(void) {
	int i;
	scc_debug_level = bpx_get_integer(bpx_get_call_arg(1,2));
	double start_time=getCPUTime();	
	init_scc();
	double scc_time=getCPUTime();
	//solution
	if(scc_debug_level>=2) {
		printf("non-linear solver: Broyden's method\n");
	}
	for(i=0; i<scc_num; i++) {
		solve_nonlinear_scc(i,compute_scc_functions);
		if(scc_debug_level>=2) {
			int n=sccs[i].size;
			int j;
			for(j=0; j<n; j++) {
				int w=sccs[i].el[j];
				printf("%d:%f\n",w,sorted_expl_graph[w]->inside);
			}
		}
	}
	double solution_time=getCPUTime();
	//free data
	free_scc();
	double prob=sorted_expl_graph[sorted_egraph_size-1]->inside;
	if(scc_debug_level>=1) {
		printf("CPU time (scc,solution,all)\n");
		printf("# %f,%f,%f\n",scc_time-start_time,solution_time-scc_time,solution_time - start_time);
	}
	return bpx_unify(bpx_get_call_arg(2,2),
			bpx_build_float(prob));
}

extern "C"
int pc_compute_nonlinear_viterbi_6(void) {
	TERM p_goal_path,p_subpath_goal,p_subpath_sw;
	int goal_id;
	double viterbi_prob;

	//scc_debug_level = bpx_get_integer(bpx_get_call_arg(1,6));
	goal_id = bpx_get_integer(bpx_get_call_arg(2,6));

	initialize_egraph_index();
	alloc_sorted_egraph(1);
	/* INIT_MIN_MAX_NODE_NOS; */
	RET_ON_ERR(sort_one_egraph(goal_id,0,1));
	if (verb_graph) print_egraph(0,PRINT_NEUTRAL);

	double start_time=getCPUTime();
    init_scc();
	double scc_time=getCPUTime();
	
	compute_nonlinear_viterbi(scc_debug_level);
	
	double solution_time=getCPUTime();

	if (debug_level) print_egraph(1,PRINT_VITERBI);

	get_most_likely_path(goal_id,&p_goal_path,&p_subpath_goal,
			&p_subpath_sw,&viterbi_prob);

	free_scc();
	if(scc_debug_level>=1) {
		printf("CPU time (scc,solution,all)\n");
		printf("# %f,%f,%f\n",scc_time-start_time,solution_time-scc_time,solution_time - start_time);
	}
	return
		bpx_unify(bpx_get_call_arg(3,6), p_goal_path)    &&
		bpx_unify(bpx_get_call_arg(4,6), p_subpath_goal) &&
		bpx_unify(bpx_get_call_arg(5,6), p_subpath_sw)   &&
		bpx_unify(bpx_get_call_arg(6,6), bpx_build_float(viterbi_prob));
}


