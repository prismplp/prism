#ifndef __INFIX_H__
#define __INFIX_H__

int pc_nonlinear_eq_2(void);
int pc_linear_eq_2(void);
double compute_nonlinear_viterbi(int nl_debug_level);
int pc_compute_nonlinear_viterbi_6(void);
int pc_cyc_em_7(void);
/*
   additional informaion to sorted_expl_graph
 */
typedef struct {
	int visited;
	int index;
	int lowlink;
	int in_stack;
} SCC_Node;
/*
   Stack for the Tarjan algorithm (to find SCCs)
 */
typedef struct SCC_StackEl {
	int index;
	struct SCC_StackEl* next;
} SCC_Stack;
/*
el  : array of indexes that corresponded to indexes of sorted_expl_graph
size: size of el
 */
typedef struct {
	int* el;
	int size;
	int order;
} SCC;


#endif /* __INFIX_H__ */
