#ifndef __SCC_H__
#define __SCC_H__

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

/*
   scc_mapping from IDs to indexes for explanation graph pointers
   ex. relationship between IDs and indexes
//	eg_ptr=sorted_expl_graph[index];
//	scc_mapping[eg_ptr->id]=index;
 */
extern int* scc_mapping;
extern int scc_num;
extern SCC* sccs;

double getCPUTime();
void init_scc();
void free_scc();
void compute_scc_functions(double* x,double* out,int scc);
void solve_nonlinear_scc(int scc,void(*compute_scc_functions)(double* x,double* o,int scc));
void compute_expectation_linear_cycle();
void solve_linear_scc(int scc);
void compute_expectation_linear();
void compute_inside_linear();
void update_inside_scc(int scc);

void print_sccs();
void print_sccs_statistics();
void print_eq_outside();
void print_eq();
#endif

