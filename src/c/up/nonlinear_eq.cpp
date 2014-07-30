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

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
}
#ifndef _MSC_VER
extern "C" {
#include <sys/time.h>
#include <sys/resource.h>
}
double getCPUTime() {
	struct rusage RU;
	getrusage(RUSAGE_SELF, &RU);
	return RU.ru_utime.tv_sec + (double)RU.ru_utime.tv_usec*1e-6;
}
#else

double getCPUTime(){
	FILETIME creationTime;
	FILETIME exitTime;
	FILETIME kernelTime;
	FILETIME userTime;
	{
		GetProcessTimes(GetCurrentProcess(), 
				&creationTime, 
				&exitTime, 
				&kernelTime, 
				&userTime);
	}
	{
		SYSTEMTIME userSystemTime;
		if ( FileTimeToSystemTime( &userTime, &userSystemTime ) != -1 )
			return (double)userSystemTime.wHour * 3600.0 +
				(double)userSystemTime.wMinute * 60.0 +
				(double)userSystemTime.wSecond +
				(double)userSystemTime.wMilliseconds / 1000.0;
	}
	return 0;
}
#endif
/*
   mapping from IDs to indexes for explanation graph pointers
   ex. relationship between IDs and indexes
//	eg_ptr=sorted_expl_graph[index];
//	mapping[eg_ptr->id]=index;
 */
static int* mapping;
/*
   mapping from IDs to indexes for msws
   ex. relationship between IDs and indexes
//	ptr = occ_switches[index];
//	while (ptr != NULL) {
//		sw_mapping[ptr->id]=index;
//		ptr = ptr->next;
//	}
 */
static int* sw_mapping;
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
   Stack for the Tarjan algorithm (finding SCCs)
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

static SCC_Node* nodes;
static SCC_Stack* stack;
static int scc_num;
static SCC* sccs;
static int nl_debug_level;

/*
   This function compute equations in a given SCC(whose ID is (int)scc) by given values((double*) x)
   and store it to (double*) o.
   ex.
   given explanation graph
//	 a
//	  <=> b & b
//	    v c & c
converted to following equations
//	a=b^2+c^2
this function compute
//	-a+b^2+c^2
and store it to (double*) o
 */
void compute_scc_eq(double* x,double* o,int scc) {
	int i;
	EG_NODE_PTR eg_ptr;
	for(i=0; i<sccs[scc].size; i++) {
		int w=sccs[scc].el[i];
		eg_ptr = sorted_expl_graph[w];
		eg_ptr->inside=x[i];
	}
	for(i=0; i<sccs[scc].size; i++) {
		int w=sccs[scc].el[i];
		EG_PATH_PTR path_ptr;
		eg_ptr = sorted_expl_graph[w];
		path_ptr = eg_ptr->path_ptr;
		double sum=-eg_ptr->inside;
		if(path_ptr == NULL) {
			sum+=1;
		} else {
			while (path_ptr != NULL) {
				int k;
				double prob=1;
				for (k = 0; k < path_ptr->children_len; k++) {
					prob*=path_ptr->children[k]->inside;
				}
				for (k = 0; k < path_ptr->sws_len; k++) {
					prob*=path_ptr->sws[k]->inside;
				}
				sum+=prob;
				path_ptr = path_ptr->next;
			}
		}
		o[i]=sum;
	}
}

void progress_broyden(double* x,double* sk,int dim) {
	int i;
	printf("x=>> ");
	for(i=0; i<dim; i++) {
		printf("%f,",x[i]);
	}
	printf("\ns=>> ");
	for(i=0; i<dim; i++) {
		printf("%f,",sk[i]);
	}
	printf("\n");
}

/*
   This function solves equations corresponded to a target SCC by the Broyden's method.
scc: target scc_id
compute_scc_eq: function pointer to compute equations
 */
void solve(int scc,void(*compute_scc_eq)(double* x,double* o,int scc)) {
	int dim =sccs[scc].size;
	int loopCount;
	int maxloop=100;
	int restart=20;
	double** s=(double**)calloc(sizeof(double*),restart+1);
	double epsilon=1.0e-10;
	double epsilonX=1.0e-10;
	int i;
	for(i=0; i<restart+1; i++) {
		s[i]=(double*)calloc(sizeof(double),dim);
	}
	double* z=(double*)calloc(sizeof(double),dim);
	double* x=(double*)calloc(sizeof(double),dim);
	double* s0=s[0];
	for(i=0; i<dim; i++) {
		x[i]=rand()/(double)RAND_MAX;
	}
	compute_scc_eq(x,s0,scc);
	int s_cnt=0;
	int convergenceCount;
	for(loopCount=0; maxloop<0||loopCount<maxloop; loopCount++) {
		if(restart>0 && ((loopCount+1)%restart==0)) {
			//clear working area
			s_cnt=0;
			compute_scc_eq(x,s0,scc);
		}
		double* sk=s[s_cnt];
		convergenceCount=0;
		//progress_broyden(x,sk,dim);
		for(i=0; i<dim; i++) {
			x[i]+=sk[i];
			if((fabs(x[i])<epsilonX&&fabs(sk[i])<epsilon)||fabs(sk[i]/x[i])<epsilon) {
				convergenceCount++;
			}
		}
		if(convergenceCount==dim) {
			break;
		}
		compute_scc_eq(x,z,scc);
		int j;
		for(j=1; j<=s_cnt; j++) {
			double denom=0;
			double num=0;
			for(i=0; i<dim; i++) {
				num+=z[i]*s[j-1][i];
				denom+=s[j-1][i]*s[j-1][i];
			}
			for(i=0; i<dim; i++) {
				if(denom!=0) {
					z[i]+=s[j][i]*num/denom;
				}
			}
		}
		//
		double c=0;
		{
			double denom=0;
			double num=0;
			for(i=0; i<dim; i++) {
				num+=z[i]*sk[i];
				denom+=sk[i]*sk[i];
			}
			if(denom!=0) {
				c=num/denom;
			} else {
				c=0;
				emit_error("Broyden's method error : underflow");
			}
		}
		double* sk1=s[s_cnt+1];
		for(i=0; i<dim; i++) {
			if(1.0-c!=0) {
				sk1[i]=z[i]/(1.0-c);
			} else {
				emit_error("Broyden's method error : C is one");
			}
		}
		s_cnt++;
	}
	for(i=0; i<dim; i++) {
		int w=sccs[scc].el[i];
		sorted_expl_graph[w]->inside=x[i];
	}
	for(i=0; i<restart+1; i++) {
		free(s[i]);
	}
	free(s);
	free(z);
	free(x);

	if(nl_debug_level>=2) {
		printf("convergence :%d/%d\n",convergenceCount,dim);
		if(maxloop<0) {
			printf("loop :%d/inf\n",loopCount);
		} else {
			printf("loop :%d/%d\n",loopCount,maxloop);
		}
	}
}

/*
   Auxiliary function for recusive call in the Tarjan algorithm
 */
void get_scc_tarjan_start(int i,int index) {
	nodes[i].visited=1;
	nodes[i].index=index;
	nodes[i].lowlink=index;
	nodes[i].in_stack=1;
	index++;
	SCC_Stack* el=(SCC_Stack*)malloc(sizeof(SCC_Stack));
	if(stack==NULL) {
		el->next=NULL;
		stack=el;
	} else {
		el->next=stack;
		stack=el;
	}
	el->index=i;
	EG_NODE_PTR eg_ptr;
	EG_PATH_PTR path_ptr;
	eg_ptr = sorted_expl_graph[i];
	path_ptr = eg_ptr->path_ptr;
	while (path_ptr != NULL) {
		int k;
		for (k = 0; k < path_ptr->children_len; k++) {
			//something todo
			int w;
			w=mapping[path_ptr->children[k]->id];
			if(nodes[w].visited==0) {
				get_scc_tarjan_start(w,index);
				int il=nodes[i].lowlink;
				int wl=nodes[w].lowlink;
				nodes[i].lowlink=il<wl?il:wl;
			} else if(nodes[w].in_stack) { //visited==2
				int il=nodes[i].lowlink;
				int wi=nodes[w].index;
				nodes[i].lowlink=il<wi?il:wi;
			}
		}
		path_ptr = path_ptr->next;
	}
	if(nodes[i].lowlink==nodes[i].index) {
		int w;
		SCC_Stack* itr=stack;
		int cnt=0;
		do {
			w=itr->index;
			itr=itr->next;
			cnt++;
		} while(w!=i);
		sccs[scc_num].size=cnt;
		sccs[scc_num].el=(int*)calloc(sizeof(int),cnt);
		cnt=0;
		do {
			w=stack->index;
			nodes[w].visited=2;
			nodes[w].in_stack=0;
			sccs[scc_num].el[cnt]=w;
			SCC_Stack* temp=stack;
			stack=stack->next;
			free(temp);
			cnt++;
		} while(w!=i);
		scc_num++;
	}
}

/*
   This function computes the reachability from i to j by DFS on the explanation graph
   0 <= i,j <= sorted_egraph_size
 */
int is_reachable(int i,int j) {
	EG_NODE_PTR eg_ptr;
	EG_PATH_PTR path_ptr;
	if(nodes[i].visited==1) {
		return 0;
	}
	nodes[i].visited=1;
	//
	eg_ptr = sorted_expl_graph[i];
	path_ptr = eg_ptr->path_ptr;
	while (path_ptr != NULL) {
		int k;
		for (k = 0; k < path_ptr->children_len; k++) {
			int w;
			w=mapping[path_ptr->children[k]->id];
			if(w==j || is_reachable(w,j)) {
				return 1;
			}
		}
		path_ptr=path_ptr->next;
	}
	return 0;
}
int check_scc(int scc_id) {
	int n=sccs[scc_id].size;
	int i,j,k;
	for(i=0; i<n; i++) {
		for(j=0; j<n; j++) {
			for(k=0; k<sorted_egraph_size; k++) {
				nodes[k].visited=0;
			}
			if(i!=j) {
				if(!is_reachable(sccs[scc_id].el[i],sccs[scc_id].el[j])) {
					emit_error("SCC reachable error: %d:%d(%d)->%d(%d)",scc_id,sccs[scc_id].el[i],i,sccs[scc_id].el[j],j);
					return 1;
				}
			}
		}
	}
	return 0;
}
int check_scc_order() {
	int err=0;
	EG_NODE_PTR eg_ptr;
	EG_PATH_PTR path_ptr;
	int l;
	for(l=0; l<sorted_egraph_size; l++) {
		nodes[l].visited=0;
	}
	int i;
	for(i=0; i<scc_num; i++) {
		int n=sccs[i].size;
		int j;
		for(j=0; j<n; j++) {
			int index=sccs[i].el[j];
			nodes[index].visited=1;
		}
		for(j=0; j<n; j++) {
			int index=sccs[i].el[j];
			eg_ptr = sorted_expl_graph[index];
			path_ptr = eg_ptr->path_ptr;
			while (path_ptr != NULL) {
				int k;
				for (k = 0; k < path_ptr->children_len; k++) {
					int w;
					w=mapping[path_ptr->children[k]->id];
					if(nodes[w].visited==0) {
						//emit_error("SCC order error");
						err++;
					}
				}
				path_ptr=path_ptr->next;
			}
		}
	}
	return err;
}
void reset_visit() {
	int l;
	for(l=0; l<sorted_egraph_size; l++) {
		nodes[l].visited=0;
	}
}
void print_sccs() {
	EG_NODE_PTR eg_ptr;
	EG_PATH_PTR path_ptr;
	int i;
	reset_visit();
	for(i=0; i<scc_num; i++) {
		int n=sccs[i].size;
		int j;
		for(j=0; j<n; j++) {
			int index=sccs[i].el[j];
			printf("%d:",index);
			eg_ptr = sorted_expl_graph[index];
			path_ptr = eg_ptr->path_ptr;
			while (path_ptr != NULL) {
				int k;
				int enable=0;
				for (k = 0; k < path_ptr->children_len; k++) {
					int w;
					w=mapping[path_ptr->children[k]->id];
					if(nodes[w].visited==0) {
						if(enable) {
							printf(",%d",w);
						} else {
							printf("(%d",w);
							enable=1;
						}
					}
				}
				if(enable) {
					printf(")");
				}
				path_ptr=path_ptr->next;
			}
			printf("\n");
		}
		for(j=0; j<n; j++) {
			int index=sccs[i].el[j];
			nodes[index].visited=1;
		}
	}
	return;
}
/*
   This function finds SCCs by the Tarjan algorithms and store SCCs to (SCC *) sccs.
 */
void get_scc_tarjan() {
	int i;
	nodes=(SCC_Node*)calloc(sizeof(SCC_Node),sorted_egraph_size);
	sccs=(SCC*)calloc(sizeof(SCC),sorted_egraph_size);
	get_scc_tarjan_start(0,0);
	while(1) {
		int next_i=0;
		int cnt=0;
		for(i=0; i<sorted_egraph_size; i++) {
			if(nodes[i].visited!=2) {
				nodes[i].visited=0;
				nodes[i].in_stack=0;
				if(next_i==0) {
					next_i=i;
				}
			} else {
				cnt++;
			}
		}
		if(cnt!=sorted_egraph_size) {
			get_scc_tarjan_start(next_i,cnt);
		} else {
			break;
		}
	}
	//check SCCs
	if(nl_debug_level>0) {
		printf("check SCCs\n");
		if(stack!=NULL) {
			emit_error("Tarjan's algorithm error:remaining stack");
		}
		int error_num=0;
		for(i=0; i<scc_num; i++) {
			error_num+=check_scc(i);
		}
		if(error_num>0) {
			printf("SCC error:%d/%d\n",scc_num-error_num,scc_num);
		} else {
			printf("[OK] SCCs\n");
		}
		error_num=check_scc_order();
		if(error_num>0) {
			printf("SCC order error\n");
		} else {
			printf("[OK] SCC order\n");
		}
	}
}
void print_eq() {
	int i,k;
	EG_NODE_PTR eg_ptr;
	EG_PATH_PTR path_ptr;
	printf("graph_size:%d\n",sorted_egraph_size);
	printf("equations\n");
	for (i = 0; i < sorted_egraph_size; i++) {
		eg_ptr = sorted_expl_graph[i];
		path_ptr = eg_ptr->path_ptr;
		printf("# x[%d] : ",i);
		while (path_ptr != NULL) {
			int first_s=1;
			for (k = 0; k < path_ptr->sws_len; k++) {
				if(first_s) {
					first_s=0;
				} else {
					printf("*");
				}
				printf("%3.3f",path_ptr->sws[k]->inside);
			}
			for (k = 0; k < path_ptr->children_len; k++) {
				if(first_s) {
					first_s=0;
				} else {
					printf("*");
				}
				printf("x[%d]",mapping[path_ptr->children[k]->id]);
			}
			if(first_s) {
				printf("1");
				first_s=0;
			}
			path_ptr = path_ptr->next;
			if(path_ptr!=NULL) {
				printf("+");
			}
		}
		printf("\n");
	}
}

void print_eq2() {
	int i,k;
	printf("sw_size:%d\n",occ_switch_tab_size);
	printf("sw_ins_size:%d\n",sw_ins_tab_size);
	EG_NODE_PTR eg_ptr;
	EG_PATH_PTR path_ptr;
	printf("graph_size:%d\n",sorted_egraph_size);
	printf("equations\n");
	for (i = 0; i < sorted_egraph_size; i++) {
		eg_ptr = sorted_expl_graph[i];
		path_ptr = eg_ptr->path_ptr;
		printf("# x[%d] : ",i);
		while (path_ptr != NULL) {
			int first_s=1;
			for (k = 0; k < path_ptr->sws_len; k++) {
				if(first_s) {
					first_s=0;
				} else {
					printf("*");
				}
				printf("%3.3f",path_ptr->sws[k]->inside);
				printf("(t[%d])",path_ptr->sws[k]->id);
			}
			for (k = 0; k < path_ptr->children_len; k++) {
				if(first_s) {
					first_s=0;
				} else {
					printf("*");
				}
				printf("x[%d]",mapping[path_ptr->children[k]->id]);
			}
			if(first_s) {
				printf("1");
				first_s=0;
			}
			path_ptr = path_ptr->next;
			if(path_ptr!=NULL) {
				printf("+");
			}
		}
		printf("\n");
	}
}

typedef struct _EqTerm {
	int id;
	double coef;
	struct _EqTerm* next;
} EqTerm;

typedef struct _Equations {
	double c;
	EqTerm* term;
	struct _Equations* next;
} Equations;
static Equations** equations;
static double* grad;
static int* occ_sw_id;
void free_equations(Equations* eq) {
	EqTerm* t=eq->term;
	while(t!=NULL) {
		EqTerm* temp=t->next;
		free(t);
		t=temp;
	}
	free(eq);
}
void print_equation(Equations* eq) {
	EqTerm* t=eq->term;
	while(t!=NULL) {
		printf("%3.3fx[%d]",t->coef,t->id);
		printf("+");
		t=t->next;
	}
	printf("%3.3f",eq->c);
	printf("\n");
}
void compute_scc_equations(double* x,double* o,int scc) {
	int i;
	Equations* eq_ptr;
	for(i=0; i<sccs[scc].size; i++) {
		int w=sccs[scc].el[i];
		eq_ptr = equations[w];
		grad[w]=x[i];
	}
	for(i=0; i<sccs[scc].size; i++) {
		int w=sccs[scc].el[i];
		eq_ptr = equations[w];
		EqTerm* term_ptr = eq_ptr->term;
		double sum=eq_ptr->c-grad[w];
		while (term_ptr != NULL) {
			sum+=term_ptr->coef*grad[mapping[term_ptr->id]];
			term_ptr = term_ptr->next;
		}
		o[i]=sum;
	}
}



Equations* get_equation(EG_NODE_PTR eg_ptr,int sw_id,int sw_ins_id) {
	Equations *eq=(Equations*)calloc(sizeof(Equations),1);
	EqTerm *t_itr=NULL;
	int k,l;
	EG_PATH_PTR path_ptr;
	path_ptr = eg_ptr->path_ptr;
	double sum=0;
	while (path_ptr != NULL) {
		double coef_sum=0.0;
		double x=1.0;
		// (c)'x
		// x
		for (k = 0; k < path_ptr->children_len; k++) {
			x*=path_ptr->children[k]->inside;
		}
		for (k = 0; k < path_ptr->sws_len; k++) {
			char enable=0;//0:disable,1:direct variable,2:indirect variable
			int id=path_ptr->sws[k]->id;
			SW_INS_PTR sw_ptr = switch_instances[sw_ins_id];
			if(id==sw_ins_id) {
				enable=1;
			} else if(sw_mapping[id]==sw_id) {
				enable=2;
			}
			if(enable>0) {
				double coef=1.0;
				//k
				if(enable==1) {
					coef*=-(1-path_ptr->sws[k]->inside);
				} else if(enable==2) {
					coef*=(sw_ptr->inside);
				}
				//l
				for (l = 0; l < path_ptr->sws_len; l++) {
					coef*=path_ptr->sws[l]->inside;
				}
				coef_sum+=coef*x;
			}
			sum+=coef_sum;
		}
		// c(x)'


		if(path_ptr->children_len>0) {
			for (k = 0; k < path_ptr->children_len; k++) {
				EqTerm *t=(EqTerm*)calloc(sizeof(EqTerm),1);
				double coef=1.0;
				for (l = 0; l < path_ptr->children_len; l++) {
					if(k!=l) {
						coef*=path_ptr->children[k]->inside;
					}
				}
				{
					t->id=mapping[path_ptr->children[k]->id];
				}
				t->coef=coef;
				if(t_itr!=NULL) {
					t_itr->next=t;
				} else {
					eq->term=t;
				}
				t_itr=t;
			}
		}
		//
		path_ptr=path_ptr->next;
	}
	eq->c=sum;
	return eq;
}

extern "C"
int pc_nonlinear_eq_2(void) {
	int i;
	EG_NODE_PTR eg_ptr;

	nodes=NULL;
	stack=NULL;
	scc_num=0;
	sccs=NULL;
	mapping=NULL;

	int max_id=0;
	nl_debug_level = bpx_get_integer(bpx_get_call_arg(1,2));
	double start_time=getCPUTime();

	for (i = 0; i < sorted_egraph_size; i++) {
		eg_ptr = sorted_expl_graph[i];
		if(max_id<eg_ptr->id) {
			max_id=eg_ptr->id;
		}
	}
	mapping=(int*)calloc(sizeof(int),max_id+1);
	for (i = 0; i < sorted_egraph_size; i++) {
		eg_ptr = sorted_expl_graph[i];
		mapping[eg_ptr->id]=i;
	}

	//print_eq
	if(nl_debug_level>=3) {
		print_eq();
	}
	//find scc
	get_scc_tarjan();
	if(nl_debug_level>=3) {
		print_sccs();
	}
	double scc_time=getCPUTime();
	//solution
	if(nl_debug_level>=2) {
		printf("non-linear solver: Broyden's method\n");
	}
	for(i=0; i<scc_num; i++) {
		solve(i,compute_scc_eq);
		if(nl_debug_level>=2) {
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
	free(nodes);
	for(i=0; i<scc_num; i++) {
		free(sccs[i].el);
	}
	free(sccs);
	free(mapping);
	double prob=sorted_expl_graph[sorted_egraph_size-1]->inside;
	if(nl_debug_level>=1) {
		printf("CPU time (scc,solution,all)\n");
		printf("# %f,%f,%f\n",scc_time-start_time,solution_time-scc_time,solution_time - start_time);
	}
	return bpx_unify(bpx_get_call_arg(2,2),
			bpx_build_float(prob));
}

int run_cyc_em() {
/*
	int	 r, iterate, old_valid, converged, saved = 0;
	double  likelihood, log_prior;
	double  lambda, old_lambda = 0.0;

	//config_em(em_ptr);

	for (r = 0; r < num_restart; r++) {
		SHOW_PROGRESS_HEAD("#cyc-em-iters", r);
		initialize_params();
		iterate = 0;
		while (1) {
			old_valid = 0;
			while (1) {
				if (CTRLC_PRESSED) {
					SHOW_PROGRESS_INTR();
					RET_ERR(err_ctrl_c_pressed);
				}

				//RET_ON_ERR(em_ptr->compute_inside());
				//RET_ON_ERR(em_ptr->examine_inside());

				//likelihood = em_ptr->compute_likelihood();
				//log_prior  = em_ptr->smooth ? em_ptr->compute_log_prior() : 0.0;
				lambda = likelihood + log_prior;

				if (verb_em) {
					//if (em_ptr->smooth) {
					prism_printf("iteration #%d:\tlog_likelihood=%.9f\tlog_prior=%.9f\tlog_post=%.9f\n", iterate, likelihood, log_prior, lambda);
					//}
					//else {
					prism_printf("iteration #%d:\tlog_likelihood=%.9f\n", iterate, likelihood);
					//}
				}

				if (!isfinite(lambda)) {
					emit_internal_error("invalid log likelihood or log post: %s (at iteration #%d)",
							isnan(lambda) ? "NaN" : "infinity", iterate);
					RET_ERR(ierr_invalid_likelihood);
				}
				if (old_valid && old_lambda - lambda > prism_epsilon) {
					emit_error("log likelihood or log post decreased [old: %.9f, new: %.9f] (at iteration #%d)",
							old_lambda, lambda, iterate);
					RET_ERR(err_invalid_likelihood);
				}

				converged = (old_valid && lambda - old_lambda <= prism_epsilon);
				if (converged || REACHED_MAX_ITERATE(iterate)) {
					break;
				}

				old_lambda = lambda;
				old_valid  = 1;

				//RET_ON_ERR(em_ptr->compute_expectation());

				SHOW_PROGRESS(iterate);
				//RET_ON_ERR(em_ptr->update_params());
				iterate++;
			}
		}

		SHOW_PROGRESS_TAIL(converged, iterate, lambda);

		saved = (r < num_restart - 1);
		if (saved) {
			save_params();
		}

	}

	if (saved) {
		restore_params();
	}

	//em_ptr->bic = compute_bic(em_ptr->likelihood);
	//em_ptr->cs  = em_ptr->smooth ? compute_cs(em_ptr->likelihood) : 0.0;
*/
	return BP_TRUE;
}

void solve_o(int scc,void(*compute_scc_eq)(double* x,double* o,int scc),double* x) {
	int dim =sccs[scc].size;
	int loopCount;
	int maxloop=100;
	int restart=20;
	double** s=(double**)calloc(sizeof(double*),restart+1);
	double epsilon=1.0e-10;
	double epsilonX=1.0e-10;
	int i;
	for(i=0; i<restart+1; i++) {
		s[i]=(double*)calloc(sizeof(double),dim);
	}
	double* z=(double*)calloc(sizeof(double),dim);
	double* s0=s[0];
	for(i=0; i<dim; i++) {
		x[i]=rand()/(double)RAND_MAX;
	}
	compute_scc_eq(x,s0,scc);
	int s_cnt=0;
	int convergenceCount;
	for(loopCount=0; maxloop<0||loopCount<maxloop; loopCount++) {
		if(restart>0 && ((loopCount+1)%restart==0)) {
			//clear working area
			s_cnt=0;
			compute_scc_eq(x,s0,scc);
		}
		double* sk=s[s_cnt];
		convergenceCount=0;
		//progress_broyden(x,sk,dim);
		for(i=0; i<dim; i++) {
			x[i]+=sk[i];
			if((fabs(x[i])<epsilonX&&fabs(sk[i])<epsilon)||fabs(sk[i]/x[i])<epsilon) {
				convergenceCount++;
			}
		}
		if(convergenceCount==dim) {
			break;
		}
		compute_scc_eq(x,z,scc);
		int j;
		for(j=1; j<=s_cnt; j++) {
			double denom=0;
			double num=0;
			for(i=0; i<dim; i++) {
				num+=z[i]*s[j-1][i];
				denom+=s[j-1][i]*s[j-1][i];
			}
			for(i=0; i<dim; i++) {
				if(denom!=0) {
					z[i]+=s[j][i]*num/denom;
				}
			}
		}
		//
		double c=0;
		{
			double denom=0;
			double num=0;
			for(i=0; i<dim; i++) {
				num+=z[i]*sk[i];
				denom+=sk[i]*sk[i];
			}
			if(denom!=0) {
				c=num/denom;
			} else {
				c=0;
				emit_error("Broyden's method error : underflow");
			}
		}
		double* sk1=s[s_cnt+1];
		for(i=0; i<dim; i++) {
			if(1.0-c!=0) {
				sk1[i]=z[i]/(1.0-c);
			} else {
				printf("[%d,%f]",i,z[i]);
				emit_error("Broyden's method error : C is one");
			}
		}
		s_cnt++;
	}
	for(i=0; i<restart+1; i++) {
		free(s[i]);
	}
	free(s);
	free(z);

	if(nl_debug_level>=2) {
		printf("convergence :%d/%d\n",convergenceCount,dim);
		if(maxloop<0) {
			printf("loop :%d/inf\n",loopCount);
		} else {
			printf("loop :%d/%d\n",loopCount,maxloop);
		}
	}
}

#include <stdio.h>
#include <lbfgs.h>
static int occ_switch_size=0;
static double* lbfgs_sw_space;
void update_sw(const double* x,const int n) {
	int i;
	for (i = 0; i < occ_switch_tab_size; i++) {
		lbfgs_sw_space[i]=0;
	}
	for (i = 0; i < n; i++) {
		SW_INS_PTR ptr = switch_instances[occ_sw_id[i]];
		int w=sw_mapping[ptr->id];
		lbfgs_sw_space[w]+=exp(-x[i]);
	}
	for (i = 0; i < n; i++) {
		SW_INS_PTR ptr = switch_instances[occ_sw_id[i]];
		int w=sw_mapping[ptr->id];
		ptr->inside=exp(-x[i])/lbfgs_sw_space[w];
	}

}
static lbfgsfloatval_t evaluate(
		void *instance,
		const lbfgsfloatval_t *x,
		lbfgsfloatval_t *g,
		const int n,
		const lbfgsfloatval_t step
		) {
	int i;
	lbfgsfloatval_t fx = 0.0;
	update_sw(x,n);
	for(i=0; i<scc_num; i++) {
		solve(i,compute_scc_eq);
	}
	for (i = 0; i < num_roots; i++) {
		int w = mapping[roots[i]->id];
		//printf("%f,",sorted_expl_graph[w]->inside);
		fx+=log(sorted_expl_graph[w]->inside);
	}
	//printf("\n");
	int k;
	for (k = 0; k < n; k++) {
		SW_INS_PTR ptr = switch_instances[occ_sw_id[k]];
		for (i = 0; i < sorted_egraph_size; i++) {
			equations[i]=get_equation(sorted_expl_graph[i],sw_mapping[ptr->id],ptr->id);
		}
		int j,scc;
		for(scc=0; scc<scc_num; scc++) {
			double* x=(double*)calloc(sizeof(double),sccs[scc].size);
			solve_o(scc,compute_scc_equations,x);
			for(j=0; j<sccs[scc].size; j++) {
				int w=sccs[scc].el[j];
				grad[w]=x[j];
			}
			free(x);
		}
		double grad_l=0;
		for (i = 0; i < num_roots; i++) {
			int w = mapping[roots[i]->id];
			grad_l+=grad[w]/sorted_expl_graph[w]->inside;
		}
		for (i = 0; i < sorted_egraph_size; i++) {
			free_equations(equations[i]);
		}
		g[k]=grad_l;
	}
	//
	return fx;
}

static int progress(
		void *instance,
		const lbfgsfloatval_t *x,
		const lbfgsfloatval_t *g,
		const lbfgsfloatval_t fx,
		const lbfgsfloatval_t xnorm,
		const lbfgsfloatval_t gnorm,
		const lbfgsfloatval_t step,
		int n,
		int k,
		int ls
		) {
	printf("Iteration %d:\n", k);
	printf("  fx = %f, x[0] = %f, x[1] = %f\n", fx, x[0], x[1]);
	printf("  xnorm = %f, gnorm = %f, step = %f\n", xnorm, gnorm, step);
	printf("\n");
	return 0;
}

extern "C"
int pc_cyc_em_6(void) {
	int i;
	EG_NODE_PTR eg_ptr;

	nodes=NULL;
	stack=NULL;
	scc_num=0;
	sccs=NULL;
	mapping=NULL;
	sw_mapping=NULL;

	int max_id=0;
	nl_debug_level = 0;
	double start_time=getCPUTime();

	for (i = 0; i < sorted_egraph_size; i++) {
		eg_ptr = sorted_expl_graph[i];
		if(max_id<eg_ptr->id) {
			max_id=eg_ptr->id;
		}
	}
	mapping=(int*)calloc(sizeof(int),max_id+1);
	for (i = 0; i < sorted_egraph_size; i++) {
		eg_ptr = sorted_expl_graph[i];
		mapping[eg_ptr->id]=i;
	}
	sw_mapping=(int*)calloc(sizeof(int),sw_ins_tab_size);
	for (i = 0; i < occ_switch_tab_size; i++) {
		SW_INS_PTR ptr;
		ptr = occ_switches[i];
		while (ptr != NULL) {
			occ_switch_size++;
			sw_mapping[ptr->id]=i;
			ptr = ptr->next;
		}
	}
	grad=(double*)calloc(sizeof(double),sorted_egraph_size);
	//
	//print_eq
	print_eq2();
	//find scc
	get_scc_tarjan();
	print_sccs();
	double scc_time=getCPUTime();
	//ret_on_err(check_smooth(&em_eng.smooth));
	//ret_on_err(run_cyc_em());
	//solution
	for(i=0; i<scc_num; i++) {
		solve(i,compute_scc_eq);
		int n=sccs[i].size;
		int j;
		for(j=0; j<n; j++) {
			int w=sccs[i].el[j];
			printf("%d:%f\n",w,sorted_expl_graph[w]->inside);
		}
	}
	double solution_time=getCPUTime();
	equations=(Equations**)calloc(sizeof(Equations*),sorted_egraph_size);
	occ_sw_id=(int*)calloc(sizeof(int),occ_switch_size);
	int j=0;
	for (i = 0; i < occ_switch_tab_size; i++) {
		SW_INS_PTR ptr = occ_switches[i];
		while (ptr != NULL) {
			occ_sw_id[j]=ptr->id;
			j++;
			ptr = ptr->next;
		}
	}

	//
	lbfgs_sw_space=(double*)calloc(sizeof(double),occ_switch_tab_size);
	/*
	   for (k = 0; k < occ_switch_size; k++) {
	   SW_INS_PTR ptr = switch_instances[occ_sw_id[k]];
	   printf("diff_eq %d\n",ptr->id);
	   for (i = 0; i < sorted_egraph_size; i++) {
	   eg_ptr = sorted_expl_graph[i];
	   equations[i]=get_equation(eg_ptr,sw_mapping[ptr->id],ptr->id);
	   print_equation(equations[i]);
	   }
	   int j,scc;
	   for(scc=0;scc<scc_num;scc++){
	   double* x=calloc(sizeof(double),sccs[scc].size);
	   solve_o(scc,compute_scc_equations,x);
	   for(j=0;j<sccs[scc].size;j++){
	   int w=sccs[scc].el[j];
	   grad[w]=x[j];
	   }
	   free(x);
	   }
	   for (j = 0; j < sorted_egraph_size; j++) {
	   printf("g[%d]=%f\n",j,grad[j]);
	   }
	   double grad_l=0;
	   printf("g L:");
	   for (i = 0; i < num_roots; i++) {
	   int w = mapping[roots[i]->id];
	   grad_l+=grad[w]/sorted_expl_graph[w]->inside;
	   printf("g[%d]",w,grad[j]);
	   }
	   printf("\n");
	   for (i = 0; i < sorted_egraph_size; i++) {
	   free_equations(equations[i]);
	   }
	   }*/
	{
		lbfgsfloatval_t fx;
		lbfgsfloatval_t *x = (lbfgsfloatval_t*)lbfgs_malloc(occ_switch_size);
		lbfgs_parameter_t param;
		if (x == NULL) {
			printf("ERROR: Failed to allocate a memory block for variables.\n");
			return BP_FALSE;
		}
		for (i = 0; i < occ_switch_size; i ++) {
			x[i]=0;
		}
		/* Initialize the parameters for the L-BFGS optimization. */
		lbfgs_parameter_init(&param);
		/*param.linesearch = LBFGS_LINESEARCH_BACKTRACKING;*/
		//
		int ret = lbfgs(occ_switch_size, x, &fx, evaluate, progress, NULL, &param);
		printf("L-BFGS optimization terminated with status code = %d\n", ret);
		printf("  fx = %f,", fx);
		for(i=0; i<occ_switch_size; i++) {
			printf("x[%d] = %f,",i,x[i]);
		}
		printf("\n");
		update_sw(x,occ_switch_size);
		lbfgs_free(x);
	}
	free(lbfgs_sw_space);
	//
	free(equations);
	free(occ_sw_id);
	//free data
	free(nodes);
	for(i=0; i<scc_num; i++) {
		free(sccs[i].el);
	}
	free(sccs);
	free(mapping);
	free(sw_mapping);
	free(grad);
	//double prob=sorted_expl_graph[sorted_egraph_size-1]->inside;
	if(1) {
		printf("CPU time (scc,solution,all)\n");
		printf("# %f,%f,%f\n",scc_time-start_time,solution_time-scc_time,solution_time - start_time);
	}
	return
		bpx_unify(bpx_get_call_arg(1,6), bpx_build_integer(1)) &&
		bpx_unify(bpx_get_call_arg(2,6), bpx_build_float  (0)) &&
		bpx_unify(bpx_get_call_arg(3,6), bpx_build_float  (0)) &&
		bpx_unify(bpx_get_call_arg(4,6), bpx_build_float  (0)) &&
		bpx_unify(bpx_get_call_arg(5,6), bpx_build_float  (0)) &&
		bpx_unify(bpx_get_call_arg(6,6), bpx_build_integer(0)) ;
}


