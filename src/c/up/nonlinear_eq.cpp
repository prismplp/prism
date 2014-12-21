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
#include "nonlinear_eq.h"
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
//static int* sw_mapping;
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
and store it to (double*) out
 */
void compute_scc_eq(double* x,double* out,int scc) {
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
		out[i]=sum;
	}
}

/*
This function computes the inside probability within scc ignoring cycles.
scc: target scc_id
 */
void update_inside_scc(int scc) {
	int i;
	EG_NODE_PTR eg_ptr;
	for(i=0; i<sccs[scc].size; i++) {
		int w=sccs[scc].el[i];
		EG_PATH_PTR path_ptr;
		eg_ptr = sorted_expl_graph[w];
		path_ptr = eg_ptr->path_ptr;
		double sum=0;
		if(path_ptr == NULL) {
			sum=1;
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
				path_ptr->inside=prob;
				sum+=prob;
				path_ptr = path_ptr->next;
			}
		}
		eg_ptr->inside=sum;
	}
}

/*
scc: target scc_id
id: sorted_expl_graph id
retrun: true if scc includes id 
*/
//TODO: binary search
int is_scc_element(int scc,int id){
	int i;
	for(i=0; i<sccs[scc].size; i++) {
		if(sccs[scc].el[i]==id){
			return 1;
		}
	}
	return 0;
}

/*
scc: target scc_id
index: sorted_expl_graph id
retrun: scc index (-1=not found)
 */
//TODO: binary search
int get_scc_element_index(int scc,int id){
	int i;
	for(i=0; i<sccs[scc].size; i++) {
		if(sccs[scc].el[i]==id){
			return i;
		}
	}
	return -1;
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
void print_eq(){
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
				printf("%3.3f(%d)",path_ptr->sws[k]->inside,path_ptr->sws[k]->id);
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
void print_eq_outside(){
	int i,k;
	EG_NODE_PTR eg_ptr;
	EG_PATH_PTR path_ptr;
	printf("graph_size:%d\n",sorted_egraph_size);
	printf("equations\n");
	for (i = 0; i < sorted_egraph_size; i++) {
		eg_ptr = sorted_expl_graph[i];
		path_ptr = eg_ptr->path_ptr;
		printf("# (%3.3f)x[%d] : ",eg_ptr->outside,i);
		while (path_ptr != NULL) {
			int first_s=1;
			for (k = 0; k < path_ptr->sws_len; k++) {
				if(first_s) {
					first_s=0;
				} else {
					printf("*");
				}
				printf("%3.3f(%d)",path_ptr->sws[k]->total_expect,path_ptr->sws[k]->id);
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


void set_scc_order() {
	int scc,i;
	EG_NODE_PTR eg_ptr;
	for(scc=0; scc<scc_num; scc++) {
		for(i=0; i<sccs[scc].size; i++) {
			int w=sccs[scc].el[i];
			int max_order=0;
			EG_PATH_PTR path_ptr;
			eg_ptr = sorted_expl_graph[w];
			path_ptr = eg_ptr->path_ptr;
			while (path_ptr != NULL) {
				int order=0;
				int k;
				for (k = 0; k < path_ptr->children_len; k++) {
					int index=mapping[path_ptr->children[k]->id];
					if(is_scc_element(scc,index)){
						order+=1;
					}
				}
				if(order>max_order){
					max_order=order;
				}
				path_ptr = path_ptr->next;
			}
			sccs[scc].order=max_order;
		}
	}
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
/*
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
   void print_equations(Equations* eq) {
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

*/
double compute_nonlinear_viterbi(int nl_debug_level) {
	int i;
	EG_NODE_PTR eg_ptr;
	EG_PATH_PTR path_ptr;

	nodes=NULL;
	stack=NULL;
	scc_num=0;
	sccs=NULL;
	mapping=NULL;

	int max_id=0;
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
		printf("non-linear viterbi: \n");
	}
	{//init
		for (i = 0; i < sorted_egraph_size; i++) {
			eg_ptr = sorted_expl_graph[i];
			eg_ptr->max_path=NULL;
			path_ptr = eg_ptr->path_ptr;
			while(path_ptr != NULL) {
				int k;
				for (k = 0; k < path_ptr->children_len; k++) {
					if (log_scale) {
						path_ptr->children[k]->max=-HUGE_PROB;
					}else{
						path_ptr->children[k]->max=0.0;
					}
				}
				path_ptr=path_ptr->next;
			}
		}
	}
	for(i=0; i<scc_num; i++) {
		int n=sccs[i].size;
		int j;
		int cnt;
		int bf_loop;
		bf_loop=n>1?n-1:1;
		for(cnt=0; cnt<bf_loop; cnt++) {
			for(j=0; j<n; j++) {
				int w=sccs[i].el[j];
				eg_ptr = sorted_expl_graph[w];
				path_ptr = eg_ptr->path_ptr;
				double max_p;
				EG_PATH_PTR max_path = NULL;
				//initialize max_p
				max_p= 0.0;
				if(path_ptr == NULL) {
					if (log_scale) {
						max_p=0.0;
					}else{
						max_p=1.0;
					}
				}else if(sorted_expl_graph[w]->max_path!=NULL){
					max_p=sorted_expl_graph[w]->max;
					max_path=sorted_expl_graph[w]->max_path;
				}
				while(path_ptr != NULL) {
					int k;
					double this_path_max;
					if (log_scale) {
						this_path_max=0.0;
					}else{
						this_path_max=1.0;
					}
					if (log_scale) {
						for (k = 0; k < path_ptr->children_len; k++) {
							this_path_max += path_ptr->children[k]->max;
						}
						for (k = 0; k < path_ptr->sws_len; k++) {
							this_path_max += log(path_ptr->sws[k]->inside);
						}
					}else{
						for (k = 0; k < path_ptr->children_len; k++) {
							this_path_max *= path_ptr->children[k]->max;
						}
						for (k = 0; k < path_ptr->sws_len; k++) {
							this_path_max *= path_ptr->sws[k]->inside;
						}
					}
					if (max_path==NULL || this_path_max > max_p) {
						max_p = this_path_max;
						max_path = path_ptr;
					}
					path_ptr->max = this_path_max;
					path_ptr=path_ptr->next;
				}
				sorted_expl_graph[w]->max = max_p;
				sorted_expl_graph[w]->max_path = max_path;
				if(nl_debug_level>=3) {
					printf("%d:%f\n",w,sorted_expl_graph[w]->max);
				}
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
	double prob=sorted_expl_graph[sorted_egraph_size-1]->max;
	if(nl_debug_level>=1) {
		printf("CPU time (scc,solution,all)\n");
		printf("# %f,%f,%f\n",scc_time-start_time,solution_time-scc_time,solution_time - start_time);
	}
	return prob;
}

extern "C"
int pc_compute_nonlinear_viterbi_6(void) {
	TERM p_goal_path,p_subpath_goal,p_subpath_sw;
	int goal_id;
	double viterbi_prob;

	nl_debug_level = bpx_get_integer(bpx_get_call_arg(1,6));
	goal_id = bpx_get_integer(bpx_get_call_arg(2,6));

	initialize_egraph_index();
	alloc_sorted_egraph(1);
	/* INIT_MIN_MAX_NODE_NOS; */
	RET_ON_ERR(sort_one_egraph(goal_id,0,1));
	if (verb_graph) print_egraph(0,PRINT_NEUTRAL);

	compute_nonlinear_viterbi(nl_debug_level);

	if (debug_level) print_egraph(1,PRINT_VITERBI);

	get_most_likely_path(goal_id,&p_goal_path,&p_subpath_goal,
			&p_subpath_sw,&viterbi_prob);

	return
		bpx_unify(bpx_get_call_arg(3,6), p_goal_path)    &&
		bpx_unify(bpx_get_call_arg(4,6), p_subpath_goal) &&
		bpx_unify(bpx_get_call_arg(5,6), p_subpath_sw)   &&
		bpx_unify(bpx_get_call_arg(6,6), bpx_build_float(viterbi_prob));
}


#include "eigen/Core"
#include "eigen/LU"
#include <iostream>
#include <set>
#include <cmath>

using namespace Eigen;
void get_scc_matrix_inside(MatrixXd& out_a,VectorXd& out_b,int scc) {
	int i;
	EG_NODE_PTR eg_ptr;
	int n=sccs[scc].size;
	for(i=0; i<n; i++) {
		out_b[i]=0;
		int j;
		for(j=0; j<n; j++) {
			if(i==j){
				out_a(i,j)=-1.0;
			}else{
				out_a(i,j)=0;
			}
		}
	}
	for(i=0; i<sccs[scc].size; i++) {
		int w=sccs[scc].el[i];
		EG_PATH_PTR path_ptr;
		eg_ptr = sorted_expl_graph[w];
		path_ptr = eg_ptr->path_ptr;
		if(path_ptr == NULL) {
			//error
		} else {
			while (path_ptr != NULL) {
				int k,j=-1;
				double prob=1;
				for (k = 0; k < path_ptr->children_len; k++) {
					int index=mapping[path_ptr->children[k]->id];
					if(is_scc_element(scc,index)){
						j=get_scc_element_index(scc,index);
					}else{
						prob*=path_ptr->children[k]->inside;
					}
				}
				for (k = 0; k < path_ptr->sws_len; k++) {
					prob*=path_ptr->sws[k]->inside;
				}
				if(j==-1){
					out_b(i)+=-prob;
				}else{
					out_a(i,j)+=prob;
				}

				path_ptr = path_ptr->next;
			}
		}
	}
}
void get_scc_matrix_outside(MatrixXd& out_a,VectorXd& out_b,int scc,int target_node_index) {
	int i;
	EG_NODE_PTR eg_ptr;
	int n=sccs[scc].size;
	for(i=0; i<n; i++) {
		out_b[i]=0;
		int j;
		for(j=0; j<n; j++) {
			if(i==j){
				out_a(i,j)=-1.0;
			}else{
				out_a(i,j)=0;
			}
		}
	}
	for(i=0; i<sccs[scc].size; i++) {
		int w=sccs[scc].el[i];
		EG_PATH_PTR path_ptr;
		eg_ptr = sorted_expl_graph[w];
		path_ptr = eg_ptr->path_ptr;
		if(path_ptr == NULL) {
			//error
		} else {
			while (path_ptr != NULL) {
				int k,j=-1;
				double prob=1;
				int contain_target=0;
				for (k = 0; k < path_ptr->children_len; k++) {
					int index=mapping[path_ptr->children[k]->id];
					if(is_scc_element(scc,index)){//A
						j=get_scc_element_index(scc,index);
					}else{//b
						prob*=path_ptr->children[k]->inside;
						if(index==target_node_index){
							contain_target++;
						}
					}
				}
				for (k = 0; k < path_ptr->sws_len; k++) {
					prob*=path_ptr->sws[k]->inside;
				}
				if(j==-1){
					if(contain_target>0){
						out_b(i)+=-contain_target*(prob/sorted_expl_graph[target_node_index]->inside);
					}
				}else{
					out_a(i,j)+=prob;
				}
				path_ptr = path_ptr->next;
			}
		}
	}
}
void get_scc_matrix_outside_sws(MatrixXd& out_m,MatrixXd& out_m_prime,VectorXd& out_y,VectorXd& out_y_prime,int scc,int target_sw_id) {
	int i;
	EG_NODE_PTR eg_ptr;
	int n=sccs[scc].size;
	for(i=0; i<n; i++) {
		out_y(i)=0;
		out_y_prime(i)=0;
		for(int j=0; j<n; j++) {
			out_m(i,j)=0;
			out_m_prime(i,j)=0;
		}
	}
	for(i=0; i<sccs[scc].size; i++) {
		int w=sccs[scc].el[i];
		EG_PATH_PTR path_ptr;
		eg_ptr = sorted_expl_graph[w];
		path_ptr = eg_ptr->path_ptr;
		if(path_ptr == NULL) {
			//error
		} else {
			while (path_ptr != NULL) {
				int k,j=-1;
				double prob=1;
				for (k = 0; k < path_ptr->children_len; k++) {
					int index=mapping[path_ptr->children[k]->id];
					if(is_scc_element(scc,index)){//A
						j=get_scc_element_index(scc,index);
					}else{//b
						prob*=path_ptr->children[k]->inside;
					}
				}
				//
				int target_count=0;
				double target_prob;
				if(j==-1){//Y or Y'
					for (k = 0; k < path_ptr->sws_len; k++) {
						prob*=path_ptr->sws[k]->inside;
						if(path_ptr->sws[k]->id==target_sw_id){
							target_count++;
							target_prob=path_ptr->sws[k]->inside;
						}
					}
					out_y(i)+=prob;
					if(target_count>0){
						prob/=target_prob;
						prob*=target_count;
						out_y_prime(i)+=prob;
					}
				}else{//M or M'
					for (k = 0; k < path_ptr->sws_len; k++) {
						prob*=path_ptr->sws[k]->inside;
						if(path_ptr->sws[k]->id==target_sw_id){
							target_count++;
							target_prob=path_ptr->sws[k]->inside;
						}
					}
					out_m(i,j)+=prob;
					if(target_count>0){
						prob/=target_prob;
						prob*=target_count;
						out_m_prime(i,j)+=prob;
					}
				}
				path_ptr = path_ptr->next;
			}
		}
	}
}



std::set<int> get_scc_children_node(int scc) {
	EG_NODE_PTR eg_ptr;
	std::set<int> result;
	int n=sccs[scc].size;
	for(int i=0; i<sccs[scc].size; i++) {
		int w=sccs[scc].el[i];
		EG_PATH_PTR path_ptr;
		eg_ptr = sorted_expl_graph[w];
		path_ptr = eg_ptr->path_ptr;
		if(path_ptr == NULL) {
			//error
		} else {
			while (path_ptr != NULL) {
				for (int k = 0; k < path_ptr->children_len; k++) {
					int index=mapping[path_ptr->children[k]->id];
					if(!is_scc_element(scc,index)){
						result.insert(index);
					}
				}
				path_ptr = path_ptr->next;
			}
		}
	}
	return result;
}


std::set<int> get_scc_containing_sws(int scc) {
	EG_NODE_PTR eg_ptr;
	std::set<int> result;
	int n=sccs[scc].size;
	for(int i=0; i<sccs[scc].size; i++) {
		int w=sccs[scc].el[i];
		EG_PATH_PTR path_ptr;
		eg_ptr = sorted_expl_graph[w];
		path_ptr = eg_ptr->path_ptr;
		if(path_ptr == NULL) {
			//error
		} else {
			while (path_ptr != NULL) {
				bool matrix_contain=false;
				for (int k = 0; k < path_ptr->children_len; k++) {
					int index=mapping[path_ptr->children[k]->id];
					if(is_scc_element(scc,index)){
						matrix_contain=true;
					}
				}
				if(matrix_contain){
					for (int k = 0; k < path_ptr->sws_len; k++) {
						result.insert(path_ptr->sws[k]->id);
					}
				}
				path_ptr = path_ptr->next;
			}
		}
	}
	return result;
}

std::set<int> get_scc_children_sws(int scc) {
	EG_NODE_PTR eg_ptr;
	std::set<int> result;
	int n=sccs[scc].size;
	for(int i=0; i<sccs[scc].size; i++) {
		int w=sccs[scc].el[i];
		EG_PATH_PTR path_ptr;
		eg_ptr = sorted_expl_graph[w];
		path_ptr = eg_ptr->path_ptr;
		if(path_ptr == NULL) {
			//error
		} else {
			while (path_ptr != NULL) {
				bool matrix_contain=false;
				for (int k = 0; k < path_ptr->children_len; k++) {
					int index=mapping[path_ptr->children[k]->id];
					if(is_scc_element(scc,index)){
						matrix_contain=true;
					}
				}
				if(!matrix_contain){
					for (int k = 0; k < path_ptr->sws_len; k++) {
						result.insert(path_ptr->sws[k]->id);
					}
				}
				path_ptr = path_ptr->next;
			}
		}
	}
	return result;
}

std::set<struct SwitchInstance *> get_scc_sws(int scc) {
	EG_NODE_PTR eg_ptr;
	std::set<struct SwitchInstance *> result;
	int n=sccs[scc].size;
	for(int i=0; i<sccs[scc].size; i++) {
		int w=sccs[scc].el[i];
		EG_PATH_PTR path_ptr;
		eg_ptr = sorted_expl_graph[w];
		path_ptr = eg_ptr->path_ptr;
		if(path_ptr == NULL) {
			//error
		} else {
			while (path_ptr != NULL) {
				for (int k = 0; k < path_ptr->sws_len; k++) {
					result.insert(path_ptr->sws[k]);
				}
				path_ptr = path_ptr->next;
			}
		}
	}
	return result;
}






void solve_linear_scc(int scc){
	int n=sccs[scc].size;
	MatrixXd A = MatrixXd(n,n);
	VectorXd b = VectorXd(n);
	VectorXd x;
	get_scc_matrix_inside(A,b,scc);
	int i, j;
	if(nl_debug_level>0){
		printf("# linear equation Ax=b\n");
		for(i = 0; i < n; i++) {
			for(j = 0; j < n; j++) {
				printf("%2.3f ",A(j,i));
			}
			printf("  %2.3f \n",b(i));
		}
	}
	FullPivLU< MatrixXd > lu(A);
	x=lu.solve(b);//Ax = b

	if(nl_debug_level>0){
		printf("# linear equation solve\n");
		for(i = 0; i < n; i++) {
			printf("  %2.3f \n",x(i));
		}
	}
	for(i=0; i<sccs[scc].size; i++) {
		EG_NODE_PTR eg_ptr;
		int w=sccs[scc].el[i];
		eg_ptr = sorted_expl_graph[w];
		eg_ptr->inside=x[i];
	}
}

void solve_linear_scc_outside_sw(int scc){
	int n=sccs[scc].size;
	MatrixXd M = MatrixXd(n,n);
	MatrixXd Mp = MatrixXd(n,n);
	VectorXd Y = VectorXd(n);
	VectorXd Yp = VectorXd(n);
	VectorXd x = VectorXd(n);
	// for computation
	VectorXd b = VectorXd(n);
	VectorXd xp;
	for(int i=0;i<sccs[scc].size;i++){
		int w=sccs[scc].el[i];
		x(i)=sorted_expl_graph[w]->inside;
	}
	std::set<struct SwitchInstance *> scc_sws=get_scc_sws(scc);
	for(std::set<struct SwitchInstance *>::iterator itr=scc_sws.begin(); itr!=scc_sws.end(); itr++) {
		get_scc_matrix_outside_sws(M,Mp,Y,Yp,scc,(*itr)->id);
		for(int i=0; i<n; i++) {
			M(i,i)-=1.0;
		}
		b=-(Mp*x+Yp);
		if(nl_debug_level>0){
			printf("# solve linear scc sw:%d\n",(*itr)->id);
			printf("# linear equation b=-(Mp*x+Yp)\n");
            for(int i = 0; i < n; i++) {
				for(int j = 0; j < n; j++) {
					printf("%2.3f ",Mp(i,j));
				}
				printf("  %2.3f ",x(i));
				printf("+  %2.3f \n",Yp(i));
			}

			printf("# linear equation Ax=b\n");
			for(int i = 0; i < n; i++) {
				for(int j = 0; j < n; j++) {
					printf("%2.3f ",M(i,j));
				}
				printf("  %2.3f \n",b(i));
			}
		}
		FullPivLU< MatrixXd > lu(M);
		xp=lu.solve(b);//Ax = b
		if(nl_debug_level>0){
			printf("# linear equation solve\n");
			for(int i = 0; i < n; i++) {
				printf("  %2.3f \n",xp(i));
			}
		}
		for(int i=0;i<n;i++){
			EG_NODE_PTR eg_ptr=sorted_expl_graph[sccs[scc].el[i]];
			double outside=eg_ptr->outside*xp[i];
			if(nl_debug_level>0){
				printf("outside(%3.3f)= outside x[%d](%3.3f)*Xp (%3.3f)\n",outside,sccs[scc].el[i],eg_ptr->outside,xp[i]);
				printf("%3.3f => %3.3f\n",(*itr)->total_expect,(*itr)->total_expect+outside*(*itr)->inside);
			}
			(*itr)->total_expect+=outside*(*itr)->inside;
		}
	}

}
void solve_linear_scc_outside_node(int scc){//,int target_node_index){
	int n=sccs[scc].size;
	MatrixXd A = MatrixXd(n,n);
	VectorXd b = VectorXd(n);
	VectorXd x;
	std::set<int> children=get_scc_children_node(scc);
	for(std::set<int>::iterator itr=children.begin(); itr!=children.end(); itr++) {
		EG_NODE_PTR target_eg_ptr = sorted_expl_graph[*itr];
		get_scc_matrix_outside(A,b,scc,*itr);
		if(nl_debug_level>0){
			printf("# linear equation Ax=b : outside(%d)\n",*itr);
			for(int i = 0; i < n; i++) {
				for(int j = 0; j < n; j++) {
					printf("%2.3f ",A(j,i));
				}
				printf("  %2.3f \n",b(i));
			}
		}
		FullPivLU< MatrixXd > lu(A);
		x=lu.solve(b);//Ax = b
		if(nl_debug_level>0){
			printf("# linear equation solve\n");
			for(int i = 0; i < n; i++) {
				printf("  %2.3f \n",x(i));
			}
			printf("# store out side\n");
		}
		for(int i=0;i<n;i++){
			EG_NODE_PTR eg_ptr=sorted_expl_graph[sccs[scc].el[i]];
			if(nl_debug_level>0){
				printf("outside x[%d](%3.3f)* X[%d] =>%3.3f\n",sccs[scc].el[i],eg_ptr->outside,i,eg_ptr->outside*x[i]);
			}
			target_eg_ptr->outside+=eg_ptr->outside*x[i];
		}
	}
}


extern "C"
int pc_linear_eq_2(void) {
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
	set_scc_order();
	if(nl_debug_level>=3) {
		print_sccs();
	}
	double scc_time=getCPUTime();
	//solution
	if(nl_debug_level>=2) {
		printf("linear solver: LU decomposition\n");
	}
	for(i=0; i<scc_num; i++) {
		if(sccs[i].size==1&&sccs[i].order==0){
			update_inside_scc(i);
		}else{
			solve_linear_scc(i);
		}
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
void compute_inside_linear_cycle(){
	if(nl_debug_level>=4) {
	printf("### compute inside\n");
	}
	for(int i=0; i<scc_num; i++) {
		if(sccs[i].size==1&&sccs[i].order==0){
			update_inside_scc(i);
		}else{
			solve_linear_scc(i);
		}
		if(nl_debug_level>=2) {
			int n=sccs[i].size;
			int j;
			for(j=0; j<n; j++) {
				int w=sccs[i].el[j];
				printf("%d:%f\n",w,sorted_expl_graph[w]->inside);
			}
		}
	}
	if(nl_debug_level>=4) {
	printf("### end inside\n");
	}
}
void compute_expectation_linear_cycle(){
	int i,k;
	EG_PATH_PTR path_ptr;
	EG_NODE_PTR eg_ptr,node_ptr;
	SW_INS_PTR sw_ptr;
	double q;

	for (i = 0; i < sw_ins_tab_size; i++) {
		switch_instances[i]->total_expect = 0.0;
	}

	for (i = 0; i < sorted_egraph_size; i++) {
		sorted_expl_graph[i]->outside = 0.0;
	}

	for (i = 0; i < num_roots; i++) {
		eg_ptr = expl_graph[roots[i]->id];
		if (i == failure_root_index) {
			eg_ptr->outside = num_goals / (1.0 - inside_failure);
		} else {
			eg_ptr->outside = roots[i]->count / eg_ptr->inside;
		}
	}
	for(i=scc_num-1; i>=0; i--) {
		int n=sccs[i].size;
		/*
		if(sccs[i].size==1&&sccs[i].order==0){
			if(nl_debug_level>=4) {
			printf("### compute expectation\n");
			}
			int w=sccs[i].el[0];
			eg_ptr = sorted_expl_graph[w];
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
						sw_ptr->total_expect += q;
					}
				}
				path_ptr = path_ptr->next;
			}
			if(nl_debug_level>=4) {
			printf("### end expectation\n");
			}
		}else{
		*/
		{
		//solve_linear_scc(i);
			if(nl_debug_level>=4) {
			printf("### compute expectation node (linear)\n");
			}
			//std::set<int> children=get_scc_children_node(i);
			//for(std::set<int>::iterator itr=children.begin();itr!=children.end();itr++){
			solve_linear_scc_outside_node(i);//(*itr));
			//}
			if(nl_debug_level>=4) {
			printf("### compute expectation sws (linear)\n");
			}
			solve_linear_scc_outside_sw(i);
			if(nl_debug_level>=4) {
			printf("### end expectation (linear)\n");
			}
		}
	}
	if(nl_debug_level>=4) {
	printf("### print (linear)\n");
	print_eq_outside();
	printf("### end print (linear)\n");
	}
}

int run_cyc_em() {

	int	 r, iterate, old_valid, converged, saved = 0;
	double  likelihood, log_prior;
	double  lambda, old_lambda = 0.0;

	//config_em(em_ptr);
	int i;
	EG_NODE_PTR eg_ptr;

	nodes=NULL;
	stack=NULL;
	scc_num=0;
	sccs=NULL;
	mapping=NULL;

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
	//print_eq
	if(nl_debug_level>=3) {
		print_eq();
	}
	//find scc
	get_scc_tarjan();
	set_scc_order();
	if(nl_debug_level>=3) {
		print_sccs();
	}
	double scc_time=getCPUTime();
	//start EM
	double itemp = 1.0;
	for (r = 0; r < num_restart; r++) {
		SHOW_PROGRESS_HEAD("#cyc-em-iters", r);
		//initialize_params();
		iterate = 0;
		while (1) {
			old_valid = 0;
			while (1) {
				if (CTRLC_PRESSED) {
					SHOW_PROGRESS_INTR();
					RET_ERR(err_ctrl_c_pressed);
				}

				
				//RET_ON_ERR(em_ptr->compute_inside());
				compute_inside_linear_cycle();
				//RET_ON_ERR(em_ptr->examine_inside());
				//examine_inside_linear_cycle();
				//likelihood = em_ptr->compute_likelihood();
				likelihood=compute_likelihood_scaling_none();
				//log_prior  = em_ptr->smooth ? em_ptr->compute_log_prior() : 0.0;
				lambda = likelihood + log_prior;
				{
				//if (verb_em) {
					//if (em_ptr->smooth) {
					//prism_printf("iteration #%d:\tlog_likelihood=%.9f\tlog_prior=%.9f\tlog_post=%.9f\n", iterate, likelihood, log_prior, lambda);
					//}
					//else {
					prism_printf("iteration #%d:\tlog_likelihood=%.9f\n", iterate, likelihood);
					if(nl_debug_level>=4) {
					print_eq();
					}
					//}
				}

				if (!std::isfinite(lambda)) {
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
				compute_expectation_linear_cycle();

				SHOW_PROGRESS(iterate);
				//RET_ON_ERR(em_ptr->update_params());
				update_params();
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

		saved = (r < num_restart - 1);
		if (saved) {
			save_params();
		}

	}
	if (saved) {
		restore_params();
	}
	//END EM

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


	//em_ptr->bic = compute_bic(em_ptr->likelihood);
	//em_ptr->cs  = em_ptr->smooth ? compute_cs(em_ptr->likelihood) : 0.0;
	return BP_TRUE;
}


extern "C"
int pc_cyc_em_6(void) {
	printf("--- start cyc_em\n");
	run_cyc_em();
	printf("--- end cyc_em\n");
	return
		bpx_unify(bpx_get_call_arg(1,6), bpx_build_integer(1)) &&
		bpx_unify(bpx_get_call_arg(2,6), bpx_build_float  (0)) &&
		bpx_unify(bpx_get_call_arg(3,6), bpx_build_float  (0)) &&
		bpx_unify(bpx_get_call_arg(4,6), bpx_build_float  (0)) &&
		bpx_unify(bpx_get_call_arg(5,6), bpx_build_float  (0)) &&
		bpx_unify(bpx_get_call_arg(6,6), bpx_build_integer(0)) ;
}



