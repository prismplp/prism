#include "up/up.h"
#include "up/flags.h"
#include "bprolog.h"
#include "core/random.h"
#include "core/gamma.h"
#include "up/graph.h"
#include "up/util.h"

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <sys/resource.h>

double getCPUTime()
{
	struct rusage RU;
	getrusage(RUSAGE_SELF, &RU);
	return RU.ru_utime.tv_sec + (double)RU.ru_utime.tv_usec*1e-6;
}
static int* mapping;
// need to declare this function before
double bpx_get_float(TERM t);
typedef struct{
	int visited;
	int index;
	int lowlink;
	int in_stack;
} SCC_Node;
typedef struct SCC_StackEl{
	int index;
	struct SCC_StackEl* next;
} SCC_Stack;

typedef struct{
	int* el;
	int size;
	int order;
} SCC;

static SCC_Node* nodes;
static SCC_Stack* stack;
static int scc_num;
static SCC* sccs;

void compute_scc_eq(double* x,double* o,int scc){
	int i;
	EG_NODE_PTR eg_ptr;
	for(i=0;i<sccs[scc].size;i++){
		int w=sccs[scc].el[i];
		eg_ptr = sorted_expl_graph[w];
		eg_ptr->inside=x[i];
	}
	for(i=0;i<sccs[scc].size;i++){
		int w=sccs[scc].el[i];
		EG_PATH_PTR path_ptr;
		eg_ptr = sorted_expl_graph[w];
		path_ptr = eg_ptr->path_ptr;
		double sum=-eg_ptr->inside;
		if(path_ptr == NULL) {
			sum+=1;
		}else{
			while (path_ptr != NULL) {
				int k;
				double prob=1;
				for (k = 0; k < path_ptr->children_len; k++) {
					//int w;
					//w=mapping[path_ptr->children[k]->id];
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
void progress(double* x,double* sk,int dim){
	int i;
	printf("x=>> ");
	for(i=0;i<dim;i++){
		printf("%f,",x[i]);
	}
	printf("\ns=>> ");
	for(i=0;i<dim;i++){
		printf("%f,",sk[i]);
	}
	printf("\n");
}
void solve(int scc){
	int dim =sccs[scc].size;
	int loopCount;
	int maxloop=100;
	int restart=20;
	double** s=calloc(sizeof(double*),restart+1);
	double epsilon=1.0e-10;
	double epsilonX=1.0e-10;
	int i;
	for(i=0;i<restart+1;i++){
		s[i]=calloc(sizeof(double),dim);
	}
	double* z=calloc(sizeof(double),dim);
	double* x=calloc(sizeof(double),dim);
	double* s0=s[0];
	for(i=0;i<dim;i++){
		x[i]=rand()/(double)RAND_MAX;
	}
	compute_scc_eq(x,s0,scc);
	int s_cnt=0;
	int convergenceCount;
	for(loopCount=0;maxloop<0||loopCount<maxloop;loopCount++){
		if(restart>0 && ((loopCount+1)%restart==0)){
			//clear working area
			s_cnt=0;
			compute_scc_eq(x,s0,scc);
		}
		double* sk=s[s_cnt];
		convergenceCount=0;
		progress(x,sk,dim);
		for(i=0;i<dim;i++){
			x[i]+=sk[i];
			if((fabs(x[i])<epsilonX&&fabs(sk[i])<epsilon)||fabs(sk[i]/x[i])<epsilon){
				convergenceCount++;
			}
		}
		if(convergenceCount==dim){
			break;
		}
		compute_scc_eq(x,z,scc);
		int j;
		for(j=1;j<=s_cnt;j++){
			double denom=0;
			double num=0;
			for(i=0;i<dim;i++){
				num+=z[i]*s[j-1][i];
				denom+=s[j-1][i]*s[j-1][i];
			}
			for(i=0;i<dim;i++){
				if(denom!=0){
					z[i]+=s[j][i]*num/denom;
				}
			}
		}
		//
		double c=0;
		{
			double denom=0;
			double num=0;
			for(i=0;i<dim;i++){
				num+=z[i]*sk[i];
				denom+=sk[i]*sk[i];
			}
			if(denom!=0){
				c=num/denom;
			}else{
				c=0;
				//status.error|=E_SS_UNDERFLOW;
				printf("error 1!!\n");
			}
		}
		double* sk1=s[s_cnt+1];
		for(i=0;i<dim;i++){
			if(1.0-c!=0){
				sk1[i]=z[i]/(1.0-c);
			}else{
				//status.error|=E_C_IS_ONE;
				printf("error 2!!\n");
			}
		}
		s_cnt++;
	}
	for(i=0;i<dim;i++){
		int w=sccs[scc].el[i];
		sorted_expl_graph[w]->inside=x[i];
	}
	for(i=0;i<restart+1;i++){
		free(s[i]);
	}
	free(s);
	free(z);
	free(x);
	printf("%d/%d\n",convergenceCount,dim);
}


void get_scc_tarjan_start(int i,int index){
	nodes[i].visited=1;
	nodes[i].index=index;
	nodes[i].lowlink=index;
	nodes[i].in_stack=1;
	index++;
	SCC_Stack* el=malloc(sizeof(SCC_Stack));
	if(stack==NULL){
		el->next=NULL;
		stack=el;
	}else{
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
			if(nodes[w].visited==0){
				get_scc_tarjan_start(w,index);
				int il=nodes[i].lowlink;
				int wl=nodes[w].lowlink;
				nodes[i].lowlink=il<wl?il:wl;
			}else if(nodes[w].in_stack){//visited==2
				int il=nodes[i].lowlink;
				int wi=nodes[w].index;
				nodes[i].lowlink=il<wi?il:wi;
			}
		}
		path_ptr = path_ptr->next;
	}
	if(nodes[i].lowlink==nodes[i].index){
		int w;
		SCC_Stack* itr=stack;
		printf("%d:",i);
		int cnt=0;
		do{
			w=itr->index;
			itr=itr->next;
			cnt++;
		}while(w!=i);
		sccs[scc_num].size=cnt;
		sccs[scc_num].el=calloc(sizeof(int),cnt);
		printf("(%d)",cnt);
		cnt=0;
		do{
			w=stack->index;
			printf("%d,",w);
			nodes[w].visited=2;
			nodes[w].in_stack=0;
			sccs[scc_num].el[cnt]=w;
			SCC_Stack* temp=stack;
			stack=stack->next;
			free(temp);
			cnt++;
		}while(w!=i);
		printf("\n");
		scc_num++;
	}
}
int is_reachable(int i,int j){
	EG_NODE_PTR eg_ptr;
	EG_PATH_PTR path_ptr;
	if(nodes[i].visited==1){
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
			if(w==j || is_reachable(w,j)){
				return 1;
			}
		}
		path_ptr=path_ptr->next;
	}
	return 0;
}
int check_scc(int scc_id){
	int n=sccs[scc_id].size;
	int i,j,k;
	for(i=0;i<n;i++){
		for(j=0;j<n;j++){
			for(k=0;k<sorted_egraph_size;k++){
				nodes[k].visited=0;
			}
			if(i!=j){
				if(!is_reachable(sccs[scc_id].el[i],sccs[scc_id].el[j])){
					printf("error!!%d:%d(%d)->%d(%d)\n",scc_id,sccs[scc_id].el[i],i,sccs[scc_id].el[j],j);
					return 0;
				}
			}
		}
	}
	return 1;
}
int check_scc_order(){
	int err=0;
	EG_NODE_PTR eg_ptr;
	EG_PATH_PTR path_ptr;
	int l;
	for(l=0;l<sorted_egraph_size;l++){
		nodes[l].visited=0;
	}
	int i;
	for(i=0;i<scc_num;i++){
		int n=sccs[i].size;
		int j;
		for(j=0;j<n;j++){
			int index=sccs[i].el[j];
			nodes[index].visited=1;
		}
		for(j=0;j<n;j++){
			int index=sccs[i].el[j];
			eg_ptr = sorted_expl_graph[index];
			path_ptr = eg_ptr->path_ptr;
			while (path_ptr != NULL) {
				int k;
				for (k = 0; k < path_ptr->children_len; k++) {
					int w;
					w=mapping[path_ptr->children[k]->id];
					if(nodes[w].visited==0){
						printf("scc order error!!\n");
						err++;
					}
				}
				path_ptr=path_ptr->next;
			}
		}
	}
	return err;
}
void reset_visit(){
	int l;
	for(l=0;l<sorted_egraph_size;l++){
		nodes[l].visited=0;
	}
}
void print_sccs(){
	EG_NODE_PTR eg_ptr;
	EG_PATH_PTR path_ptr;
	int i;
	reset_visit();
	for(i=0;i<scc_num;i++){
		int n=sccs[i].size;
		int j;
		for(j=0;j<n;j++){
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
					if(nodes[w].visited==0){
						if(enable){
							printf(",%d",w);
						}else{
							printf("(%d",w);
							enable=1;
						}
					}
				}
				if(enable){
					printf(")");
				}
				path_ptr=path_ptr->next;
			}
			printf("\n");
		}
		for(j=0;j<n;j++){
			int index=sccs[i].el[j];
			nodes[index].visited=1;
		}
	}
	return;
}

void get_scc_tarjan(){
	int i;
	nodes=calloc(sizeof(SCC_Node),sorted_egraph_size);
	sccs=calloc(sizeof(SCC),sorted_egraph_size);
	get_scc_tarjan_start(0,0);
	while(1){
		int next_i=0;
		int cnt=0;
		for(i=0;i<sorted_egraph_size;i++){
			if(nodes[i].visited!=2){
				nodes[i].visited=0;
				nodes[i].in_stack=0;
				if(next_i==0){
					next_i=i;
				}
			}else{
				cnt++;
			}
		}
		if(cnt!=sorted_egraph_size){
			get_scc_tarjan_start(next_i,cnt);
		}else{
			break;
		}
	}
	//check
	if(stack!=NULL){
		printf("stack remain\n");
	}
	for(i=0;i<scc_num;i++){
		check_scc(i);
	}
	check_scc_order();
	
}

int pc_infix_2(void) {
	int i,k;
	EG_NODE_PTR eg_ptr;
	EG_PATH_PTR path_ptr;
	
nodes=NULL;
stack=NULL;
scc_num=0;
sccs=NULL;
mapping=NULL;
	
	int max_id=0;
    int debug_level = bpx_get_integer(bpx_get_call_arg(1,2));

	for (i = 0; i < sorted_egraph_size; i++) {
		eg_ptr = sorted_expl_graph[i];
		if(max_id<eg_ptr->id){
			max_id=eg_ptr->id;
		}
	}
	mapping=calloc(sizeof(int),max_id+1);
	for (i = 0; i < sorted_egraph_size; i++) {
		eg_ptr = sorted_expl_graph[i];
		mapping[eg_ptr->id]=i;
	}
	
	//print_eq
	printf("graph_size:%d\n",sorted_egraph_size);
	printf("equations\n");
	for (i = 0; i < sorted_egraph_size; i++) {
		eg_ptr = sorted_expl_graph[i];
		path_ptr = eg_ptr->path_ptr;
		printf("# x[%d] : ",i);
		while (path_ptr != NULL) {
			for (k = 0; k < path_ptr->children_len; k++) {
				//path_ptr->children[k]
				printf("x[%d]*",mapping[path_ptr->children[k]->id]);
			}
			for (k = 0; k < path_ptr->sws_len; k++) {
				printf("%3.3f*",path_ptr->sws[k]->inside);
			}
			printf("+");
			path_ptr = path_ptr->next;
		}
		printf("\n");
	}
	//find scc
	get_scc_tarjan();
	
	printf("===scc\n");
	print_sccs();
	//
	printf("===solve\n");
	for(i=0;i<scc_num;i++){
		solve(i);
		int n=sccs[i].size;
		int j;
		for(j=0;j<n;j++){
			int w=sccs[i].el[j];
			printf("%d:%f\n",w,sorted_expl_graph[w]->inside);
		}
	}
	//free data
	free(nodes);
	for(i=0;i<scc_num;i++){
		free(sccs[i].el);
	}
	free(sccs);
	free(mapping);
	double prob=sorted_expl_graph[sorted_egraph_size-1]->inside;
	
    return bpx_unify(bpx_get_call_arg(2,2),
                     bpx_build_float(prob));
}

