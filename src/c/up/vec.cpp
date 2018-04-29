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

#include <iostream>
#include <set>
#include <cmath>
#include <string>
#include "external/expl.pb.h"
#include <google/protobuf/text_format.h>
#include <google/protobuf/message.h>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/io/zero_copy_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/util/json_util.h>

#include <fstream>
using namespace std;
using namespace google::protobuf;

prism::PredTerm get_pred(TERM term){
	prism::PredTerm pred;
	const char* name=bpx_get_name(term);
	pred.set_name(name);
	int arity=bpx_get_arity(term);
	for(BPLONG j=1; j<=arity; j++){
		TERM el= bpx_get_arg(j, term);
		char*arg= bpx_term_2_string(el);
		pred.add_args(arg);
	}
	return pred;
}
prism::ExplGraphNode get_node(int id){
	prism::ExplGraphNode node;
	TERM term=prism_goal_term(id);
	prism::PredTerm* pred=node.mutable_goal();
	*pred=get_pred(term);
	node.set_id(id);
	return node;
}

prism::SwIns get_swins(int id){
	prism::SwIns sw;
	prism::PredTerm* name_pred=sw.mutable_name();
	prism::PredTerm* value_pred=sw.mutable_value();
	TERM term=prism_sw_ins_term(id);
	const char* s=bpx_get_name(term);
	int arity=bpx_get_arity(term);
	if(arity==2){
		TERM el1= bpx_get_arg(1, term);
		TERM el2= bpx_get_arg(2, term);
		*name_pred=get_pred(el1);
		*value_pred=get_pred(el2);
	}
	sw.set_id(id);
	return sw;
}

enum SaveFormat{
	FormatJson=0,
	FormatPb=1,
	FormatPbTxt=2,
};

void save_expl(const string& outfilename,prism::ExplGraph& goals,SaveFormat format) {
	switch(format){
		case FormatJson:
		{
			fstream output(outfilename.c_str(), ios::out | ios::trunc);
			string s;
			util::MessageToJsonString(goals,&s);
			output<<s<<endl;
			cout<<"[SAVE:json] "<<outfilename<<endl;
			break;
		}
		case FormatPbTxt:
		{
			fstream output(outfilename.c_str(), ios::out | ios::trunc);
			io::OstreamOutputStream* oss = new io::OstreamOutputStream(&output); 
			if (!TextFormat::Print(goals, oss)) {     
				cerr << "Failed to write explanation graph." << endl;  
			}
			delete oss;
			cout<<"[SAVE:PBTxt]"<<outfilename<<endl;
			break;
		}
		case FormatPb:
		{
			fstream output(outfilename.c_str(), ios::out | ios::trunc | ios::binary);
			if (!goals.SerializeToOstream(&output)) {
				cerr << "Failed to write explanation graph." << endl;  
			}
			cout<<"[SAVE:PB]"<<outfilename<<endl;
			break;
		}
		default:
			cerr << "Unknown format." << endl;  
			break;
			
	}
	
}
int run_vec(const string& outfilename,SaveFormat format) {
	//config_em(em_ptr);
	double start_time=getCPUTime();
	init_scc();
	double scc_time=getCPUTime();
	initialize_params();
	print_eq();
	save_params();
	print_sccs_statistics();
	double solution_time=getCPUTime();
	
	EG_NODE_PTR eg_ptr;
	EG_PATH_PTR path_ptr;
	prism::ExplGraph goals;
	for (int i = 0; i < sorted_egraph_size; i++) {
		eg_ptr = sorted_expl_graph[i];
		int id= eg_ptr->id;
		prism::ExplGraphGoal* goal=goals.add_goals();
		prism::ExplGraphNode* node=goal->mutable_node();
		*node=get_node(id);
		path_ptr = eg_ptr->path_ptr;
		while (path_ptr != NULL) {
			prism::ExplGraphPath* path=goal->add_paths();
			for (int k = 0; k < path_ptr->children_len; k++) {
				int id= path_ptr->children[k]->id;
				prism::ExplGraphNode* node= path->add_nodes();
				*node=get_node(id);
			}
			for (int k = 0; k < path_ptr->sws_len; k++) {
				int id= path_ptr->sws[k]->id;
				prism::SwIns* sw= path->add_sws();
				*sw=get_swins(id);
			}
			path_ptr = path_ptr->next;
		}
	}
	//
	for (int i = 0; i < num_roots; i++) {
		prism::Root* r=goals.add_roots();
		eg_ptr = expl_graph[roots[i]->id];
		r->set_id(eg_ptr->id);
		r->set_count(roots[i]->count);
	}
	save_expl(outfilename,goals,format);

//free data
	free_scc();
	if(scc_debug_level>=1) {
		printf("CPU time (scc,solution,all)\n");
		printf("# %f,%f,%f\n",scc_time-start_time,solution_time-scc_time,solution_time - start_time);
	}
	
	
	return BP_TRUE;
}

extern "C"
int pc_prism_vec_1(void) {
	//struct EM_Engine em_eng;
	//RET_ON_ERR(check_smooth(&em_eng.smooth));
	//scc_debug_level = bpx_get_integer(bpx_get_call_arg(7,7));
	run_vec("expl.bin",FormatPb);
	return bpx_unify(bpx_get_call_arg(1,1), bpx_build_integer(1));
}

