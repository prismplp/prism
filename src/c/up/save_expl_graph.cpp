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
#include "up/rank.h"
}

#include <iostream>
#include <set>
#include <cmath>
#include <string>
#include "external/expl.pb.h"
#include "up/save_expl_graph.h"
#include <google/protobuf/text_format.h>
#include <google/protobuf/message.h>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/io/zero_copy_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/util/json_util.h>

#include <fstream>
using namespace std;
using namespace google::protobuf;

prism::GoalTerm get_goal(TERM term){
	prism::GoalTerm goal;
	const char* name=bpx_get_name(term);
	goal.set_name(name);
	int arity=bpx_get_arity(term);
	for(BPLONG j=1; j<=arity; j++){
		TERM el= bpx_get_arg(j, term);
		char*arg= bpx_term_2_string(el);
		goal.add_args(arg);
	}
	return goal;
}
prism::ExplGraphNode get_node(int id,int sorted_id){
	prism::ExplGraphNode node;
	TERM term=prism_goal_term(id);
	prism::GoalTerm* pred=node.mutable_goal();
	*pred=get_goal(term);
	node.set_id(id);
	node.set_sorted_id(sorted_id);
	return node;
}

void set_value_list(prism::SwIns* sw,TERM term){
	if(bpx_is_list(term)){
		while(!bpx_is_nil(term)){
			TERM el = bpx_get_car(term);
			char* el_str= bpx_term_2_string(el);
			sw->add_values(el_str);
			term = bpx_get_cdr(term);
		}
	}else{
		char* term_str= bpx_term_2_string(term);
		sw->add_values(term_str);
	}
}
prism::SwIns get_swins(SW_INS_PTR sw_ins){
	prism::SwIns sw;
	int id= sw_ins->id;
	TERM term=prism_sw_ins_term(id);
	// msw
	//const char* s=bpx_get_name(term);
	BPLONG arity=bpx_get_arity(term);
	if(arity>=2){
		TERM el1= bpx_get_arg(1, term);
		TERM el2= bpx_get_arg(2, term);
		char* el1_name= bpx_term_2_string(el1);
		string name=bpx_get_name(el1);
		if(name=="tensor"){
			sw.set_name(el1_name);
			sw.set_sw_type(prism::Tensor);
			sw.set_inside(0.0);
			set_value_list(&sw,el2);
		}else if(name=="$operator"){
			sw.set_name(name);
			TERM op_term= bpx_get_arg(1, el1);
			BPLONG op_arity=bpx_get_arity(op_term);
			for(BPLONG j=1; j<=op_arity; j++){
				TERM el= bpx_get_arg(j, op_term);
				char* el_str= bpx_term_2_string(el);
				sw.add_values(el_str);
			}
			string op_name=bpx_get_name(op_term);
			sw.set_name(op_name);
			sw.set_sw_type(prism::Operator);
			sw.set_inside(0.0);
		}else{
			sw.set_name(el1_name);
			sw.set_sw_type(prism::Probabilistic);
			sw.set_inside(sw_ins->inside);
			set_value_list(&sw,el2);
		}
		
	}
	sw.set_id(id);
	return sw;
}

std::vector<int> build_mapping_goal_id_to_sorted_id(){
	std::vector<int> mapping(egraph_size); 
	for (int i = 0; i < sorted_egraph_size; i++) {
		EG_NODE_PTR eg_ptr = sorted_expl_graph[i];
		int id= eg_ptr->id;
		mapping.at(id)=i;
	}
	return mapping;
}

void save_expl_graph(const string& outfilename,prism::ExplGraph& goals,SaveFormat format) {
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

int run_save_expl_graph(const string& outfilename,SaveFormat format) {
	//config_em(em_ptr);
	double start_time=getCPUTime();
	init_scc();
	double scc_time=getCPUTime();
	initialize_params();
	//print_eq();
	save_params();
	print_sccs_statistics();
	double solution_time=getCPUTime();
	std::vector<int> mapping_to_sorted_id=build_mapping_goal_id_to_sorted_id();
	EG_NODE_PTR eg_ptr;
	EG_PATH_PTR path_ptr;
	prism::ExplGraph goals;
	for (int i = 0; i < sorted_egraph_size; i++) {
		eg_ptr = sorted_expl_graph[i];
		int id= eg_ptr->id;
		prism::ExplGraphGoal* goal=goals.add_goals();
		prism::ExplGraphNode* node=goal->mutable_node();
		int sorted_id=mapping_to_sorted_id.at(id);
		*node=get_node(id,sorted_id);
		path_ptr = eg_ptr->path_ptr;
		while (path_ptr != NULL) {
			prism::ExplGraphPath* path=goal->add_paths();
			for (int k = 0; k < path_ptr->children_len; k++) {
				int id= path_ptr->children[k]->id;
				int sorted_id=mapping_to_sorted_id.at(id);
				prism::ExplGraphNode* node= path->add_nodes();
				*node=get_node(id,sorted_id);
			}
			for (int k = 0; k < path_ptr->sws_len; k++) {
				prism::SwIns temp_sw=get_swins(path_ptr->sws[k]);
				prism::SwIns* sw;
				if(temp_sw.sw_type()==prism::Probabilistic){
					sw= path->add_prob_switches();
				}else if(temp_sw.sw_type()==prism::Operator){
					sw= path->add_operators();
				}else{
					sw= path->add_tensor_switches();
				}
				*sw=temp_sw;
			}
			path_ptr = path_ptr->next;
		}
	}
	//
	/*
	for (int i = 0; i < num_roots; i++) {
		prism::Root* r=goals.add_roots();
		eg_ptr = expl_graph[roots[i]->id];
		r->set_id(eg_ptr->id);
		int sorted_id=mapping_to_sorted_id.at(id);
		r->set_count(roots[i]->count);
	}
	*/
	for (RNK_NODE_PTR itr = rank_root; itr != NULL; itr=itr->next) {
		//int index = i%num_minibatch;
		prism::RankRoot* rr=goals.add_root_list();
		for(int j=0;j<itr->goal_count;j++){
			prism::Root* r=rr->add_roots();
			int id = itr->goals[j];
			int sorted_id=mapping_to_sorted_id.at(id);
			r->set_id(id);
			r->set_sorted_id(sorted_id);
		}
		rr->set_count(1);
	}
	save_expl_graph(outfilename,goals,format);

	//free data
	free_scc();
	if(scc_debug_level>=1) {
		printf("CPU time (scc,solution,all)\n");
		printf("# %f,%f,%f\n",scc_time-start_time,solution_time-scc_time,solution_time - start_time);
	}
	
	return BP_TRUE;
}

extern "C"
int pc_prism_save_expl_graph_2(void) {
	const char* filename=bpx_get_name(bpx_get_call_arg(1,2));
	SaveFormat format = (SaveFormat) bpx_get_integer(bpx_get_call_arg(2,2));
	run_save_expl_graph(filename,format);
	//return bpx_unify(bpx_get_call_arg(1,1), bpx_build_integer(1));
	return BP_TRUE;
}

