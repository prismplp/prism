#define CXX_COMPILE 

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
#include "up/scc.h"
#include "up/rank.h"
#include "up/tensor_preds.h"
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
std::map<string,string> exported_flags;
std::map<string,int> exported_index_range;


void save_message(const string& outfilename,::google::protobuf::Message* msg,SaveFormat format) {
	switch(format){
		case FormatJson:
		{
			fstream output(outfilename.c_str(), ios::out | ios::trunc);
			string s;
			util::MessageToJsonString(*msg,&s);
			output<<s<<endl;
			cout<<"[SAVE:json] "<<outfilename<<endl;
			break;
		}
		case FormatPbTxt:
		{
			fstream output(outfilename.c_str(), ios::out | ios::trunc);
			io::OstreamOutputStream* oss = new io::OstreamOutputStream(&output); 
			if (!TextFormat::Print(*msg, oss)) {     
				cerr << "Failed to write explanation graph." << endl;  
			}
			delete oss;
			cout<<"[SAVE:PBTxt]"<<outfilename<<endl;
			break;
		}
		case FormatPb:
		{
			fstream output(outfilename.c_str(), ios::out | ios::trunc | ios::binary);
			if (!msg->SerializeToOstream(&output)) {
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

void export_flags(TERM el2){
	if(bpx_is_list(el2)){
		while(!bpx_is_nil(el2)){
			TERM el = bpx_get_car(el2);
			
			TERM key = bpx_get_car(el);
			TERM val_el = bpx_get_cdr(el);
			TERM val = bpx_get_car(val_el);
			string key_str= bpx_term_2_string(key);
			string val_str= bpx_term_2_string(val);
			exported_flags[key_str]=val_str;
			
			el2 = bpx_get_cdr(el2);
		}
	}
	return;
}

void construct_placeholder_goal(prism::PlaceholderGoal* g, TERM ph_term, TERM data_term) {
	if(bpx_is_list(ph_term)){
		while(!bpx_is_nil(ph_term)){
			TERM ph_el= bpx_get_car(ph_term);
			const char* name=bpx_get_name(ph_el);
			prism::Placeholder* ph=g->add_placeholders();
			ph->set_name(name);
			ph_term = bpx_get_cdr(ph_term);
		}
	}
	//
	if(!bpx_is_list(data_term)){
		printf("[ERROR] A data term is not a list");
	}
	while(!bpx_is_nil(data_term)){
		TERM data_sample_term= bpx_get_car(data_term);
		prism::DataRecord* r=g->add_records();
		
		if(!bpx_is_list(data_sample_term)){
			char* s =bpx_term_2_string(data_sample_term);
			printf("[ERROR] A sample term is not a list: %s\n",s);
		}
		while(!bpx_is_nil(data_sample_term)){
			TERM data_el_term= bpx_get_car(data_sample_term);
			char* name =bpx_term_2_string(data_el_term);
			//const char* name=bpx_get_name(data_el_term);
			r->add_items(name);
			data_sample_term = bpx_get_cdr(data_sample_term);
		}
		data_term = bpx_get_cdr(data_term);
	}
}
extern "C"
int pc_save_placeholder_data_4(void) {
	TERM ph  =bpx_get_call_arg(3,4);
	TERM data=bpx_get_call_arg(4,4);
	
	prism::PlaceholderData ph_data;
	if(bpx_is_list(ph) && bpx_is_list(data)){
		while(!bpx_is_nil(ph) && !bpx_is_nil(data)){
			TERM ph_el   = bpx_get_car(ph);
			TERM data_el = bpx_get_car(data);
			prism::PlaceholderGoal* g=ph_data.add_goals();
			construct_placeholder_goal(g, ph_el, data_el);
			
			ph   = bpx_get_cdr(ph);
			data = bpx_get_cdr(data);
		}
	}
	const char* filename=bpx_get_name(bpx_get_call_arg(1,4));
	SaveFormat format = (SaveFormat) bpx_get_integer(bpx_get_call_arg(2,4));
	save_message(filename,&ph_data,format);
	return BP_TRUE;
}
extern "C"
int pc_set_export_flags_1(void) {
	TERM t=bpx_get_call_arg(1,1);
	export_flags(t);
	return BP_TRUE;
}

extern "C"
int pc_save_options_2(void) {
	//char* filename =bpx_term_2_string(bpx_get_call_arg(1,2));
	const char* filename=bpx_get_name(bpx_get_call_arg(1,2));
	SaveFormat format = (SaveFormat) bpx_get_integer(bpx_get_call_arg(2,2));
	prism::Option op;
	// set flags
	for(auto itr:exported_flags){
		prism::Flag* f=op.add_flags();
		f->set_key(itr.first);
		f->set_value(itr.second);
	}
	//set index_range
	for(auto itr:exported_index_range){
		prism::IndexRange* ir=op.add_index_range();
		ir->set_index(itr.first);
		ir->set_range(itr.second);
	}
	// save
	save_message(filename,&op,format);
	return BP_TRUE;
}

extern "C"
int pc_set_index_range_2(void) {
	char* index_name= bpx_term_2_string(bpx_get_call_arg(1,2));
	int range=bpx_get_integer(bpx_get_call_arg(2,2));
	string s=index_name;
	exported_index_range[s]=range;
	return BP_TRUE;
}


