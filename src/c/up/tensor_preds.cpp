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
#include <fstream>
#include "external/expl.pb.h"
#include "up/save_expl_graph.h"
#include <google/protobuf/text_format.h>
#include <google/protobuf/message.h>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/io/zero_copy_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/util/json_util.h>

#include <H5Cpp.h>

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

TERM construct_placeholder_goal(prism::PlaceholderGoal* g, TERM ph_term, TERM data_term,int split_num) {
	// adding placeholder list
	if(bpx_is_list(ph_term)){
		while(!bpx_is_nil(ph_term)){
			TERM ph_el= bpx_get_car(ph_term);
			const char* name=bpx_get_name(ph_el);
			prism::Placeholder* ph=g->add_placeholders();
			ph->set_name(name);
			ph_term = bpx_get_cdr(ph_term);
		}
	}
	// adding records
	if(!bpx_is_list(data_term)){
		printf("[ERROR] A data term is not a list");
	}
	for(int i=0; i<split_num && !bpx_is_nil(data_term); i++){
		TERM data_sample_term= bpx_get_car(data_term);
		prism::DataRecord* r=g->add_records();
		
		if(!bpx_is_list(data_sample_term)){
			char* s =bpx_term_2_string(data_sample_term);
			printf("[ERROR] A sample term is not a list: %s\n",s);
		}
		while(!bpx_is_nil(data_sample_term)){
			TERM data_el_term= bpx_get_car(data_sample_term);
			char* name =bpx_term_2_string(data_el_term);
			r->add_items(name);
			data_sample_term = bpx_get_cdr(data_sample_term);
		}
		data_term = bpx_get_cdr(data_term);
	}
	return data_term;
}

void save_placeholder_data_hdf5(TERM ph, TERM data, string filename,SaveFormat format) {
	int goal_counter=0;
	if(!bpx_is_list(ph) || !bpx_is_list(data)){
	}
	while(!bpx_is_nil(ph) && !bpx_is_nil(data)){
		string group_name=std::to_string(goal_counter);
		vector<string> placeholders;
		TERM ph_term   = bpx_get_car(ph);
		TERM data_term = bpx_get_car(data);
		//
		if(!bpx_is_list(ph_term)){
		}
		while(!bpx_is_nil(ph_term)){
			TERM ph_el= bpx_get_car(ph_term);
			const char* name=bpx_get_name(ph_el);
			placeholders.push_back(name);
			ph_term = bpx_get_cdr(ph_term);
		}
		int n=placeholders.size();
		// compute length
		int length=0;
		TERM temp_data_term=data_term;
		while(!bpx_is_nil(temp_data_term)){
			temp_data_term = bpx_get_cdr(temp_data_term);
			length++;
		}
		
		H5::CompType mtype(sizeof(int)*n);
		for(int j=0;j<n;j++){
			mtype.insertMember(placeholders[j], j*sizeof(int), H5::PredType::NATIVE_INT);
		}
		//
		{
			int* data_table=new int[length*placeholders.size()];
			for(int i=0;!bpx_is_nil(data_term);i++){
				TERM data_sample_term= bpx_get_car(data_term);
				if(!bpx_is_list(data_sample_term)){
					char* s =bpx_term_2_string(data_sample_term);
					printf("[ERROR] A sample term is not a list: %s\n",s);
				}
				for(int j=0;!bpx_is_nil(data_sample_term);j++){
					TERM data_el_term= bpx_get_car(data_sample_term);
					char* name =bpx_term_2_string(data_el_term);
					int v=stoi(name);
					data_table[i*n+j]=v;
					data_sample_term = bpx_get_cdr(data_sample_term);
				}
				data_term = bpx_get_cdr(data_term);
			}
			// save dataset
			H5::H5File file( filename, H5F_ACC_TRUNC );
			{
				hsize_t     dimsf[1];              // dataset dimensions
				dimsf[0] = length;
				dimsf[1] = placeholders.size();
				H5::DataSpace dataspace(2, dimsf );
				H5::IntType datatype( H5::PredType::NATIVE_INT );
				datatype.setOrder( H5T_ORDER_LE );
				file.createGroup( group_name);
				H5::DataSet dataset = file.createDataSet( group_name+"/data", datatype, dataspace );
				dataset.write( data_table, H5::PredType::NATIVE_INT );
			
				{
					hsize_t  dimsf[1] = {n};
					H5::DataSpace dataspace(1, dimsf );

					H5::StrType str_type(H5::PredType::C_S1, H5T_VARIABLE);
					const char * cStrArray[n];
					for(int index = 0; index < n; ++index){
						cStrArray[index]=placeholders[index].c_str();
					}
					H5::Attribute attr = dataset.createAttribute("placeholders", str_type, dataspace);
					attr.write(str_type,(void*)&cStrArray[0]);
				}
			}
		}
		// next
		ph   = bpx_get_cdr(ph);
		data = bpx_get_cdr(data);
		goal_counter++;
	}
	//save_message(filename+std::to_string(file_count),&ph_data,format);
	
}
void save_placeholder_data(TERM ph, TERM data, string filename,SaveFormat format,int split_num) {
	int goal_counter=0;
	TERM data_el=bpx_build_nil();
	if(!bpx_is_list(ph) || !bpx_is_list(data)){
	}
	// for split save
	for(int file_count=0;!bpx_is_nil(ph) && !bpx_is_nil(data);file_count++){
		prism::PlaceholderData ph_data;
		while(!bpx_is_nil(ph) && !bpx_is_nil(data)){
			TERM ph_el   = bpx_get_car(ph);
			if(bpx_is_nil(data_el)){
				data_el = bpx_get_car(data);
			}
			prism::PlaceholderGoal* g=ph_data.add_goals();
			g->set_id(goal_counter);
			data_el=construct_placeholder_goal(g, ph_el, data_el,split_num);
			if(bpx_is_nil(data_el)){
				ph   = bpx_get_cdr(ph);
				data = bpx_get_cdr(data);
				goal_counter++;
			}else{
				break;
			}
		}
		save_message(filename+std::to_string(file_count),&ph_data,format);
	}
}

extern "C"
int pc_save_placeholder_data_5(void) {
	const char* filename=bpx_get_name(bpx_get_call_arg(1,5));
	SaveFormat format = (SaveFormat) bpx_get_integer(bpx_get_call_arg(2,5));
	TERM ph  =bpx_get_call_arg(3,5);
	TERM data=bpx_get_call_arg(4,5);
	int split_num=bpx_get_integer(bpx_get_call_arg(5,5));//=20000000
	if(FormatHDF5==format){
		save_placeholder_data_hdf5(ph, data, filename, format);
	}else{
		save_placeholder_data(ph, data, filename, format, split_num);
	}
	return BP_TRUE;
}
extern "C"
int pc_set_export_flags_1(void) {
	TERM t=bpx_get_call_arg(1,1);
	export_flags(t);
	return BP_TRUE;
}

extern "C"
int pc_save_options_3(void) {
	//char* filename =bpx_term_2_string(bpx_get_call_arg(1,2));
	const char* filename=bpx_get_name(bpx_get_call_arg(1,3));
	SaveFormat format = (SaveFormat) bpx_get_integer(bpx_get_call_arg(2,3));
	TERM sw_list=bpx_get_call_arg(3,3);
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
	
	//
	while(!bpx_is_nil(sw_list)){
		prism::TensorShape* ts=op.add_tensor_shape();
		TERM pair=bpx_get_car(sw_list);
		TERM tensor_atom=bpx_get_car(pair);
		string tensor_str= bpx_term_2_string(tensor_atom);
		TERM shape=bpx_get_car(bpx_get_cdr(pair));
		tensor_str="tensor("+tensor_str+")";
		ts->set_tensor_name(tensor_str);
		cout<<tensor_str<<endl;
		while(!bpx_is_nil(shape)){
			int dim=bpx_get_integer(bpx_get_car(shape));
			ts->add_shape(dim);
			shape=bpx_get_cdr(shape);
			cout<<"  "<<dim<<endl;
		}
		sw_list=bpx_get_cdr(sw_list);
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


