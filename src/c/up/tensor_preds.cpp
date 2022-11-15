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
#include <vector>
#include "up/save_expl_graph.h"

#ifdef USE_PROTOBUF
#include "external/expl.pb.h"
#include <google/protobuf/text_format.h>
#include <google/protobuf/message.h>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/io/zero_copy_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/util/json_util.h>
using namespace google::protobuf;
#endif

#include <nlohmann/json.hpp>
using json = nlohmann::json;

#ifdef USE_NPY
#include<filesystem> //C++17
#include<libnpy/npy.hpp>
#endif

#ifdef USE_H5
#include <H5Cpp.h>
#endif

#include <regex>

using namespace std;
std::map<string,string> exported_flags;
std::map<string,int> exported_index_range;


#ifdef USE_PROTOBUF
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
#endif

void save_json(const string& outfilename,json& msg) {
	fstream output(outfilename.c_str(), ios::out | ios::trunc);
	output<< msg <<endl;
	cout<<"[SAVE:json] "<<outfilename<<endl;
}


#ifdef USE_NPY
void save_placeholder_data_npy(TERM ph, TERM data, string save_path,SaveFormat format) {
	int goal_counter=0;
	if(!bpx_is_list(ph) || !bpx_is_list(data)){
	}
	std::filesystem::create_directory(save_path);
	json info_data=json::array();
	string filename_json=save_path+"/placeholder.npy.json";
	while(!bpx_is_nil(ph) && !bpx_is_nil(data)){
		string filename_npy=save_path+"/placeholder_"+std::to_string(goal_counter)+".npy";
		json group;
		group["filename"]=filename_npy;
		group["placeholders"]=json::array();
		TERM ph_term   = bpx_get_car(ph);
		TERM data_term = bpx_get_car(data);
		//
		vector<string> placeholders;
		if(!bpx_is_list(ph_term)){
		}
		int n=0;
		while(!bpx_is_nil(ph_term)){
			TERM ph_el= bpx_get_car(ph_term);
			const char* name=bpx_get_name(ph_el);
			group["placeholders"].push_back(name);
			ph_term = bpx_get_cdr(ph_term);
			n++;
		}
		// compute length
		int length=0;
		TERM temp_data_term=data_term;
		while(!bpx_is_nil(temp_data_term)){
			temp_data_term = bpx_get_cdr(temp_data_term);
			length++;
		}
		//
		{
			std::vector<int> data_table;
			for(int i=0;!bpx_is_nil(data_term);i++){ //length
				TERM data_sample_term= bpx_get_car(data_term);
				if(!bpx_is_list(data_sample_term)){
					char* s =bpx_term_2_string(data_sample_term);
					printf("[ERROR] A sample term is not a list: %s\n",s);
				}
				for(int j=0;!bpx_is_nil(data_sample_term);j++){ //n
					TERM data_el_term= bpx_get_car(data_sample_term);
					char* name =bpx_term_2_string(data_el_term);
					int v=stoi(name);
					//data_table[i*n+j]=v;
					data_table.push_back(v);
					data_sample_term = bpx_get_cdr(data_sample_term);
				}
				data_term = bpx_get_cdr(data_term);
			}
			// save dataset
			{
				bool fortran_order=false;
				std::vector<long unsigned> shape={length, n};
				npy::SaveArrayAsNumpy(filename_npy, fortran_order, shape.size(), shape.data(), data_table);
			}
		}
		info_data.push_back(group);
		// next
		ph   = bpx_get_cdr(ph);
		data = bpx_get_cdr(data);
		goal_counter++;
	}
	save_json(filename_json,info_data);
}
#endif

#ifdef USE_H5
/*
 * ph  =[ph_term1,   ph_term2,  ...]
 * data=[data_term1, data_term2,...]
 * Consifering pred(ph1,ph2) as ph_term1, then,
 * ph_term1=[$ph1,$ph2]
 * n=2
 * data_term1: length x n
 * */
void save_placeholder_data_hdf5(TERM ph, TERM data, string filename,SaveFormat format) {
	int goal_counter=0;
	if(!bpx_is_list(ph) || !bpx_is_list(data)){
	}
	H5::H5File file( filename, H5F_ACC_TRUNC );
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
			for(int i=0;!bpx_is_nil(data_term);i++){ //length
				TERM data_sample_term= bpx_get_car(data_term);
				if(!bpx_is_list(data_sample_term)){
					char* s =bpx_term_2_string(data_sample_term);
					printf("[ERROR] A sample term is not a list: %s\n",s);
				}
				for(int j=0;!bpx_is_nil(data_sample_term);j++){ //n
					TERM data_el_term= bpx_get_car(data_sample_term);
					char* name =bpx_term_2_string(data_el_term);
					int v=stoi(name);
					data_table[i*n+j]=v;
					data_sample_term = bpx_get_cdr(data_sample_term);
				}
				data_term = bpx_get_cdr(data_term);
			}
			// save dataset
			{
				hsize_t     dimsf[2];              // dataset dimensions
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
}
#endif

TERM construct_placeholder_goal_json(json &g, TERM ph_term, TERM data_term,int split_num) {
	// adding placeholder list
	g["placeholders"]=json::array();
	if(bpx_is_list(ph_term)){
		while(!bpx_is_nil(ph_term)){
			TERM ph_el= bpx_get_car(ph_term);
			const char* name=bpx_get_name(ph_el);
			json ph;
			ph["name"]=name;
			g["placeholders"].push_back(ph);
			ph_term = bpx_get_cdr(ph_term);
		}
	}
	// adding records
	g["records"]=json::array();
	if(!bpx_is_list(data_term)){
		printf("[ERROR] A data term is not a list");
	}
	for(int i=0; i<split_num && !bpx_is_nil(data_term); i++){
		TERM data_sample_term= bpx_get_car(data_term);
		//prism::DataRecord
		json r;
		if(!bpx_is_list(data_sample_term)){
			char* s =bpx_term_2_string(data_sample_term);
			printf("[ERROR] A sample term is not a list: %s\n",s);
		}
		r["items"]=json::array();
		while(!bpx_is_nil(data_sample_term)){
			TERM data_el_term= bpx_get_car(data_sample_term);
			char* name =bpx_term_2_string(data_el_term);
			r["items"].push_back(name);
			data_sample_term = bpx_get_cdr(data_sample_term);
		}
		g["records"].push_back(r);
		data_term = bpx_get_cdr(data_term);
	}
	return data_term;
}


void save_placeholder_data_json(TERM ph, TERM data, string filename,SaveFormat format,int split_num) {
	int goal_counter=0;
	TERM data_el=bpx_build_nil();
	if(!bpx_is_list(ph) || !bpx_is_list(data)){
	}
	// for split save
	for(int file_count=0;!bpx_is_nil(ph) && !bpx_is_nil(data);file_count++){
		// ph_data: one file
		json ph_data;
		ph_data["goals"]=json::array();
		while(!bpx_is_nil(ph) && !bpx_is_nil(data)){
			TERM ph_el   = bpx_get_car(ph);
			if(bpx_is_nil(data_el)){
				data_el = bpx_get_car(data);
			}
			json g;
			g["id"]=goal_counter;
			data_el=construct_placeholder_goal_json(g, ph_el, data_el,split_num);
			ph_data["goals"].push_back(g);
			if(bpx_is_nil(data_el)){
				ph   = bpx_get_cdr(ph);
				data = bpx_get_cdr(data);
				goal_counter++;
				file_count=0;
			}else{
				break;
			}
		}
		save_json(filename+std::to_string(goal_counter)+"_"+std::to_string(file_count),ph_data);
	}
}

#ifdef USE_PROTOBUF

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


void save_placeholder_data(TERM ph, TERM data, string filename,SaveFormat format,int split_num) {
	int goal_counter=0;
	TERM data_el=bpx_build_nil();
	if(!bpx_is_list(ph) || !bpx_is_list(data)){
	}
	// for split save
	for(int file_count=0;!bpx_is_nil(ph) && !bpx_is_nil(data);file_count++){
		// ph_data: one file
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
				file_count=0;
			}else{
				break;
			}
		}
		save_message(filename+std::to_string(goal_counter)+"_"+std::to_string(file_count),&ph_data,format);
	}
}

#endif

extern "C"
int pc_save_placeholder_data_5(void) {
	const char* filename=bpx_get_name(bpx_get_call_arg(1,5));
	SaveFormat format = (SaveFormat) bpx_get_integer(bpx_get_call_arg(2,5));
	TERM ph  =bpx_get_call_arg(3,5);
	TERM data=bpx_get_call_arg(4,5);
	int split_num=bpx_get_integer(bpx_get_call_arg(5,5));//=20000000
	switch(format){
		case FormatHDF5:
#ifdef USE_H5
			save_placeholder_data_hdf5(ph, data, filename, format);
#else
			printf("[ERROR] hdf5 format is not implemented (please compile prism with the USE_H5 option)\n");
#endif
			break;
		case FormatNPY:
#ifdef USE_NPY
			save_placeholder_data_npy(ph, data, filename, format);
#else
			printf("[ERROR] npy format is not implemented (please compile prism with the USE_NPY option)\n");
#endif
			break;
		case FormatPb:
#ifdef USE_PROTOBUF
			save_placeholder_data(ph, data, filename, format, split_num);
#else
			printf("[ERROR] Pb format is not implemented (please compile prism with the USE_PB option)\n");
#endif
			break;
		case FormatJson:
			save_placeholder_data_json(ph, data, filename, format, split_num);
			break;
		default:
			cerr << "Unknown format." << endl;  
			break;
	}
	return BP_TRUE;
}


///////////

string get_dataset_name(string term_name){
	std::string r = std::regex_replace(term_name, std::regex("[\\[\\],\\)\\(\\'$]+"), "_");
	return r;
}

#ifdef USE_NPY
void save_embedding_tensor_npy(const string filename, const string group_name, const string dataset_name, TERM term_list, TERM shape){
	string filename_npy=filename+".npy";
	string filename_json=filename+".npy.json";
	TERM el   = bpx_get_car(term_list);
	TERM next = bpx_get_cdr(term_list);
	if(!bpx_is_list(term_list) || !bpx_is_list(shape)){
		return;	
	}
	json embedding_data;
	// shape=[n1,n2]
	embedding_data["group"]=group_name;
	embedding_data["name"]=dataset_name;
	embedding_data["shape"]=json::array();
	std::vector<unsigned long> shape_t;
	int n=1;
	while(!bpx_is_nil(shape)){
		TERM term_n1 = bpx_get_car(shape);
		int n1=bpx_get_integer(term_n1);
		shape = bpx_get_cdr(shape);
		shape_t.push_back(n1);
		embedding_data["shape"].push_back(n1);
		n*=n1;
	}
	std::vector<double> data(n);
	//std::vector<float> data();
	while(!bpx_is_nil(term_list)){
		TERM el = bpx_get_car(term_list);
		term_list = bpx_get_cdr(term_list);
		//el=[index1,index2]
		int i=0;
		int idx=0;
		while(!bpx_is_nil(el)){
			TERM term_index1 = bpx_get_car(el);
			idx+=bpx_get_integer(term_index1);
			//data.push_back(idx)
			if(i+1<shape_t.size()){
				idx=idx*shape_t[i+1];
			}
			i++;
			el = bpx_get_cdr(el);
		}
		data[idx]=1.0f;
	}
	// save dataset
	{
		bool fortran_order=false;
		npy::SaveArrayAsNumpy(filename_npy, fortran_order, shape_t.size(), shape_t.data(), data);
		embedding_data["filename"]=filename_npy;
		save_json(filename_json,embedding_data);
	}
}
#endif

#ifdef USE_H5
void save_embedding_matrix_hdf5(const string filename, const string group_name, const string dataset_name, TERM term_list, TERM shape){
	TERM el   = bpx_get_car(term_list);
	TERM next = bpx_get_cdr(term_list);
	if(!bpx_is_list(term_list) || !bpx_is_list(shape)){
		return;	
	}
	// shape=[n1,n2]
	TERM term_n1 = bpx_get_car(shape);
	int n1=bpx_get_integer(term_n1);
	TERM shape1 = bpx_get_cdr(shape);
	TERM term_n2 = bpx_get_car(shape1);
	int n2=bpx_get_integer(term_n2);
	float* data_table=new float[n1*n2]();
	while(!bpx_is_nil(term_list)){
		TERM el = bpx_get_car(term_list);
		term_list = bpx_get_cdr(term_list);
		//el=[index1,index2]
		TERM term_index1 = bpx_get_car(el);
		int index1=bpx_get_integer(term_index1);
		TERM el2 = bpx_get_cdr(el);
		TERM term_index2 = bpx_get_car(el2);
		int index2=bpx_get_integer(term_index2);
		data_table[n1*index1+index2]=1.0f;
	}
	// save dataset
	H5::H5File file( filename, H5F_ACC_TRUNC );
	{
		hsize_t dimsf[2];   // dataset dimensions
		dimsf[0] = n1;
		dimsf[1] = n2;
		H5::DataSpace dataspace(2, dimsf );
		H5::FloatType datatype( H5::PredType::NATIVE_FLOAT );
		datatype.setOrder( H5T_ORDER_LE );
		file.createGroup( group_name);
		H5::DataSet dataset = file.createDataSet( group_name+"/"+dataset_name, datatype, dataspace );
		dataset.write( data_table, H5::PredType::NATIVE_FLOAT );
	}
}

void save_embedding_vector_hdf5(const string filename, const string group_name, const string dataset_name, TERM term_list, TERM shape){
	TERM el   = bpx_get_car(term_list);
	TERM next = bpx_get_cdr(term_list);
	if(!bpx_is_list(term_list) || !bpx_is_list(shape)){
		return;	
	}
	// shape=[n1,n2]
	TERM term_n1 = bpx_get_car(shape);
	int n1=bpx_get_integer(term_n1);
	float* data_table=new float[n1]();
	while(!bpx_is_nil(term_list)){
		TERM el = bpx_get_car(term_list);
		term_list = bpx_get_cdr(term_list);
		//el=[index1,index2]
		TERM term_index1 = bpx_get_car(el);
		int index1=bpx_get_integer(term_index1);
		data_table[index1]=1.0f;
	}
	// save dataset
	H5::H5File file( filename, H5F_ACC_TRUNC );
	{
		hsize_t dimsf[1];   // dataset dimensions
		dimsf[0] = n1;
		H5::DataSpace dataspace(1, dimsf );
		H5::FloatType datatype( H5::PredType::NATIVE_FLOAT );
		datatype.setOrder( H5T_ORDER_LE );
		file.createGroup( group_name);
		H5::DataSet dataset = file.createDataSet( group_name+"/"+dataset_name, datatype, dataspace );
		dataset.write( data_table, H5::PredType::NATIVE_FLOAT );
	}
}
#endif


extern "C"
int pc_save_embedding_tensor_6(void) {
	const char* filename=bpx_get_name(bpx_get_call_arg(1,6));
	const char* group=bpx_get_name(bpx_get_call_arg(2,6));
	const char* term_name=bpx_term_2_string(bpx_get_call_arg(3,6));
	TERM term_list =bpx_get_call_arg(4,6);
	TERM shape=bpx_get_call_arg(5,6);
	SaveFormat format = (SaveFormat) bpx_get_integer(bpx_get_call_arg(6,6));
	// 
	string dataset_name = get_dataset_name(term_name);
	
	printf("[SAVE] %s\n",filename);
	printf("[INFO] %s\n",group);
	printf("[INFO] %s -> %s\n",term_name,dataset_name.c_str());
	
	// compute length
	int length=0;
	TERM temp_data_term=shape;
	while(!bpx_is_nil(temp_data_term)){
		temp_data_term = bpx_get_cdr(temp_data_term);
		length++;
	}
	switch(format){
		case FormatHDF5:
#ifdef USE_H5
			// only supported matrix and FormatHDF5
			if(length==1){
				save_embedding_vector_hdf5(filename, group, dataset_name, term_list, shape);
			}else if(length==2){
				save_embedding_matrix_hdf5(filename, group, dataset_name, term_list, shape);
			}else{
				return BP_FALSE;
			}
#else
			printf("[ERROR] hdf5 format is not implemented (please compile prism with the USE_H5 option)\n");
#endif
			break;
		case FormatNPY:
#ifdef USE_NPY
			save_embedding_tensor_npy(filename, group, dataset_name, term_list, shape);
#else
			printf("[ERROR] npy format is not implemented (please compile prism with the USE_NPY option)\n");
#endif
			break;
		default:
			cerr << "Unknown format." << endl;  
			break;
	}
	return BP_TRUE;
}

/////////

/*
 * This function stores the list of received key-value pairs in exported_flags (global variable)
 * */
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

extern "C"
int pc_set_export_flags_1(void) {
	TERM t=bpx_get_call_arg(1,1);
	export_flags(t);
	return BP_TRUE;
}

///////////
int run_save_options_json(const char* filename, SaveFormat format,TERM sw_list){
	json op;
	op["flags"]=json::array();
	// set flags
	for(auto itr:exported_flags){
		json f;
		f["key"]=itr.first;
		f["value"]=itr.second;
		op["flags"].push_back(f);
	}
	//set index_range
	op["index_range"]=json::array();
	for(auto itr:exported_index_range){
		json ir;
		ir["index"]=itr.first;
		ir["range"]=itr.second;
		op["index_range"].push_back(ir);
	}
	//
	op["tensor_shape"]=json::array();
	while(!bpx_is_nil(sw_list)){
		//prism::TensorShape
		json ts;
		TERM pair=bpx_get_car(sw_list);
		TERM tensor_atom=bpx_get_car(pair);
		string tensor_str= bpx_term_2_string(tensor_atom);
		TERM shape=bpx_get_car(bpx_get_cdr(pair));
		tensor_str="tensor("+tensor_str+")";
		ts["tensor_name"]=tensor_str;
		ts["shape"]=json::array();
		//cout<<tensor_str<<endl;
		while(!bpx_is_nil(shape)){
			int dim=bpx_get_integer(bpx_get_car(shape));
			ts["shape"].push_back(dim);
			shape=bpx_get_cdr(shape);
			//cout<<"  "<<dim<<endl;
		}
		op["tensor_shape"].push_back(ts);
		sw_list=bpx_get_cdr(sw_list);
	}
	// save
	save_json(filename,op);
	return BP_TRUE;
}

#ifdef USE_PROTOBUF
int run_save_options(const char* filename, SaveFormat format,TERM sw_list){
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
		//cout<<tensor_str<<endl;
		while(!bpx_is_nil(shape)){
			int dim=bpx_get_integer(bpx_get_car(shape));
			ts->add_shape(dim);
			shape=bpx_get_cdr(shape);
			//cout<<"  "<<dim<<endl;
		}
		sw_list=bpx_get_cdr(sw_list);
	}
	// save
	save_message(filename,&op,format);
	return BP_TRUE;
}
#endif

extern "C"
int pc_save_options_3(void) {
	//char* filename =bpx_term_2_string(bpx_get_call_arg(1,2));
	const char* filename=bpx_get_name(bpx_get_call_arg(1,3));
	SaveFormat format = (SaveFormat) bpx_get_integer(bpx_get_call_arg(2,3));
	TERM sw_list=bpx_get_call_arg(3,3);
#ifdef USE_PROTOBUF
	int r=run_save_options(filename,format,sw_list);
#else
	int r=run_save_options_json(filename,format,sw_list);
#endif
	return r;
}

extern "C"
int pc_set_index_range_2(void) {
	char* index_name= bpx_term_2_string(bpx_get_call_arg(1,2));
	int range=bpx_get_integer(bpx_get_call_arg(2,2));
	string s=index_name;
	exported_index_range[s]=range;
	return BP_TRUE;
}


