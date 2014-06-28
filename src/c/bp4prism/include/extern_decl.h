/* Updated the arguments for these functions to so that they are valid prototypes for the system  branch */
#include <time.h>
#include <signal.h>
#include <stdio.h>

extern char *inst_name[];
extern BPLONG foreign_exception;
extern void (*user_signal_action[NSIG])(int,void *);
extern void *user_signal_data[NSIG];
extern EVENT_FUNC event_func;


// Prototypes are listed by module where they are defined
//
//	arith.c prototypes
//
extern void init_arith_sym(void);
extern BPLONG bp_math_add(BPLONG op1,BPLONG op2);
extern BPLONG bp_math_sub(BPLONG op1,BPLONG op2);
extern BPLONG bp_math_mul(BPLONG op1,BPLONG op2);
extern BPLONG bp_math_divge(BPLONG op1,BPLONG op2);
extern BPLONG bp_math_divle(BPLONG op1,BPLONG op2);
extern BPLONG bp_math_div(BPLONG op1,BPLONG op2);
extern BPLONG bp_math_idiv(BPLONG op1,BPLONG op2);
extern BPLONG bp_math_idiv_div(BPLONG op1,BPLONG op2);
extern BPLONG bp_math_mod(BPLONG op1,BPLONG op2);
extern BPLONG bp_math_rem(BPLONG op1,BPLONG op2);
extern BPLONG bp_bitwise_and(BPLONG op1,BPLONG op2);
extern BPLONG bp_bitwise_or(BPLONG op1,BPLONG op2);
extern BPLONG bp_bitwise_shiftl(BPLONG op1,BPLONG op2);
extern BPLONG bp_bitwise_shiftr(BPLONG op1,BPLONG op2);
extern BPLONG bp_bitwise_complement(BPLONG op1);
extern BPLONG bp_math_integer(BPLONG op1);
extern BPLONG bp_math_sign(BPLONG op1);
extern int b_FLOAT_SIGN_cf(BPLONG op1,BPLONG op2);
extern BPLONG bp_math_max(BPLONG op1,BPLONG op2);
extern BPLONG bp_math_min(BPLONG op1,BPLONG op2);
extern BPLONG bp_math_min1(BPLONG op1);
extern BPLONG bp_math_max1(BPLONG op1);
extern BPLONG bp_math_sum1(BPLONG op1);
extern BPLONG bp_math_fract_part(BPLONG op1);
extern BPLONG bp_math_int_part(BPLONG op1);
extern BPLONG bp_math_random0(void);
extern BPLONG bp_math_random1(BPLONG op1);
extern BPLONG bp_gcd_int_int(BPLONG i1,BPLONG i2);
extern BPLONG bp_math_gcd(BPLONG op1,BPLONG op2);
extern int b_GCD_ccf(BPLONG op1,BPLONG op2,BPLONG gcd);
extern int equal_to(BPLONG op1,BPLONG op2);
extern int greater_than(BPLONG op1,BPLONG op2);
extern int greater_equal(BPLONG op1,BPLONG op2);
extern int b_FLOAT_MINUS_cf(BPLONG op1,BPLONG op2);
extern BPLONG bp_float_log(BPLONG op1);
extern int b_FLOAT_LOG_cf(BPLONG op1,BPLONG op2);
extern BPLONG bp_float_log2(BPLONG op1,BPLONG op2);
extern int b_FLOAT_LOG2_ccf(BPLONG op1,BPLONG op2,BPLONG op3);
extern BPLONG bp_pow_int_int(BPLONG base, BPLONG ex);
extern BPLONG bp_math_pow(BPLONG op1,BPLONG op2);
extern int b_FLOAT_POW_ccf(BPLONG op1,BPLONG op2,BPLONG op3);
extern int b_REM_ccf(BPLONG op1,BPLONG op2,BPLONG op3);
extern BPLONG bp_float_sqrt(BPLONG op1);
extern int b_FLOAT_SQRT_cf(BPLONG op1,BPLONG op2);
extern BPLONG bp_float_abs(BPLONG op1);
extern int b_FLOAT_ABS_cf(BPLONG op1,BPLONG op2);
extern BPLONG bp_float_exp(BPLONG op1);
extern int b_FLOAT_EXP_cf(BPLONG op1,BPLONG op2);
extern BPLONG bp_float_sin(BPLONG op1);
extern int b_FLOAT_SIN_cf(BPLONG op1,BPLONG op2);
extern BPLONG bp_float_cos(BPLONG op1);
extern int b_FLOAT_COS_cf(BPLONG op1,BPLONG op2);
extern BPLONG bp_float_tan(BPLONG op1);
extern int b_FLOAT_TAN_cf(BPLONG op1,BPLONG op2);
extern BPLONG bp_float_atan(BPLONG op1);
extern int b_FLOAT_ATAN_cf(BPLONG op1,BPLONG op2);
extern BPLONG bp_float_atan2(BPLONG op1,BPLONG op2);
extern BPLONG bp_float_asin(BPLONG op1);
extern int b_FLOAT_ASIN_cf(BPLONG op1,BPLONG op2);
extern BPLONG bp_float_acos(BPLONG op1);
extern int b_FLOAT_ACOS_cf(BPLONG op1,BPLONG op2);
extern int b_FLOAT_WRITE_c(BPLONG op);
extern BPLONG bp_float_floor(BPLONG op1);
extern int b_FLOAT_FLOOR_cf(BPLONG op1,BPLONG op2);
extern BPLONG bp_float_float(BPLONG op1);
extern int b_FLOAT_FLOAT_cf(BPLONG op1,BPLONG op2);
extern BPLONG bp_float_round(BPLONG op1);
extern int b_FLOAT_ROUND_cf(BPLONG op1,BPLONG op2);
extern BPLONG bp_float_truncate(BPLONG op1);
extern int b_FLOAT_TRUNCATE_cf(BPLONG op1,BPLONG op2);
extern BPLONG bp_float_ceiling(BPLONG op1);
extern int b_FLOAT_CEILING_cf(BPLONG op1,BPLONG op2);
extern BPLONG bp_math_xor(BPLONG op1,BPLONG op2);
extern int b_XOR_ccf(BPLONG op1,BPLONG op2,BPLONG op3);
extern int b_FLOAT_PI_f(BPLONG op1);
extern int b_FLOAT_E_f(BPLONG op1);
extern int b_RANDOM_f(BPLONG op1);
extern int b_RANDOM_cf(BPLONG op1,BPLONG op2);
extern int b_MAX_ccf(BPLONG op1,BPLONG op2,BPLONG op3);
extern int b_MIN_ccf(BPLONG op1,BPLONG op2,BPLONG op3);
extern int b_MAX_cf(BPLONG op1,BPLONG op2);
extern int b_MIN_cf(BPLONG op1,BPLONG op2);
extern int b_SUM_cf(BPLONG op1,BPLONG op2);
extern int b_FLOAT_FRACT_PART_cf(BPLONG op1,BPLONG op2);
extern int b_FLOAT_INT_PART_cf(BPLONG op1,BPLONG op2);
extern int prettymuch_equal(double op1,double op2);
extern BPLONG bp_access_array(BPLONG arr,BPLONG indexes);
extern BPLONG bp_access_one_array(BPLONG arr,BPLONG index);
extern BPLONG eval_arith(BPLONG ex);
extern int b_EVAL_ARITH_cf(BPLONG ex,BPLONG res);

//
//	asp.c prototypes
//
extern void c_init_rel_num(void);
extern int b_ASP_TRUE_cc(BPLONG FDVar,BPLONG E);
extern int asp_domain_true(BPLONG_PTR dv_ptr,BPLONG elm);
extern int asp_domain_false(BPLONG_PTR dv_ptr,BPLONG elm);
extern void asp_domain_set_false(BPLONG_PTR dv_ptr,BPLONG elm);
extern int b_ASP_NEW_REL(BPLONG Arity,BPLONG MinSizeVect,BPLONG RelNo);
extern void print_rel_info(BPLONG rel);
extern int c_asp_get_rel_col_min_size(void);
extern int asp_is_indexed_tuple(BPLONG this_asp_rel_num,BPLONG code,BPLONG I,BPLONG Ai);
extern int c_asp_filter_indexed1(void);
extern int b_ASP_ADD_TUPLE(BPLONG n);
extern int b_ASP_ENCODE_TUPLE(BPLONG n);
extern int b_ASP_DECODE(BPLONG n);
extern int b_ASP_CARD1_PROP_IN_TO_OUT(BPLONG OUT,BPLONG E);
extern int b_ASP_PROP_BASE_TO_INDEXED(BPLONG FDVar,BPLONG Code,BPLONG WhichBase);
extern int b_ASP_SELECT_REL_cff(BPLONG INOUTs,BPLONG IN,BPLONG OUT);

//
//	assert.c prototypes
//
extern void initialize_free_records(void);
extern BPLONG total_free_records_size(void);
extern int c_print_pred_ref_count(void);
extern int c_set_dyn_hashtable_size(void);
extern int b_INC_PRED_REF_COUNT_c(BPLONG PredPtr);
extern int b_DEC_PRED_REF_COUNT_c(BPLONG PredPtr);
extern int b_INC_PRED_RETR_COUNT_c(BPLONG PredPtr);
extern int b_DEC_PRED_RETR_COUNT_c(BPLONG PredPtr);
extern int b_ABOLISH_cc(BPLONG f,BPLONG n);
extern void abolish_pred(InterpretedPredPtr pred_ptr);
extern void release_clause_record_space(InterpretedClausePtr clause_record_ptr);
extern void release_term_space(BPLONG term);
extern int b_REMOVE_CLAUSE_c(BPLONG clause_record);
extern void locate_and_free_clause_record(InterpretedPredPtr pred_ptr,BPLONG clause_record);
extern void free_cell_of_removed_in_bucket(InterpretedPredPtr pred_ptr,InterpretedPredBucketPtr bucket_ptr,BPLONG removed_clause_record);
extern void disconnect_cell_of_removed_clause(InterpretedPredBucketPtr bucket_ptr,BPLONG_PTR cell_ptr);
extern int c_initialize_interpreted_pred(void);
extern int b_ASSERTA_cc(BPLONG Head,BPLONG Body);
extern int b_ASSERTZ_cc(BPLONG Head,BPLONG Body);
extern InterpretedPredBucketPtr new_interpreted_bucket(void);
extern BPLONG_PTR new_interpreted_pred_hashtable(int size);
extern InterpretedPredPtr new_interpreted_pred_record(BPLONG size);
extern InterpretedPredPtr initialize_interpreted_pred(SYM_REC_PTR sym_ptr,BPLONG type,BPLONG size);
extern int rehash_interpreted_pred(InterpretedPredPtr pred_ptr);
extern BPLONG create_clause_record(InterpretedPredPtr pred_ptr,BPLONG head,BPLONG body);
extern BPLONG_PTR asserta_clause_record(InterpretedPredBucketPtr bucket_ptr,BPLONG clause_record);
extern int asserta_interpreted_pred(InterpretedPredPtr pred_ptr,BPLONG head,BPLONG body);
extern BPLONG_PTR assertz_clause_record(InterpretedPredBucketPtr bucket_ptr,BPLONG clause_record);
extern int assertz_interpreted_pred(InterpretedPredPtr pred_ptr,BPLONG head,BPLONG body);
extern BPLONG hashval_in_assert(InterpretedClausePtr clause_record_ptr);
extern int b_ASSERTABLE_c(BPLONG Head);
extern int b_RETRACTABLE_c(BPLONG Head);
extern int b_GET_PRED_PTR_cff(BPLONG Head,BPLONG PredPtr,BPLONG IsDynamic);
extern int b_GET_CLAUSES_cfff(BPLONG Head,BPLONG Clauses,BPLONG Key, BPLONG TimeStamp);
extern void Cboot_assert(void);
extern BPLONG numberVarCopyToParea(BPLONG term,BPLONG *varno);
extern BPLONG numberVarCopyListToParea(BPLONG term,BPLONG *varno);
extern BPLONG numberVarCopyCommaToParea(BPLONG term, BPLONG *varno);


//
//	findall.c
//
int identical_gterms(BPLONG op1, BPLONG op2);
extern void init_faa_gterms_htable();
extern void gterms_table_statistics(GTERMS_HTABLE_PTR gterms_htable_ptr, int *nTerms, int *maxTTCollisions, float *aveTTCollisions);
extern void allocate_gterms_htable(GTERMS_HTABLE_PTR gterms_htable_ptr, int size);
extern int gterms_htable_num_of_occupied_slots(GTERMS_HTABLE_PTR gterms_htable_ptr);
extern void init_gterms_htable(GTERMS_HTABLE_PTR gterms_htable_ptr);
extern BPLONG register_gterms_htable(GTERMS_HTABLE_PTR gterms_htable_ptr,BPLONG term,BPLONG hcode);
extern void expand_gterms_htable(GTERMS_HTABLE_PTR gterms_htable_ptr);
extern int b_FINDALL_COPY_ARGS(void);
extern void init_findall_area(void);
extern int c_findall_pre(void);
extern int c_findall_post(void);
extern int b_FINDALL_INSERT_cc(BPLONG SymPtr, BPLONG value);
extern int c_FINDALL_GET(void);
extern int c_FINDALL_AREA_SIZE(void);
extern BPLONG copy_answers_faa_to_heap(BPLONG list);
extern BPLONG copy_answers_shared_faa_to_heap(BPLONG list);
extern BPLONG copy_term_heap_to_faa(BPLONG value);
extern BPLONG make_cons_in_faa(BPLONG car, BPLONG cdr);
extern BPLONG numbered_area_size(NUMBERED_TERM_AREA_RECORD_PTR area_record_ptr);
extern BPLONG numberVarCopyToNumberedArea(NUMBERED_TERM_AREA_RECORD_PTR area_record_ptr,BPLONG term, BPLONG_PTR hcode_ptr, BPLONG_PTR ground_flag_ptr);
extern BPLONG numberVarCopyListToNumberedArea(NUMBERED_TERM_AREA_RECORD_PTR area_record_ptr,BPLONG term, BPLONG_PTR hcode_ptr, BPLONG_PTR ground_flag_ptr);
extern BPLONG unnumberVarSharedTerm_faa_to_heap(BPLONG term);
extern BPLONG unnumberVarSharedList_faa_to_heap(BPLONG term);


//
//	bigint.c
//
extern int c_test_bigint(void);
extern BPLONG bp_int_to_bigint(BPLONG a);
extern BPLONG bp_ubig_to_int(BPLONG size, UBIGINT x);
extern BPLONG bp_bigint_to_int(BPLONG op);
extern double bp_ubig_to_double(BPLONG size, UBIGINT x);
extern double bp_bigint_to_double(BPLONG op);
extern BPLONG bp_double_to_bigint(double a);
extern void bp_inc_ubig(BPLONG_PTR ysize_ptr, UBIGINT y);
extern void bp_add_ubig_ubig(BPLONG xsize, UBIGINT x, BPLONG ysize, UBIGINT y, BPLONG_PTR zsize_ptr, UBIGINT z);
extern void bp_dec_ubig(BPLONG_PTR ysize_ptr, UBIGINT y);
extern void bp_sub_ubig_ubig(BPLONG ysize, UBIGINT y, BPLONG xsize, UBIGINT x, BPLONG_PTR zsize_ptr, UBIGINT z);
extern void bp_mul_ubig_ubig(BPLONG xsize, UBIGINT x, BPLONG ysize, UBIGINT y, BPLONG_PTR zsize_ptr, UBIGINT z);
extern void bp_div_ubig_ubig(BPLONG xsize, UBIGINT x, BPLONG ysize, UBIGINT y, BPLONG_PTR qsize_ptr, UBIGINT q,
                             BPLONG_PTR rsize_ptr, UBIGINT r, UBIGINT subtractBuf);
extern void bp_and_ubig_ubig(BPLONG xsize, UBIGINT x, BPLONG ysize, UBIGINT y, BPLONG_PTR zsize_ptr, UBIGINT z);
extern void bp_or_ubig_ubig(BPLONG xsize, UBIGINT x, BPLONG ysize, UBIGINT y, BPLONG_PTR zsize_ptr, UBIGINT z);
extern void bp_xor_ubig_ubig(BPLONG xsize, UBIGINT x, BPLONG ysize, UBIGINT y, BPLONG_PTR zsize_ptr, UBIGINT z);
extern void bp_shiftl_ubig_int(BPLONG xsize, UBIGINT x, BPLONG y, BPLONG_PTR zsize_ptr, UBIGINT z);
extern void bp_shiftr_ubig_int(BPLONG xsize, UBIGINT x, BPLONG y, BPLONG_PTR zsize_ptr, UBIGINT z);
extern void bp_onescomplement_ubig(BPLONG_PTR xsize_ptr, UBIGINT x);
extern void bp_twoscomplement_ubig(BPLONG xsize, UBIGINT x);
extern void bp_twoscomplement_magnitude(BPLONG_PTR zsize_ptr, UBIGINT z);
extern void bp_shiftr_twoscomplement_int(BPLONG xsize, UBIGINT x, BPLONG y, BPLONG_PTR zsize_ptr, UBIGINT z);
extern void bp_copy_ubig(BPLONG xsize, UBIGINT x, BPLONG ysize, UBIGINT y);

extern BPLONG bp_abs_bigint(BPLONG op);
extern BPLONG bp_neg_bigint(BPLONG op);
extern int bp_sign_bigint(BPLONG op);
extern int bp_size_bigint(BPLONG op);
extern int bp_compare_mag_mag(BPLONG size, UBIGINT x, UBIGINT y);
extern BPLONG bp_add_bigint_bigint(BPLONG op1,BPLONG op2);
extern BPLONG bp_sub_bigint_bigint(BPLONG op1,BPLONG op2);
extern BPLONG bp_mul_bigint_bigint(BPLONG op1,BPLONG op2);
extern BPLONG bp_div_bigint_bigint(BPLONG op1,BPLONG op2);
extern BPLONG bp_mod_bigint_bigint(BPLONG op1,BPLONG op2);
extern int bp_compare_bigint_bigint(BPLONG op1,BPLONG op2);
extern BPLONG bp_and_bigint_bigint(BPLONG op1, BPLONG op2);
extern BPLONG bp_or_bigint_bigint(BPLONG op1, BPLONG op2);
extern BPLONG bp_xor_bigint_bigint(BPLONG op1, BPLONG op2);
extern BPLONG bp_shiftl_bigint_int(BPLONG op1, BPLONG op2);
extern BPLONG bp_shiftr_bigint_int(BPLONG op1, BPLONG op2);
extern BPLONG bp_updiv_bigint_bigint(BPLONG op1, BPLONG op2);
extern BPLONG bp_lowdiv_bigint_bigint(BPLONG op1, BPLONG op2);
extern BPLONG bp_gcd_bigint_bigint(BPLONG i1,BPLONG i2);
extern BPLONG bp_pow_bigint_int(BPLONG op1,BPLONG op2);
extern void bp_print_bigint(BPLONG op);

//
//	buildtins.c prototypes
//
//extern int (*builtins[MAXBS])(void);
extern void init_builtins(void);
extern int b_GET_ARCH_f(BPLONG arch);

//
//	cfd.c prototypes
//
extern BPLONG_PTR new_bv_domain_var(BPLONG from,BPLONG to,BPLONG first,BPLONG last,BPLONG size,BPULONG bv_word);
extern void domain_add_elm(BPLONG_PTR dv_ptr,BPLONG elm);
extern void domain_set_true_bv(BPLONG_PTR dv_ptr,BPLONG elm);
extern BPLONG_PTR new_it_domain_var(BPLONG first,BPLONG last);
extern int b_CFD_COMPUTE_MINS_MAXS(BPLONG Arity,BPLONG Tuples,BPLONG Mins,BPLONG Maxs);
extern void cfd_transform_tuples(BPLONG n,BPLONG Tuples,BPLONG_PTR MinArray);
extern int b_CFD_TRANSFORM_TUPLES(BPLONG Arity,BPLONG Tuples,BPLONG Mins);
extern int b_CFD_BUILD_TRIES_IN(BPLONG Maxs,BPLONG Tuples,BPLONG A2Tries);
extern int b_CFD_BUILD_TRIES_NOTIN(BPLONG CompVars,BPLONG HTable,BPLONG A2Tries);
extern void initialize_min_max_arrays(BPLONG n,BPLONG nk,BPLONG_PTR MinArray,BPLONG_PTR MaxArray);
extern void compute_mins_maxs_in(BPLONG n,BPLONG nk,BPLONG Tuples,BPLONG_PTR MinArray,BPLONG_PTR MaxArray);
extern void compute_mins_maxs_notin(BPLONG arg_no,BPLONG n,BPLONG nk,BPLONG_PTR htable_ptr,BPLONG htable_size,
                                    BPLONG_PTR tuple_ptr,BPLONG_PTR CompVarArray,
                                    BPLONG_PTR MinArray,BPLONG_PTR MaxArray);
extern void print_mins_maxs(BPLONG n,BPLONG nk,BPLONG_PTR MinArray,BPLONG_PTR MaxArray);
extern void initialize_supports(BPLONG n,BPLONG nk,BPLONG_PTR tries_ptr,BPLONG_PTR CompVarMaxArray,
                                BPLONG_PTR MinArray,BPLONG_PTR MaxArray);
extern void compute_supports_in(BPLONG n,BPLONG Tuples,BPLONG_PTR tries_ptr,BPLONG_PTR MinArray,BPLONG_PTR MaxArray);
extern void compute_supports_notin(BPLONG arg_no,BPLONG n,BPLONG_PTR htable_ptr,BPLONG htable_size,
                                   BPLONG_PTR tuple_ptr,BPLONG_PTR CompVarArray,BPLONG_PTR tries_ptr,
                                   BPLONG_PTR MinArray,BPLONG_PTR MaxArray);
extern BPLONG_PTR  add_tuple_into_trie(BPLONG_PTR tuple_ptr,BPLONG i,BPLONG n,
                                       BPLONG_PTR last_cell_ptr,BPLONG_PTR children_ptr);
extern int c_TUPLES_TO_TRIE(void);
extern int exclude_ac_unsupported_from_fd(BPLONG X,BPLONG Y,BPLONG_PTR trie_xy_ptr);
extern int b_CFD_REMOVE_AC_UNSUPPORTED(BPLONG CompVars,BPLONG Tries);
extern int findSupport(BPLONG i,BPLONG n,BPLONG_PTR CompVarArray,BPLONG_PTR SupportArray,BPLONG Nodes);
extern int b_CFD_REMOVE_GAC_UNSUPPORTED(BPLONG CompVars,BPLONG BigTrieRoot);
extern int b_CFD_TRIE_INTERSECT(BPLONG X,BPLONG OldTrie1,BPLONG OldTrie2,BPLONG NewTrie);
extern int cfd_mac_bin_ins(BPLONG x,BPLONG Y,BPLONG TrieXY);
extern int cfd_mac_bin_dom(BPLONG x,BPLONG_PTR dv_ptr_x,BPLONG Y,BPLONG TrieXY,BPLONG TrieYX);
extern int b_CFD_INS(BPLONG X,BPLONG AttachedPairs);
extern int b_CFD_DOM(BPLONG X,BPLONG Ex,BPLONG AttachedPairs);
extern int b_CFD_DIFF_TUPLE(BPLONG Tuple,BPLONG CompVars);
extern int b_CFD_FORWARD_CHECKING(BPLONG HTable,BPLONG CompVars);

//
//	clause.c prototypes
//
extern int aux_copy_clause(BPLONG ref,BPLONG head,BPLONG body);
extern int clause_unify_head_arg(BPLONG head_arg,BPLONG cl_head_arg);

//
//	clpfd.c prototypes
//
extern int dvar_bv(BPLONG op);
extern int dvar_excludable_or_int(BPLONG op);
extern int b_EXCLUDABLE_LIST_c(BPLONG list);
extern int nondvar(BPLONG op);
extern int dvar(BPLONG op);
extern int dvar_or_int(BPLONG op);
extern int count_vars(BPLONG op);
extern int n_vars_gt(BPLONG count0,BPLONG op,BPLONG limit);
extern int trigger_vars_ins(BPLONG op);
extern int trigger_vars_minmax(BPLONG op);
extern int trigger_vars_dom(BPLONG op);
extern int trigger_vars_any_dom(BPLONG op);
extern int exclude_elm_dvars(void);
extern int b_EXCLUDE_ELM_DVARS(BPLONG P_elm,BPLONG P_list1,BPLONG P_list2);
extern int exclude_elm_vcs(void);
extern int b_EXCLUDE_ELM_VCS(BPLONG elm,BPLONG P_list);
extern int b_select_ff(BPLONG Vars,BPLONG BestVar);
extern int b_SELECT_FF_MIN_cf(BPLONG Vars,BPLONG BestVar);
extern int b_SELECT_FF_MAX_cf(BPLONG Vars,BPLONG BestVar);
extern int b_SELECT_MIN_cf(BPLONG Vars,BPLONG BestVar);
extern int b_SELECT_MAX_cf(BPLONG Vars,BPLONG BestVar);
extern int b_CONSTRAINTS_NUMBER_cf(BPLONG Var,BPLONG);
extern int count_cs_list(BPLONG list);
extern int c_fd_degree(void);
extern int dvar_degree_count_connected_vars_cs(BPLONG cs);
extern int dvar_degree_count_connected_vars_frame(BPLONG_PTR f);
extern int dvar_degree_count_connected_vars_term(BPLONG term);
extern void dvar_degree_reset_cs_tags_cs(BPLONG cs);
extern void dvar_degree_reset_cs_tags_frame(BPLONG_PTR f);
extern void dvar_degree_reset_cs_tags_term(BPLONG term);
extern void display_constraint(BPLONG_PTR frame);
extern int display_constraints(void);
extern int c_VV_EQ_C_CON(void);
extern int b_VV_EQ_C_CON_ccc(BPLONG X,BPLONG Y,BPLONG C);
extern int c_VV_EQ_C_CON_aux(BPLONG_PTR dv_ptr_x,BPLONG_PTR dv_ptr_y,BPLONG C);
extern int c_V_EQ_VC_CON(void);
extern int b_V_EQ_VC_CON_ccc(BPLONG X,BPLONG Y,BPLONG C);
extern int c_V_EQ_VC_CON_aux(BPLONG_PTR dv_ptr_x,BPLONG_PTR dv_ptr_y,BPLONG C);
extern int c_UU_EQ_C_CON(void);
extern int c_UU_EQ_C_CON_aux(BPLONG A,BPLONG_PTR dv_ptr_x,BPLONG B,BPLONG_PTR dv_ptr_y,BPLONG C);
extern void print_event_queue(void);
extern int c_U_EQ_UC_CON(void);
extern int c_U_EQ_UC_CON_aux(BPLONG A,BPLONG_PTR dv_ptr_x,BPLONG B,BPLONG_PTR dv_ptr_y,BPLONG C);
extern int b_CLPSET_CARD_BOUND_c(BPLONG SetTerm);
extern int b_CLPSET_LOW_UPDATED_c(BPLONG SetTerm);
extern int b_CLPSET_UP_UPDATED_c(BPLONG SetTerm);
extern BPLONG clpset_pickup_all_possible(BPLONG_PTR sp_dv_ptr,BPLONG_PTR sa_dv_ptr);
extern BPLONG clpset_pickup_only_in(BPLONG_PTR sp_dv_ptr,BPLONG_PTR sa_dv_ptr,BPLONG card);
extern void clpset_exclude_all_possible(BPLONG_PTR sp_dv_ptr,BPLONG cur,BPLONG last);
extern int b_REIFY_EQ_CONSTR_ACTION(BPLONG B,BPLONG X,BPLONG Y);
extern int b_REIFY_GE_CONSTR_ACTION(BPLONG B,BPLONG X,BPLONG Y);
extern int b_REIFY_NEQ_CONSTR_ACTION(BPLONG B,BPLONG X,BPLONG Y);
extern int b_ABS_CON_cc(BPLONG X,BPLONG Y);
extern int b_FD_ABS_X_TO_Y(BPLONG X,BPLONG Y);
extern int b_MOD_CON_ccc(BPLONG X,BPLONG Y,BPLONG Z);
extern int b_IDIV_CON_ccc(BPLONG X,BPLONG Y,BPLONG Z);
extern int b_ABS_DIFF_CON_ccc(BPLONG X,BPLONG Y,BPLONG);
extern int b_ABS_DIFF_X_TO_Y(BPLONG Ex);
extern int b_ABS_ABS_DIFF_NEQ(void);
extern int b_COMP_PROP_FAPP1(void);
extern int b_COMP_PROP_FAPP2(void);

//
//	clpfd_libs.c prototypes
//
extern int nary_interval_consistent_eq(BPLONG n);
extern int nary_interval_consistent_nocoe(BPLONG n);
extern int nary_interval_consistent_ge(BPLONG n);
extern int varorint_domain_region(BPLONG X,BPLONG from,BPLONG to);
extern int domain_region_noint(BPLONG_PTR dv_ptr,BPLONG from,BPLONG to);
extern int domain_region_aux(BPLONG_PTR dv_ptr,BPLONG from,BPLONG to);
extern void print_linear_constr(BPLONG n);
extern int b_CONSTR_COES_TYPE(BPLONG n);
extern int b_IS_BOOLEAN_VAR_CONSTR(BPLONG coes_type);

//
//	cpreds.c prototypes
//
extern int bp_signal(int signo, void user_signal_handler(int, void *), void *userdata);
extern void *bp_get_address(BPLONG t);
extern BPLONG bp_build_address(void *address);
extern int identical_VAR_VAR(BPLONG t1,BPLONG t2);
extern BPLONG bp_get_call_arg(BPLONG i,BPLONG arity);
extern int bp_is_var(BPLONG t);
extern int is_VAR(BPLONG t);
extern int bp_is_atom(BPLONG t);
extern int is_ATOM(BPLONG t);
extern int bp_is_integer(BPLONG t);
extern int is_INTEGER(BPLONG t);
extern int bp_is_float(BPLONG t);
extern int is_FLOAT(BPLONG t);
extern int bp_is_nil(BPLONG t);
extern int is_NIL(BPLONG t);
extern int bp_is_list(BPLONG t);
extern int is_LIST(BPLONG t);
extern int bp_is_structure(BPLONG t);
extern int is_STRUCTURE(BPLONG t);
extern int bp_is_compound(BPLONG t);
extern int is_COMPOUND(BPLONG t);
extern int bp_is_unifiable(BPLONG t1,BPLONG t2);
extern int bp_is_identical(BPLONG t1,BPLONG t2);
extern BPLONG bp_get_integer(BPLONG t);
extern BPLONG get_INTEGER(BPLONG t);
extern double bp_get_float(BPLONG t);
extern double get_FLOAT(BPLONG t);
extern char *bp_get_name(BPLONG t);
extern char *get_NAME(BPLONG t);
extern int bp_get_arity(BPLONG t);
extern int get_ARITY(BPLONG t);
extern int bp_unify(BPLONG t1,BPLONG t2);
extern BPLONG bp_get_arg(BPLONG i,BPLONG t);
extern BPLONG get_ARG(BPLONG i,BPLONG t);
extern BPLONG bp_get_car(BPLONG t);
extern BPLONG bp_get_cdr(BPLONG t);
extern BPLONG bp_build_var(void);
extern BPLONG build_VAR(void);
extern BPLONG bp_build_integer(BPLONG i);
extern BPLONG build_INTEGER(BPLONG i);
extern BPLONG bp_build_float(double f);
extern BPLONG build_FLOAT(double f);
extern BPLONG bp_build_atom(const char *name) ;
extern BPLONG build_ATOM(const char *name);
extern BPLONG bp_build_list(void);
extern BPLONG bp_build_nil(void);
extern BPLONG build_NIL(void);
extern BPLONG bp_build_structure(char *name,BPLONG arity);
extern int atom_2_term(void);
extern int string_2_term(void);
extern int term_2_atom(void);
extern int term_2_string(void);
extern int bp_string_2_term(char *str, BPLONG term, BPLONG vars);
extern int bp_call_term(BPLONG term);
extern int bp_call_term_once(BPLONG term);
extern int bp_call_term_catch(BPLONG term);
extern int c_set_bp_exception(void);
extern int bp_call_string(char *cmd);
extern int bp_mount_query_string(char *cmd);
extern int bp_mount_query_term(BPLONG term);
extern int bp_next_solution(void);
extern void initialize_bp_str_bag(void);
extern void expand_bp_str_bag(void);
extern void append_str_to_solution_bag(char *str,BPLONG len,BPLONG check_quote);
extern char *bp_term_2_string(BPLONG term);
extern void aux_term_2_string_term(BPLONG term);
extern void aux_term_2_string_list(BPLONG term);
extern void bp_write(BPLONG term);
extern int currentTime(void);
extern int test(void);
extern int c_CallBpFromC(void);
extern int c_START_TRACE(void);
extern int c_END_TRACE(void);
extern void Cboot(void);
extern void Cboot_TP(void);

//
//	delay.c prototypes
//
extern BPLONG build_delayed_call_on_the_heap(BPLONG_PTR frame);
extern int c_frozen_cf(void);
extern BPLONG_PTR frozen_cs(BPLONG_PTR cs,BPLONG_PTR Plist);
extern int c_frozen_f(void);
extern int b_SUSP_ATTACH_TERM_cc(BPLONG Var,BPLONG Term);
extern int b_SUSP_ATTACHED_TERM_cf(BPLONG Var,BPLONG Term);
extern int b_SUSP_VAR_c(BPLONG var);
extern BPLONG next_alive_susp_call(BPLONG cs_list,BPLONG_PTR breg);
extern void print_cs(BPLONG cs_list);
extern void Cboot_delay(void);

//
//	dis.c prototypes
//
extern void symbol_table_statistics(int *num_of_empty_buckets, int *len_of_longest_chain);
extern void dis(void);
extern void dis_data(void);
extern void dis_text(void);
extern void print_inst(FILE *filedes);
extern void dis_addr(FILE *filedes,BPLONG operand);
extern void dis_y(FILE *filedes,BPLONG operand);
extern void dis_constant(FILE *filedes,BPLONG operand);
extern void dis_literal(FILE *filedes,BPLONG operand);
extern void dis_z(FILE *filedes,BPLONG operand);
extern void dis_struct(FILE *filedes,BPLONG operand);
extern void dis_ys(FILE *filedes,BPLONG n);
extern void dis_zs(FILE *filedes,BPLONG n);

//
//	domain.c prototypes
//
extern int c_DM_CREATE_BV_DVAR(void);
extern int c_DM_CREATE_DVAR(void);
extern int b_DM_CREATE_DVAR(BPLONG Var,BPLONG From,BPLONG To);
extern int c_DM_CREATE_DVARS(void);
extern int aux_create_domain_var(BPLONG Var,BPLONG from,BPLONG to);
extern int aux_create_bv_domain_var(BPLONG Var,BPLONG from,BPLONG to);
extern BPLONG_PTR dm_clone(BPLONG_PTR dv_ptr);
extern int c_create_susp_var(void);
extern int create_susp_var(BPLONG Var);
extern int fd_vector_min_max(void);
extern void it_to_bv(BPLONG_PTR dv_ptr);
extern void it_to_bv_with_hole(BPLONG_PTR dv_ptr,BPLONG elm);
extern void it_to_bv_with_interval_holes(BPLONG_PTR dv_ptr,BPLONG interval_start,BPLONG interval_end);
extern int b_DM_DISJOINT_cc(BPLONG Var1,BPLONG Var2);
extern int c_DM_INCLUDE(void);
extern int b_DM_INCLUDE(BPLONG Var1,BPLONG Var2);
extern int domain_include_bv(BPLONG_PTR dv_ptr1,BPLONG_PTR dv_ptr2);
extern int varorint_set_false(BPLONG X,BPLONG elm);
extern void domain_set_false_noint(BPLONG_PTR dv_ptr,BPLONG elm);
extern int domain_set_false_aux(BPLONG_PTR dv_ptr,BPLONG elm);
extern int domain_set_false_bv(BPLONG_PTR dv_ptr,BPLONG elm);
extern void domain_exclude_first_bv(BPLONG_PTR dv_ptr);
extern void domain_exclude_last_bv(BPLONG_PTR dv_ptr);
extern void exclude_inner_elm_bv(BPLONG_PTR dv_ptr,BPLONG elm);
extern int exclude_dom_aux(BPLONG_PTR dv_ptr_x,BPLONG_PTR dv_ptr_y);
extern int exclude_dom_comp_aux(BPLONG_PTR dv_ptr_x,BPLONG_PTR dv_ptr_y);
extern int b_DM_NEXT_ccf(BPLONG DVar,BPLONG E,BPLONG NextE);
extern BPLONG domain_next_bv(BPLONG_PTR dv_ptr,BPLONG elm);
extern int b_DM_PREV_ccf(BPLONG DVar,BPLONG E,BPLONG PrevE);
extern BPLONG domain_prev_bv(BPLONG_PTR dv_ptr,BPLONG elm);
extern int b_DM_TRUE_cc(BPLONG Var,BPLONG E);
extern int b_DM_FALSE_cc(BPLONG Var,BPLONG E);
extern int b_DM_INNER_TRUE_cc(BPLONG Var,BPLONG E);
extern int varorint_dm_true(BPLONG X,BPLONG elm);
extern int dm_true(BPLONG_PTR dv_ptr,BPLONG elm);
extern int dm_true_bv(BPLONG_PTR dv_ptr,BPLONG elm);
extern int dm_true_bv_nbt(BPLONG_PTR dv_ptr,BPLONG elm);
extern int b_DM_MIN_MAX_cff(BPLONG Var,BPLONG Min,BPLONG Max);
extern int b_DM_MIN_cf(BPLONG Var,BPLONG Min);
extern int b_DM_MAX_cf(BPLONG Var,BPLONG Max);
extern int b_DM_COUNT_cf(BPLONG Var,BPLONG Count);
extern int count_domain_elms(BPLONG_PTR dv_ptr,BPLONG from,BPLONG to);
extern int exclude_domain_elms(BPLONG_PTR dv_ptr,BPLONG from,BPLONG to);
extern int c_reachability_test(void);
extern int c_path_from_to_reachability_test(void);
extern int path_from_to_reachable(BPLONG from,BPLONG to,BPLONG_PTR var_vector,BPLONG_PTR visited);
extern int b_PATH_FROM_TO_REACHABLE(void);
extern int check_reach_cuts_in_one_connection(BPLONG n,BPLONG Start,BPLONG End,BPLONG Lab,BPLONG TaggedLab,
        BPLONG_PTR LabVarArray,BPLONG_PTR SuccVarArray,BPLONG_PTR VisitedArray);
extern int path_from_to_node_degree(BPLONG n,BPLONG NodeNum,BPLONG Lab,BPLONG TaggedLab,
                                    BPLONG_PTR LabVarArray,BPLONG_PTR SuccVarArray);
extern void check_reach_withno_cut(BPLONG Start,BPLONG Lab,BPLONG TaggedLab,
                                   BPLONG_PTR LabVarArray,BPLONG_PTR SuccVarArray,BPLONG_PTR VisitedArray);
extern void check_reach_with_one_cut(BPLONG Start,BPLONG Lab,BPLONG TaggedLab,BPLONG_PTR LabVarArray,
                                     BPLONG_PTR SuccVarArray,BPLONG_PTR VisitedArray,BPLONG cutNodeNum);
extern void check_two_cuts_in_two_connections(BPLONG n,BPLONG firstCon,BPLONG nConnections,BPLONG_PTR ConStartArray,
        BPLONG_PTR ConEndArray,BPLONG_PTR ConLabArray,BPLONG_PTR ConFlagArray,
        BPLONG_PTR LabVarArray,BPLONG_PTR SuccVarArray,BPLONG_PTR VisitedArray);
extern void check_cuts_in_another_connection(BPLONG v1,BPLONG v2,BPLONG n,BPLONG firstCon,BPLONG firstLab,
        BPLONG nConnections,BPLONG_PTR ConStartArray,BPLONG_PTR ConEndArray,
        BPLONG_PTR ConLabArray,BPLONG_PTR LabVarArray,
        BPLONG_PTR SuccVarArray,BPLONG_PTR VisitedArray);
extern void check_reach_with_two_cuts(BPLONG Start,BPLONG Lab,BPLONG TaggedLab,
                                      BPLONG_PTR LabVarArray,BPLONG_PTR SuccVarArray,BPLONG_PTR VisitedArray,
                                      BPLONG cutNodeNum1,BPLONG cutNodeNum2);
extern void domain_reduce_to_two(BPLONG_PTR dv_ptr,BPLONG val1,BPLONG val2);
extern int b_ALLDISTINCT_CHECK_HALL_VAR_cccc(BPLONG X,BPLONG NumElmsLeft,BPLONG L,BPLONG R);
extern void print_domain_var(BPLONG op);
extern void print_domain(BPLONG_PTR dv_ptr);
extern int b_DM_INTERSECT(BPLONG X,BPLONG Y);
extern int dm_intersect(BPLONG_PTR dv_ptr_x,BPLONG_PTR dv_ptr_y);
extern int b_DM_INTERSECT2(BPLONG X,BPLONG Low,BPLONG Up);
extern int dm_disjoint(BPLONG_PTR dv_ptr1,BPLONG_PTR dv_ptr2);
extern int c_var_notin_ints(void);
extern int b_VAR_NOTIN_D_cc(BPLONG X,BPLONG List);
extern int check_var_notin_d(BPLONG x,BPLONG List);
extern int b_TASKS_EXCLUDE_NOGOOD_INTERVAL_ccc(BPLONG start,BPLONG duration,BPLONG tasks);
extern int b_DISJUNCTIVE_TASKS_AC(BPLONG n);
extern int b_DISJUNCTIVE_TASKS_EF(BPLONG n);
extern int b_EXCLUDE_NOGOOD_INTERVAL_ccc(BPLONG Var,BPLONG Low,BPLONG Up);
extern int c_integers_intervals_list(void);
extern int b_VAR_IN_D_cc(BPLONG X,BPLONG List);
extern int check_var_in_d(BPLONG x,BPLONG List);
extern void exclude_inner_interval_bv(BPLONG_PTR dv_ptr,BPLONG from,BPLONG to);
extern int domain_exclude_interval_aux(BPLONG_PTR dv_ptr,BPLONG from,BPLONG to);
extern int domain_exclude_interval_noint(BPLONG_PTR dv_ptr,BPLONG from,BPLONG to);
extern int b_CLPFD_MULTIPLY_INT_ccc(BPLONG X,BPLONG Y,BPLONG Z);
extern BPLONG con05_hashtable_get(BPLONG table,BPLONG key);
extern int c_cpcon_decrement_counters(void);
extern int cpcon_decrement_counter(BPLONG Counters,BPLONG J,BPLONG elm,BPLONG_PTR dv_ptr);
extern void Cboot_domain(void);

//
//	debug.c prototypes
//
extern int c_init_dg_flag(void);
extern int c_set_dg_flag(void);
extern int c_get_dg_flag(void);
extern int c_remove_spy_point(void);
extern int c_is_spy_point(void);
extern int c_add_spy_point(void);
extern int c_get_spy_points(void);
extern int c_remove_spy_points(void);
extern int c_is_debug_mode(void);
extern int b_IS_DEBUG_MODE(void);
extern int c_is_spy_mode(void);
extern void set_global_call_number(BPLONG CallNo);
extern int c_set_global_call_number(void);
extern int c_next_global_call_number(void);
extern int c_set_skip_call_no(void);
extern int c_is_skip_call_no(void);
extern int c_backtrace(void);
extern int c_get_dg_print_depth(void);
extern int c_set_dg_print_depth(void);
extern int c_get_dg_choice_point(void);
extern int c_dg_in_undo_mode(void);
extern void Cboot_debug(void);

//
//	event.c prototypes
//
extern int c_post_event(void);
extern int b_POST_EVENT_ccc(BPLONG x,BPLONG no,BPLONG e);
extern void cg_initialize(void);
extern int c_cg_is_component(void);
extern int cg_is_component(BPLONG op);
extern void cg_init_sym(void); /* called init_sym */
extern int is_correct_event_source(BPLONG event_no,BPLONG source);
extern BPLONG event_handler_type(BPLONG_PTR frame);
extern BPLONG cg_get_component_event_handler(BPLONG comp);
extern BPLONG cg_lookup_event_handler(BPLONG type,BPLONG comp_no);
extern BPLONG cg_lookup_component(BPLONG type,BPLONG comp_no);
extern BPLONG register_event_source(BPLONG event_no,BPLONG source);
extern void attach_event_source(BPLONG list,BPLONG source,BPLONG source_no,BPLONG event_no);
extern int cg_get_default_window(void);
extern int c_cg_print_component(void);
extern void cg_print_component(BPLONG op);
extern BPLONG cg_get_component_no(BPLONG comp);
extern int c_timer(void);
extern int c_kill_timer();
extern int c_sleep(void);
extern void add_to_event_pool(BPLONG no,BPLONG type,void *event_object);
extern void post_event_pool(void);
extern void post_cg_event(BPLONG handler,BPLONG type,void *eo);
extern int c_global_set_bpp(void);
extern int c_global_get_bpp(void);
extern int b_GET_ATTACHED_AGENTS_ccf(BPLONG var,BPLONG port,BPLONG lists);
extern int b_AGENT_OCCUR_IN_CONJUNCTIVE_CHANNELS_cc(BPLONG frame_offset,BPLONG lists);
extern int b_AGENT_OCCUR_IN_DISJUNCTIVE_CHANNELS_cc(BPLONG frame_offset,BPLONG lists);
extern int c_start_critical_region(void);
extern int c_end_critical_region(void);
extern int c_in_critical_region(void);

//
//	expand.c prototypes
//
extern BPLONG_PTR expand_trail(BPLONG_PTR trail_top,BPLONG_PTR breg);
extern void my_memcpy_top_down(BPLONG_PTR des,BPLONG_PTR src,BPLONG size);
extern void my_memcpy_btm_up(BPLONG_PTR des,BPLONG_PTR src,BPLONG size);
extern int expand_local_global_stacks(BPLONG preferred_size);
extern void expand_initialize_frames(BPLONG maxs);
extern void expand_initialize_ar_chain(BPLONG_PTR f,BPLONG maxs);
extern void initialize_frame(BPLONG_PTR f,BPLONG maxs);
extern int expandStackNullifyUntaggedCells(BPLONG diff_s);
extern void expandStackNullifyUntaggedCellsBFrames(BPLONG_PTR breg,BPLONG diff_s);
extern void expandStackNullifyUntaggedCellsSfFrames(BPLONG_PTR sf,BPLONG diff_s);
extern void expandStackNullifyUntaggedCellsArFrames(BPLONG_PTR ar,BPLONG diff_s);
extern void expandStackNullifyUntaggedCellsFrame(BPLONG_PTR ar,BPLONG diff_s);
extern void expandStackNullifyUntaggedCellsFrameSlots(BPLONG_PTR f,BPLONG nslots,BPLONG diff_s);
extern void expandStackNullifyUntaggedCellsTrail(void);
extern void expandStackNullifyUntaggedCellsTerm(BPLONG term);
extern void expandStackResetPointers(BPLONG diff_h,BPLONG diff_s);
extern BPLONG_PTR expandStackResetAddr(BPLONG_PTR addr,BPLONG diff_h,BPLONG diff_s);
extern void expandStackResetHeap(BPLONG diff_h);
extern void expandStackResetStack(BPLONG diff_h,BPLONG diff_s);
extern void expandStackResetTrail(BPLONG diff_h,BPLONG diff_s);

//
//	file.c prototypes
//
extern int b_STREAM_IS_OPEN_c(BPLONG Index);
extern int b_STREAM_SET_TYPE_cc(BPLONG Index,BPLONG Type);
extern int b_STREAM_SET_EOF_ACTION_cc(BPLONG Index,BPLONG Action);
extern int b_STREAM_ADD_ALIAS_cc(BPLONG Index,BPLONG Atom);
extern int b_STREAM_UPDATE_EOS(void);
extern int b_STREAM_GET_FILE_NAME_cf(BPLONG Index,BPLONG Name);
extern int b_STREAM_GET_MODE_cf(BPLONG Index,BPLONG Mode);
extern int b_STREAM_GET_ALIASES_cf(BPLONG Index,BPLONG Aliases);
extern int b_STREAM_GET_EOS_cf(BPLONG Index,BPLONG Eos);
extern int b_STREAM_GET_EOF_ACTION_cf(BPLONG Index,BPLONG EofAction);
extern int b_STREAM_GET_TYPE_cf(BPLONG Index,BPLONG Type);
extern int b_STREAM_CHECK_CURRENT_TEXT_INPUT(void);
extern int b_STREAM_CHECK_CURRENT_TEXT_OUTPUT(void);
extern void bp_trim_trailing_zeros(char *bp_buf);
extern void bp_write_insert_dot_zero_if_needed(char *bp_buf);
extern void bp_write_double(BPLONG op);
extern int bp_write_double_update_pos(BPLONG op);
extern int bp_write_bigint_bp_buf(BPLONG op);
extern int bp_write_bigint(BPLONG op);
extern int bp_write_bigint_update_pos(BPLONG op);
extern int get_file_name(BPLONG op);
extern void bp_write_pname(CHAR_PTR name_ptr);
extern int bp_write_pname_update_pos(CHAR_PTR name_ptr,BPLONG length);
extern int graphic_char(char ch);
extern int single_quote_needed(CHAR_PTR name_ptr,BPLONG length);
extern int strcpy_add_quote(CHAR_PTR buf,CHAR_PTR str,BPLONG len);
extern void bp_write_qname_to_bp_buf(CHAR_PTR name_ptr,UW16 length);
extern void bp_write_qname(CHAR_PTR name_ptr,BPLONG length);
extern int bp_write_qname_update_pos(CHAR_PTR name_ptr,BPLONG length);
extern int bp_write_var_update_pos(BPLONG op);
extern int bp_write_domain_update_pos(BPLONG_PTR dv_ptr);
extern int bp_write_suspvar_update_pos(BPLONG op);
extern int bp_write_int_update_pos(BPLONG op);
extern int bp_write_char_update_pos(char c);
extern int get_file_index(BPLONG cword, BPLONG mode);
extern int next_file_index(void);
extern void release_file_index(int i);
extern int b_ATOM_MODE_cf(BPLONG op1,BPLONG op2);
extern int b_NORMAL_ATOM_c(BPLONG op);
extern int b_WRITENAME_c(BPLONG op);
extern int c_format_set_dest(void);
extern int c_format_get_line_pos(void);
extern int c_format_retrieve_codes(void);
extern int c_fmt_writename(void);
extern int b_WRITE_QUICK_c(BPLONG op);
extern int c_fmt_write_quick(void);
extern int b_WRITEQNAME_c(BPLONG op);
extern int c_fmt_writeqname(void);
extern int b_WRITEQ_QUICK_c(BPLONG op);
extern int c_fmt_writeq_quick(void);
extern int c_fmt_nl(void);
extern int b_NL(void);
extern int b_PUT_BYTE_c(BPLONG op);
extern int b_PUT_CODE_c(BPLONG op);
extern int b_PUT_c(BPLONG op);
extern int c_put_update_pos(void);
extern int b_TAB_c(BPLONG op);
extern int b_GET_BYTE_f(BPLONG op);
extern int b_GET_CODE_f(BPLONG op);
extern int b_GET0_f(BPLONG op);
extern int b_PEEK_BYTE_f(BPLONG op);
extern int b_PEEK_CODE_f(BPLONG op);
extern int c_UNGETC(void);
extern int b_GET_f(BPLONG op);
extern int c_rm_file(void);
extern int c_cp_file(void);
extern int b_SEE_c(BPLONG fop);
extern int b_SEEN(void);
extern int b_SEEING_f(BPLONG temp);
extern int b_TELL_cc(BPLONG fop,BPLONG mode);
extern int b_TOLD(void);
extern int b_TELLING_f(BPLONG temp); /* arg1: unified with the current out put file name */
extern int b_OPEN_ccf(BPLONG fop,BPLONG sop,BPLONG Index);
extern int get_socket_fd(int index);
extern int allocate_socket_file(FILE *file, char *name);
extern int b_CLOSE_c(BPLONG Index);
extern void file_init(void);
extern int b_ACCESS_ccf(BPLONG op1,BPLONG op2,BPLONG op3);
extern int socket_file_name(const char* filename);
extern int file_stat(void);
extern int c_file_permission(void);
extern int c_file_type(void);
#ifndef CYGWIN
extern int c_directory_list(void);
#endif
extern int c_mkdir(void);
extern int c_rmdir(void);
extern int c_rename(void);
extern int c_chdir(void);
extern int c_get_cwd(void);
extern int c_write_term(void);
extern void write_term1(BPLONG op,FILE *fp);
extern void write_list(BPLONG op);
extern int write_term(BPLONG op);
extern int b_WRITE_IMAGE_c(BPLONG op);
extern int write_image(BPLONG op);
extern int b_CURRENT_INPUT_f(BPLONG Index);
extern int b_CURRENT_OUTPUT_f(BPLONG Index);
extern int b_SET_BINARY_INPUT_cc(BPLONG Index,BPLONG Source);
extern int b_SET_TEXT_INPUT_cc(BPLONG Index,BPLONG Source);
extern int b_SET_INPUT_cc(BPLONG Index,BPLONG Source);
extern int cc_set_input(BPLONG temp_in_file_i);
extern int b_SET_BINARY_OUTPUT_cc(BPLONG Index,BPLONG Source);
extern int b_SET_TEXT_OUTPUT_cc(BPLONG Index,BPLONG Source);
extern int b_SET_OUTPUT_cc(BPLONG Index,BPLONG Source);
extern int cc_set_output(BPLONG temp_out_file_i);
extern int b_FLUSH_OUTPUT(void);
extern int b_ASPN_i(BPLONG op1);
extern int c_ASPN_i(void);
extern int b_ASPN_c(BPLONG op1);
extern int b_ASPN2_cc(BPLONG op1,BPLONG op2);
extern int b_ASPN3_ccc(BPLONG op1,BPLONG op2,BPLONG op3);
extern int b_ASPN4_cccc(BPLONG op1,BPLONG op2,BPLONG op3,BPLONG op4);
extern int aspn(void);
extern int b_GET_LINE_NO_cf(BPLONG Index, BPLONG no);
extern int b_GET_LINE_POS_cf(BPLONG Index,BPLONG pos);
extern int c_FORMAT_PRINT_INTEGER(void);
extern int c_FORMAT_PRINT_FLOAT(void);
extern char *format_comma_separated_int(BPLONG amt);
extern int b_NAME0_cf(BPLONG op1,BPLONG op2); /* op2 is made to be the string of the name of op1*/

//
//	float1.c prototypes
//
extern double floatval(BPLONG op);
extern BPLONG encodefloat1(double op);

//
//	gcheap.c prototypes
//
extern int gcHeap(void);
extern int eliminateDuplicatedTrailInTopSegment(void);
extern void eliminateDuplicatedTrailInTopSegmentSquare(void);
extern int eliminateDuplicatedTrailInTopSegmentLinear(void);
extern void packEntireTrail(void);
extern int alreadyTrailed(BPLONG_PTR addr,BPLONG_PTR from_ptr,BPLONG_PTR to_ptr);
extern void copyHeapBack(void);
extern void gcRescueFreeVars(void);
extern void gcRescueBFrame(BPLONG_PTR ar);
extern void gcRescueArFrames(BPLONG_PTR ar);
extern void gcRescueSfFrames(BPLONG_PTR sf);
extern void gcRescueFrame(BPLONG_PTR f,BPLONG noReservedSlots);
extern void gcRescueTrail(void);
extern void gcRescueFreeVar(BPLONG_PTR addr,BPLONG term);
extern void gcRescueTriggeredCs(void);
extern BPLONG gcRescueBitVector(BPLONG_PTR bv_ptr);
extern void gcRescueTerm(BPLONG_PTR addr,BPLONG term);

//
//	gcqueue.c prototypes
//
extern void gcQueueConstruct(void);
extern void gcQueueExpand(void);
extern BPLONG_PTR gcQueueCopy(void);
extern void gcInitDynamicArray(void);
extern void gcExpandDynamicArray(void);
extern void gcDisposeDynamicArray(void);
extern int allocateMaskArea(BPLONG size);
extern void gcSetMask(BPLONG_PTR addr,BPLONG size,BPLONG_PTR base);
extern int gcIsMarked(BPLONG_PTR addr,BPLONG_PTR base);

//
//	gcstack.c prototypes
//
extern int set_max_heap_mark(void);
extern int number_of_gcs(void);
extern int garbage_collector(void);
extern void gc_initialize_ar_chain(void);
extern int gc_globalize_sf_chain(void);
extern void mark_stack_references_ar_chain(void);
extern void mark_stack_references_sf_chain(void);
extern void mark_stack_references_frame(BPLONG_PTR f);
extern void globalize_stack_vars_in_frame(BPLONG_PTR f,BPLONG size);
extern int allocateCopyArea(BPLONG size);
extern int gcStack(void);
extern void copyStackBack(void);
extern void gcMoveFrameSlotOut(BPLONG term,BPLONG_PTR des);
extern BPLONG_PTR gcMoveFrameOut(BPLONG_PTR src_f,BPLONG nReservedSlots);
extern BPLONG_PTR gcMoveAliveFramesOutSf(void);
extern BPLONG_PTR gcReverseArChain(BPLONG_PTR f,BPLONG_PTR prev);
extern BPLONG_PTR gcMoveAliveFramesOutReversedAr(BPLONG_PTR prev,BPLONG_PTR f);
extern BPLONG_PTR gcMoveAliveFramesOutAr(void);
extern void gcStackMarkSuspVars(void);
extern void gcTrailMarkSuspVars(void);
extern void gcMarkSuspVarsTerm(BPLONG term);
extern void gcMarkSuspVarsCs(BPLONG cs);
extern void gcResetSuspVars(void);
extern void gcResetTriggeredCs(void);
extern void gcResetCs(BPLONG_PTR addr,BPLONG list);
extern int b_ENABLE_GC(void);
extern int b_DISABLE_GC(void);
extern int c_GET_GC_TIME(void);

//
//	getline.c prototypes
//

#if MSDOS || __EMX__ || __GO32__ || WIN32
extern int pc_keymap(int c);
#endif
extern int c_GET_NONEMPTY_LINE(void);
extern int getline_is_spaces(char *s);
extern int c_FINISH_GET_LINE(void);

//
//	global.c prototypes
//
extern int b_IS_CONSULTED_c(BPLONG goal);
extern int b_SET_DYNAMIC_cc(BPLONG name,BPLONG arity);
extern int b_GET_SYM_TYPE_ccf(BPLONG name,BPLONG arity,BPLONG type);
extern int b_IS_DYNAMIC_cc(BPLONG name,BPLONG arity);
extern int b_IS_ORDINARY_cc(BPLONG name,BPLONG arity);
extern int b_GLOBAL_SET_cccc(BPLONG name,BPLONG arity,BPLONG value,BPLONG part);
extern int b_GLOBAL_SET_ccc(BPLONG name,BPLONG arity,BPLONG value);
extern int c_OLD_GLOBAL_SET(void);
extern int c_OLD_GLOBAL_GET(void);
extern int b_GLOBAL_GET_ccf(BPLONG name,BPLONG arity,BPLONG value);
extern int b_ISGLOBAL_cc(BPLONG name,BPLONG arity);
extern int b_GLOBAL_DEL_cc(BPLONG name,BPLONG arity);
extern int c_SET_ARG_PAREA(void);
extern int b_GLOBAL_INSERT_HEAD_cccc(BPLONG name,BPLONG arity,
                                     BPLONG value,BPLONG part);
extern int b_GLOBAL_INSERT_TAIL_ccc(BPLONG name,BPLONG arity,BPLONG value);
extern void assert_tail(SYM_REC_PTR sym_ptr,BPLONG term);
extern BPLONG make_cons_in_parea(BPLONG car, BPLONG cdr);
extern BPLONG copy_term_heap_to_parea(BPLONG value);
extern BPLONG copy_term_heap_to_parea_with_varno(BPLONG value,BPLONG *varno);
extern int c_UNNUMBER_VARS(void);

//
//	init.c prototypes
//
extern void init_toam(int argc, char **argv);
extern void init_stack(BPLONG bsize);
extern int init_loading(int argc, char **argv);
extern int load_bp_out(void);
extern int is_bc_file(CHAR_PTR main_arg);
extern int load_user_bc_file(char *name);
extern void add_main_arg(CHAR_PTR main_arg);
extern int c_init_global_each_session(void);

//
//	init_sym.c prototypes
//
extern void init_sym(void);
extern void init_error_sym(void);

//
//	loader.c prototypes
//
extern int loader(CHAR_PTR file,BPLONG file_type,BPLONG load_damon);
extern int load_syms(BPLONG file_type);
extern int load_text(void);
extern int load_hashtab(void);
extern int get_index_tab(BPLONG clause_no,BPLONG_PTR lenptr);
extern BPLONG_PTR gen_index(BPLONG hash_inst_addr,BPLONG clause_no,BPLONG alt);
extern BPLONG bp_hsize(BPLONG numentry);
extern int dyn_loader(SYM_REC_PTR sym_ptr, BPLONG file_type, BPLONG load_damon);
extern SYM_REC_PTR insert(const char *name, BPLONG length, BPLONG arity);
extern SYM_REC_PTR insert_cpred(CHAR_PTR name,BPLONG arity,int (*func)(void));
extern int set_temp_ep(SYM_REC_PTR sym_ptr, BPLONG ep, BPLONG file_type);
extern void set_real_ep(SYM_REC_PTR sym_ptr, CHAR_PTR base);
extern void namestring(SYM_REC_PTR sym_ptr, CHAR_PTR s);
extern int c_CURRENT_PREDICATE(void);
extern int c_CURRENT_PREDICATES(void);
extern SYM_REC_PTR look_for_sym_with_entrance(BPLONG_PTR p);

//
//	mic.c prototypes
//
extern int check_file_term(BPLONG term);
extern int c_ref_equal(void);
extern int c_NEXT_PRIME(void);
extern int c_set_debugging_susp(void);
extern int c_unset_debugging_susp(void);
extern int c_UNDERSCORE_NAME(void);
extern BPLONG time_t_2_prolog(time_t *t);
extern int c_confirm_copy_right(void);
extern BPLONG cputime(void);
extern int c_TIME_ffffff(void);
extern int c_GETENV_cf(void);
extern int b_COMPARE_fcc(BPLONG op1,BPLONG op2,BPLONG op3);
extern int bp_compare(BPLONG val1, BPLONG val2);
extern int compare_float_unknown(BPLONG val1,BPLONG val2);
extern int comalpha(SYM_REC_PTR psc_ptr1,SYM_REC_PTR  psc_ptr2);
extern int c_INCREMENTARG(void);
extern int c_DECREMENTARG(void);
extern int b_DESTRUCTIVE_SET_ARG_ccc(BPLONG op1,BPLONG op2,BPLONG op3);
extern int b_CHAR_CODE_cf(BPLONG ch,BPLONG code); /* the code of op1 is op2 */
extern int b_CHAR_CODE_fc(BPLONG ch,BPLONG code); /* the code of op1 is op2 */
extern int string2codes(char *str,BPLONG list);
extern int var_or_atomic(BPLONG op);
extern int b_BLDATOM_fc(BPLONG op1,BPLONG op2);
extern int b_BLDNUM_fc(BPLONG op1,BPLONG op2);
extern int b_SYSTEM0_cf(BPLONG op1,BPLONG op2); /* op1: a list of int (string) for CShell commands */
extern int c_LOAD_cfc(void);
extern int b_LOAD_cfc(BPLONG op1,BPLONG op2,BPLONG op3);
extern BPLONG compute_total_parea_size(void);
extern int c_STATISTICS(void);
extern int c_STATISTICS0(void);
extern int b_HASHVAL_cf(BPLONG op1,BPLONG op2); /* op1 a term, op2 the hash value of op1 */
extern BPLONG bp_hashval(BPLONG op);
extern BPLONG bp_hashval_list(BPLONG term);
extern int b_HASHVAL1_cf(BPLONG op1,BPLONG op2); /* op1 a term,
								      op2 the hash value of the main functor of op1 */
extern BPLONG hashval1(BPLONG op);
extern int b_HASHTABLE_GET_ccf(BPLONG table,BPLONG key,BPLONG value);
extern int hashtable_contains_key(BPLONG table,BPLONG key);
extern int bp_is_hashtable(BPLONG term);
extern BPLONG bp_hashtable_get(BPLONG table,BPLONG key);
extern BPLONG hashtable_lookup_chain(BPLONG chain,BPLONG key);
extern BPLONG make_struct1(char *f,BPLONG op1);
extern BPLONG make_struct2(char *f,BPLONG op1,BPLONG op2);
extern BPLONG make_struct3(char *f,BPLONG op1,BPLONG op2,BPLONG op3);
extern BPLONG make_struct4(char *f,BPLONG op1,BPLONG op2,BPLONG op3,BPLONG op4);
extern BPLONG make_struct5(char *f,BPLONG op1,BPLONG op2,BPLONG op3,BPLONG op4,BPLONG op5);
extern BPLONG make_struct_dummy(SYM_REC_PTR psc_ptr);
extern BPLONG make_struct_holders(SYM_REC_PTR psc_ptr,BPLONG init_val);
extern BPLONG make_cons_dummy(void);
extern BPLONG make_struct_with_args(BPLONG_PTR fp,SYM_REC_PTR psc_ptr);
extern int b_GET_LENGTH_cf(BPLONG op1,BPLONG op2);
extern int c_SHOW_NONDET_FRAME(void);
extern int b_CPUTIME_f(BPLONG op1);
extern int membchk(void);
extern int membchk2(BPLONG x,BPLONG list);
extern void quit(CHAR_PTR s);
extern void scat(CHAR_PTR s1, CHAR_PTR s2, CHAR_PTR s3);
extern int c_functor(void);
extern int cfunctor1(BPLONG op1,BPLONG op2,BPLONG op3);
extern int c_arg(void);
extern int b_GEN_ARG_ccf(BPLONG Index,BPLONG Comp,BPLONG Arg);
extern int carg1(BPLONG op1,BPLONG op2,BPLONG op3);
extern int c_get_main_args(void);
extern int c_SAVE_AR(void);
#ifdef GC
extern int c_set_gc_threshold(void);
#endif
extern int b_NTH_ELM_ccf(BPLONG i,BPLONG l,BPLONG v);
extern void myquit(BPLONG overflow_type,char *src);
extern int c_HTABLE_HCODE(void);
extern int htable_contains_tuple(BPLONG_PTR htable_ptr,BPLONG n,BPLONG_PTR tuple_ptr,BPLONG arity);
extern int b_HTABLE_CONTAINS_TUPLE(BPLONG Htable,BPLONG Tuple);
extern int b_GLOBAL_HEAP_VTABLE_REF_f(BPLONG op);
extern int b_GLOBAL_HEAP_GET_cf(BPLONG key,BPLONG value);
extern int c_GET_REDEFINE_WARNING(void);
extern int c_SET_REDEFINE_WARNING(void);
extern BPLONG fast_get_attr(BPLONG_PTR sv_ptr,BPLONG attr);
extern int b_GET_ATTR_ccf(BPLONG var,BPLONG attr,BPLONG value);
extern BPLONG c_domain_error(BPLONG type,BPLONG term);
extern BPLONG c_evaluation_error(BPLONG type);
extern BPLONG c_existence_error(BPLONG type,BPLONG term);
extern BPLONG c_permission_error(BPLONG type1,BPLONG type2,BPLONG term);
extern BPLONG c_representation_error(BPLONG type);
extern BPLONG is_iso_exception(BPLONG exception);
extern BPLONG c_builtin_error1(BPLONG type,BPLONG op1);
extern BPLONG c_builtin_error2(BPLONG type,BPLONG op1,BPLONG op2);
extern BPLONG c_builtin_error3(BPLONG type,BPLONG op1,BPLONG op2,BPLONG op3);
extern BPLONG c_builtin_error4(BPLONG type,BPLONG op1,BPLONG op2,BPLONG op3,BPLONG op4);
extern BPLONG c_type_error(BPLONG type,BPLONG term);
extern BPLONG c_syntax_error(BPLONG term);
extern BPLONG c_stream_struct(BPLONG Index);
extern BPLONG c_error_src(char *str,int arity);
extern int b_LIST_LENGTH_cff(BPLONG lst,BPLONG lstr,BPLONG len);
extern BPLONG intarray_to_intlist(BPLONG_PTR array,BPLONG len);
extern int bp_presort_int_list(BPLONG lst,BPLONG_PTR len_ptr,BPLONG_PTR sorted_ptr);
extern void intlist_to_intarray(BPLONG lst,BPLONG_PTR array);
extern void qsort_intarray(BPLONG_PTR arr, BPLONG len);
extern BPLONG bp_remove_eq_neibs(BPLONG lst);
extern BPLONG bp_reverse_list(BPLONG lst);
extern int b_sort_int_list(BPLONG lst,BPLONG sorted);
extern int c_sort_int_list(void);
extern int b_DEREF_c(BPLONG T);
extern int c_set_exception(void);
extern void Cboot_mic(void);

//
//	numbervars.c prototypes
//
extern void numberVarTermOpt(BPLONG term);
extern BPLONG unnumberVarTermOpt(BPLONG term);
extern BPLONG unnumberVarListOpt(BPLONG term);
extern BPLONG unnumberVarCommaOpt(BPLONG term);
extern BPLONG unnumberVarTerm(BPLONG term,BPLONG_PTR varVector,BPLONG_PTR maxVarNo);
extern void Cboot_numbervars(void);
extern int c_NUMBER_VARS(void);
extern int aux_number_vars__3(BPLONG op1,BPLONG n0);
extern int c_COPY_TERM(void);
extern BPLONG copy_term(BPLONG term);
extern int c_VARS_SET(void);
extern int c_VARS_SET_INTERSECT(void);
extern int vars_set_extract_vars(BPLONG term,BPLONG ex_term);
extern BPLONG collect_shared_vars(BPLONG term,BPLONG list0);
extern int c_SINGLETON_VARS(void);
extern int vars_set_extract_singleton_vars(BPLONG term);
extern int is_marked_as_singleton(BPLONG var);

//
//	sicstus_support.c prototypes
//
extern int c_current_host(void);
extern int c_process_id(void);
extern int c_socket_server_open(void);
extern int c_socket_server_accept(void);
extern int c_socket_server_close(void);
extern void check_hash(void);
extern int c_format_to_codes(void);
extern int c_now(void);
extern int c_term_hash(void);
extern int c_random(void);
extern int c_simple(void);

//
//	table.c prototypes
//
extern void init_table_area();
extern BPLONG table_area_size();
extern BPLONG table_area_notin_use();
extern void subgoal_table_statistics(int *nSubgoals, int *maxGTCollisions, float *aveGTCollisions,
                                     int *nAnswers, int *maxATCollisions, float *aveATCollisions,
                                     int *nTerms, int *maxTTCollisions, float *aveTTCollisions);
extern int table_area_num_expansions();
extern int c_INITIALIZE_TABLE();
extern void init_subgoal_table();
extern int identicalTabledTerms(BPLONG t1, BPLONG t2);
extern void match_term_tabledTerm(BPLONG t1,BPLONG t2);
extern BPLONG unnumberVarTabledTerm(BPLONG term);
extern void expandSubgoalTable();
extern BPLONG_PTR lookupSubgoalTable(BPLONG_PTR stack_arg_ptr,int arity,SYM_REC_PTR sym_ptr,int registerIfNotFound,int mode_bits);
extern int numberVarCopySubgoalArgsToTableArea(BPLONG_PTR stack_arg_ptr, BPLONG_PTR table_arg_ptr, int arity, BPLONG hcode0, BPLONG_PTR hcode_ptr);
extern int numberVarCopyAnswerArgsToTableArea(BPLONG_PTR stack_arg_ptr, BPLONG_PTR table_arg_ptr, int arity, BPLONG_PTR hcode_ptr);
extern BPLONG hashval_of_tabled_subgoal(BPLONG_PTR subgoal_entry);
extern BPLONG hashval_of_tabled_answer(BPLONG_PTR answer,int arity);
extern BPLONG hashval_of_numbered_term(BPLONG term);
extern int isGroundNumberedTerm(BPLONG term);
extern void propagate_scc_root(BPLONG_PTR fp,BPLONG_PTR subgoal_entry,BPLONG_PTR scc_root,BPLONG_PTR scc_root_ar);
extern void initialize_scc_elms(BPLONG_PTR subgoal_entry);
extern void complete_scc_elms(BPLONG_PTR subgoal_entry);
extern void print_scc(BPLONG_PTR scc_root);
extern BPLONG_PTR addFirstTableAnswer(BPLONG_PTR stack_arg_ptr, int arity);
extern BPLONG_PTR allocateAnswerTable(BPLONG_PTR first_answer, int arity);
extern int addTableAnswer(BPLONG_PTR stack_arg_ptr, int arity, BPLONG_PTR subgoal_entry);
extern void expandAnswerTable(BPLONG_PTR answer_table,int arity);
extern int addTableOptimalAnswer(BPLONG_PTR stack_arg_ptr, int arity, BPLONG_PTR subgoal_entry,int opt_arg_index,int maximize,int table_card);
extern int isBiggerTabledAnswer(BPLONG_PTR ans1,BPLONG_PTR ans2,int opt_arg_index);
extern int isSmallerTabledAnswer(BPLONG_PTR ans1,BPLONG_PTR ans2,int opt_arg_index);
extern void removeAnswerFromAnswerTable(BPLONG_PTR answer_table,BPLONG_PTR this_answer, int arity);
extern void copyTabledAnswerArgs(BPLONG_PTR src_ans, BPLONG_PTR des_ans, int arity);
extern void repositionLastTabledAnswerMin(BPLONG_PTR answer_table,BPLONG_PTR last_answer,int opt_arg_index);
extern void repositionLastTabledAnswerMax(BPLONG_PTR answer_table,BPLONG_PTR last_answer,int opt_arg_index);
extern void resetLastTabledAnswer(BPLONG_PTR answer_table, BPLONG_PTR prev_answer, BPLONG_PTR last_answer);
extern void insertTabledAnswerMin(BPLONG_PTR answer_table,BPLONG_PTR this_answer,int opt_arg_index);
extern void insertTabledAnswerMax(BPLONG_PTR answer_table,BPLONG_PTR this_answer,int opt_arg_index);
extern int c_VARIANT();
extern int is_variant(BPLONG op1, BPLONG op2);
extern int term_subsume_numberedterm(BPLONG t1, BPLONG t2);
extern int term_subsume_term(BPLONG op1, BPLONG op2);
extern int c_table_reset_subgoal_ar();
extern int c_table_reset_subgoal_ar();
extern int c_SUBGOAL_TABLE_SIZE();
extern int c_TABLE_GET_ONE_ANSWER();
extern int c_TABLE_GET_ALL_ANSWERS();
extern int table_subsume(BPLONG Call, SYM_REC_PTR sym_ptr, BPLONG_PTR arg_ptr);
extern BPLONG answer_table_entry_2_struct(SYM_REC_PTR sym_ptr, BPLONG_PTR ptr0);
extern int c_table_cardinality_limit();
extern int c_set_all_table_cardinality_limit();
extern int table_statistics();


//
//	thread.c prototypes
//
/*
extern BPLONG create_thread_record(void);
extern int b_THREAD_SET_CALL_cc(BPLONG MyNo,BPLONG Call);
extern int b_THREAD_GET_CALL_cf(BPLONG MyNo,BPLONG Call);
extern int b_THREAD_SET_TIMER_cc(BPLONG MyNo,BPLONG Timer);
extern int b_THREAD_GET_TIMER_cf(BPLONG MyNo,BPLONG Timer);
extern int b_THREAD_SET_STATUS_cc(BPLONG MyNo,BPLONG Status);
extern int b_THREAD_GET_STATUS_cf(BPLONG MyNo,BPLONG Status);
extern int b_THREAD_GET_CENTRAL_TIMER_f(BPLONG ct);
extern int b_THREAD_SET_CENTRAL_TIMER_c(BPLONG ct);
extern int b_THREAD_REGISTER_NEW_c(BPLONG Call);
extern int b_THREAD_RESCHEDULE(void);
extern int b_IS_LATEST_CHOICE_POINT_c(BPLONG cp);
extern int b_THREAD_ENABLE_INTERRUPT(void);
extern int b_THREAD_DISABLE_INTERRUPT(void);
*/

//
//	toam.c prototypes
//
extern void exception_handler(int signo);
extern void init_signals(void);
extern int initialize_bprolog(int argc,char *argv[]);
extern int toam(BPLONG_PTR P, BPLONG_PTR AR, BPLONG_PTR LOCAL_TOP);
extern void roll_tabled_frames(BPLONG_PTR b,BPLONG_PTR end_b);

//
//	token.c prototypes
//
extern void SyntaxError(CHAR_PTR message);
extern int handleEndInQuoted();
extern int read_character(FILE *card, int q);
extern int handleEolInQuoted2();
extern int read_character_string(int q);
extern int com0plain(FILE *card, int endeol);
extern int com0plain_string(int endeol);
extern int com2plain(FILE *card, int astcom, int endcom);
extern int com2plain_string(int astcom, int endcom);
extern int GetToken(void);
extern void printAtomStr(char *message);
extern int GetTokenString(void);
extern int b_NEXT_TOKEN_ff(BPLONG op1,BPLONG op2);
extern int b_SET_FLAG_DOUBLE_QUOTES_c(BPLONG FlagVal);
extern int b_GET_FLAG_DOUBLE_QUOTES_f(BPLONG FlagVal);
extern int c_get_line_nos(void);
extern int c_set_line_no(void);
extern int c_init_chars_pool(void);
extern int c_update_term_start_line_no(void);
extern int c_report_syntax_error(void);

//
//	unify.c prototypes
//
extern int unify(BPLONG op1, BPLONG op2);
extern void unify_nsv_v(BPLONG op1,BPLONG op2);
extern int unify_lst_lst(BPLONG op1,BPLONG op2);
extern int unify_str_str(BPLONG op1,BPLONG op2);
extern int is_IDENTICAL(BPLONG op1, BPLONG op2);
extern int identical(BPLONG op1, BPLONG op2);
extern int is_UNIFIABLE(BPLONG op1, BPLONG op2);
extern int c_UNIFIABLE(void);
extern int unify_suspvar_suspvar(BPLONG op1,BPLONG op2);
extern int unify_dvar_dvar(BPLONG_PTR dv_ptr1,BPLONG_PTR dv_ptr2);
extern int bind_susp_susp(BPLONG_PTR dv_ptr1,BPLONG_PTR dv_ptr2);
extern void merge_cs(BPLONG_PTR addr_head_cs1,BPLONG cs2);
extern int key_identical(BPLONG op1, BPLONG op2);
extern int unifyNumberedTerms(BPLONG t1, BPLONG t2);
extern int c_SAME_ADDR();

//
//	univ.c prototypes
//
extern BPLONG univ_lst2str(BPLONG L);
extern BPLONG univ_str2lst(BPLONG op1);
extern int list_length(BPLONG L, BPLONG orig_L);
extern int b_UNIV_cc(BPLONG op1,BPLONG op2);
