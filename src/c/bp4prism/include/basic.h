#ifndef BP_BASIC_H
#define BP_BASIC_H


/*
#define TRACE_INSTS

#ifdef TRACE_INSTS
extern int exec_trace[];
#endif
*/
#include <stdio.h>
#include <math.h>
/*
#define TOAM_LOCAL_REGS
*/
#define MAX_STR_LEN 3072
#define MAXREGS       5000
#define WRITEFLAG     1
#define READFLAG      0
#define BUCKET_CHAIN  65536
#define BP_ERROR -1L
#define BP_FALSE 0
#define BP_TRUE 1
#define MAXTRIGGERS   150000 /* maximum number of events that can be raised at one time. overflow not checked */
#define MULTIPLIER 37
#define MAX_ARITY 65535

#define T_ORDI        0    /* constant-type: no ep definition */
#define T_DYNA        1    /* constant-type: dynamic */
#define T_PRED        2    /* constant-type: ep points to compiled code */
#define C_PRED        3    /* c function */
#define T_INTP        4    /* interpreted */
#define T_TEMP_PRED  15    /* psc entry for predicate containing offset
                              entry point */
#define STACK_OVERFLOW 1
#define TRAIL_OVERFLOW 2
#define PAREA_OVERFLOW 3
#define OUT_OF_MEMORY 4

#define SMALL_MARGIN 1250
#define LARGE_MARGIN 12500

#define BP_MAXINT 268435455
#define BP_MININT -BP_MAXINT

/* ---------- Type Specifiers ----------------------------------------------- */

typedef char                CHAR;     /*  8 bits */
typedef unsigned char       BYTE;     /*  8 bits */
typedef unsigned short int  UW16;     /* 16 bits */
typedef long int            BPLONG;    /* 32 or 64 bits only for return types of functions */
typedef unsigned long int   BPULONG;    /* 32 or 64 bits */
typedef unsigned int        UW32;     /* 32 bits */

typedef CHAR  *CHAR_PTR; 
typedef BYTE  *BYTE_PTR;
typedef UW16  *UW16_PTR;
typedef BPLONG  *BPLONG_PTR;
typedef UW32  *UW32_PTR; 

/* [10 Apr 2007, by yuizumi]
 * Required to build a universal binary on Mac OS X.
 */
#ifdef DARWIN
#ifdef __LP64__
#define M64BITS
#endif
#endif

#ifdef M64BITS
#define NBITS_IN_LONG 64
#else
#define NBITS_IN_LONG 32
#endif

#define UBIGINT BPLONG_PTR

typedef struct sym_rec *SYM_REC_PTR;

typedef struct sym_rec {
  BYTE      etype;
  BYTE      spy;
  UW16      arity;
  UW16      length;
  CHAR_PTR  nameptr;
  SYM_REC_PTR next;
  int       (*ep)(void);   /* entry point to byte code,global var, c function, booltable */
} SYM_REC;


typedef struct event_func_data {	
  SYM_REC_PTR func;
  int signo;
} EVENT_FUNC;

/* An interpreted predicate (consulted or dynamic) is stored as a record of the following form:
   ref_count :   Number of unfinished calls of clause(Head,Body) 
   retr_count :  Number of unfinished calls of retract(Clause). 
                 The space of a retracted clause can be freed if (retra_count<=1 && ref_count==0).
   cl_count :    Number of clauses in the predicate
   bucket_size : Hashtable size. Initial value = number of clauses if static; 3 if dynamic
                 Hashtable size is doubled automatically if the count is greater than bucket_size.
   hashtable:    Pointer to the hashtable
*/		 
typedef struct {
  BPLONG ref_count;
  BPLONG retr_count;
  BPLONG cl_count;
  BPLONG time_stamp;
  BPLONG bucket_size;
  BPLONG_PTR hashtable;
} InterpretedPred;
typedef InterpretedPred *InterpretedPredPtr;

/* each bucket is a record of the following form:
   tail : pointer to the last clause record
   list : list of clauses hashed to this bucket
*/
typedef struct {
  BPLONG_PTR tail;
  BPLONG list;
} InterpretedPredBucket;
typedef InterpretedPredBucket *InterpretedPredBucketPtr;

/* each clause record contains the following fields:
   cl_ref: reference to the containing cell of the record
   head: head of the clause
   body: body of the class
   birth_time_stamp: time it was asserted
   death_time_stamp: time it was retracted
*/
typedef struct {
  BPLONG cl_ref;
  BPLONG head;
  BPLONG body;
  BPLONG birth_time_stamp;
  BPLONG death_time_stamp;
} InterpretedClause;
typedef InterpretedClause *InterpretedClausePtr;

/* used for hash-consing for findall area and table area */
typedef struct {
  BPLONG_PTR htable; 
  BPLONG size;     
  BPLONG count;    
} GTERMS_HTABLE;

typedef GTERMS_HTABLE *GTERMS_HTABLE_PTR;

/* used for findall area and table area*/
typedef struct {
  BPLONG_PTR low_addr,up_addr,top;
  GTERMS_HTABLE_PTR gterms_htable_ptr;
  int num_expansions;
} NUMBERED_TERM_AREA_RECORD;

typedef NUMBERED_TERM_AREA_RECORD *NUMBERED_TERM_AREA_RECORD_PTR;

#ifdef BUILTIN_PROTO_CHECK
typedef union {	
int  (*func0)(void);
int  (*func1)(BPLONG w);
int  (*func2)(BPLONG w,BPLONG x);
int  (*func3)(BPLONG w,BPLONG x,BPLONG y);
int  (*func4)(BPLONG w,BPLONG x,BPLONG y,BPLONG z);
} Builtin_funcs;

typedef struct {
	int argcount;
	Builtin_funcs func;
} Builtins;


#define INITIALIZE_BUILTIN0(index,function) {builtins[index].argcount = 0; builtins[index].func.func0 = function;}
#define INITIALIZE_BUILTIN1(index,function) {builtins[index].argcount = 1; builtins[index].func.func1 = function;}
#define INITIALIZE_BUILTIN2(index,function) {builtins[index].argcount = 2; builtins[index].func.func2 = function;}
#define INITIALIZE_BUILTIN3(index,function) {builtins[index].argcount = 3; builtins[index].func.func3 = function;}
#define INITIALIZE_BUILTIN4(index,function) {builtins[index].argcount = 4; builtins[index].func.func4 = function;}
#define EXECUTE_BUILTIN0(res,index) {int tempindex = index; \
				     if (builtins[tempindex].argcount != 0) \
					printf("builtin[%d] called with 0 args, expected %d\n",tempindex,builtins[tempindex].argcount); \
				     res = (*builtins[tempindex].func.func0)();}
#define EXECUTE_BUILTIN1(res,index,op) {int tempindex = index; \
				     if (builtins[tempindex].argcount != 1) \
					printf("builtin[%d] called with 1 args, expected %d\n",tempindex,builtins[tempindex].argcount); \
				     res = (*builtins[tempindex].func.func1)(op);}
#define EXECUTE_BUILTIN2(res,index,op1,op2) {int tempindex = index; \
				     if (builtins[tempindex].argcount != 2) \
					printf("builtin[%d] called with 2 args, expected %d\n",tempindex,builtins[tempindex].argcount); \
				     res = (*builtins[tempindex].func.func2)(op1,op2);}
#define EXECUTE_BUILTIN3(res,index,op1,op2,op3) {int tempindex = index; \
				     if (builtins[tempindex].argcount != 3) \
					printf("builtin[%d] called with 3 args, expected %d\n",tempindex,builtins[tempindex].argcount); \
				     res = (*builtins[tempindex].func.func3)(op1,op2,op3);}
#define EXECUTE_BUILTIN4(res,index,op1,op2,op3,op4) {int tempindex = index; \
				     if (builtins[tempindex].argcount != 4) \
					printf("builtin[%d] called with 4 args, expected %d\n",tempindex,builtins[tempindex].argcount); \
				     res = (*builtins[tempindex].func.func4)(op1,op2,op3,op4);}
#else
typedef int (*Builtins)();

#define INITIALIZE_BUILTIN0(index,function) builtins[index] = function
#define INITIALIZE_BUILTIN1(index,function) builtins[index] = function
#define INITIALIZE_BUILTIN2(index,function) builtins[index] = function
#define INITIALIZE_BUILTIN3(index,function) builtins[index] = function
#define INITIALIZE_BUILTIN4(index,function) builtins[index] = function
#define EXECUTE_BUILTIN0(res,index) res = (*builtins[index])()
#define EXECUTE_BUILTIN1(res,index,op) res = (*builtins[index])(op)
#define EXECUTE_BUILTIN2(res,index,op1,op2) res = (*builtins[index])(op1,op2)
#define EXECUTE_BUILTIN3(res,index,op1,op2,op3) res = (*builtins[index])(op1,op2,op3)
#define EXECUTE_BUILTIN4(res,index,op1,op2,op3,op4) res = (*builtins[index])(op1,op2,op3,op4)
#endif

extern int curr_toam_status;

#ifdef WIN32
#define INLINE 
#else
#define INLINE inline
#endif

#include "extern_decl.h"

#define TOAM_NOTSET 0
#define TOAM_MOUNTED 1
#define TOAM_STARTED 2
#define TOAM_DONE 3

#define MAX_CHARS_IN_POOL 10000

#define BP_BIGINT_BASE 268435456
#define BP_IN_SMALLINT_RANGE(op) ((op) > -268435456 && (op) < 268435456)
#define BP_IN_TINYINT_RANGE(op) ((op) > -65536 && (op) < 65536)

/* t1 is less than t2 */
#define TABLE_ANS_IS_LT(t1,t2) ((ISINT(t1) && ISINT(t2)) ? (INTVAL(t1)<INTVAL(t2)) : bp_compare(t1,t2)<0)

/* t1 is greater than t2 */
#define TABLE_ANS_IS_GT(t1,t2) ((ISINT(t1) && ISINT(t2)) ? (INTVAL(t1)>INTVAL(t2)) : bp_compare(t1,t2)>0)

extern int bp_gc;
extern int old_bp_gc;
extern int redefine_warning;

extern int confirm_copy_right;

extern BPLONG n_backtracks;
extern int     use_tabling;
extern BPLONG_PTR subgoalTable;
extern BPLONG subgoalTableBucketSize;
extern BPLONG subgoalTableEntriesCount;
extern BPLONG InitAnswerTableBucketSize;

extern BPLONG global_call_number;
extern BPLONG number_of_spy_points;
extern UW16 dg_flag_word;

extern FILE *curr_in, *curr_out;

extern BPLONG     stack_size,stack_size_limit, parea_size, findall_area_size, trail_size; /* declared in init.c */

extern BPLONG_PTR  parea_low_addr;          /* psc records, instructions, p-names */
extern BPLONG_PTR  parea_water_mark;        /* overflow occurs when this is reached */
extern BPLONG_PTR  trail_low_addr;          /* trail stack */
extern BPLONG_PTR  trail_water_mark;        /* overflow occurs when this is reached after compaction*/
extern BPLONG_PTR  trail_water_mark0;       /* compact when this is reached */

extern BPLONG_PTR  stack_up_addr;
extern BPLONG_PTR  stack_low_addr;
extern BPLONG_PTR  trail_up_addr;

extern BPLONG_PTR cpreg;
extern CHAR_PTR  curr_fence;      /* ptr to next free byte in perm space */
extern BPLONG_PTR  parea_up_addr;       /* ptr to last+1 free byte in perm space */
extern BPLONG_PTR  inst_begin;      /* ptr to the beginning of inst. array */
extern BPLONG_PTR  addr_halt;
extern BPLONG_PTR  addr_halt0;
extern BPLONG_PTR  addr_fail;
extern BPLONG_PTR  addr_table_consume;
extern BPLONG_PTR  addr_table_consume0;

extern BPLONG_PTR  gc_low;
extern BPLONG_PTR  gc_upper;
extern int      gc_is_working;
extern BPLONG_PTR  copy_area_low, copy_area_high;

                                  /* ptr to the failb inst */
extern BPLONG_PTR  top;
extern BPLONG_PTR  arreg;           /* last activation record       */
extern BPLONG_PTR  local_top;
extern BPLONG_PTR  local_top0;
extern BPLONG_PTR  breg;            /* latest choice point            */
extern BPLONG_PTR  breg0;           /* choice point where global variables for cglib are stored */
extern BPLONG_PTR  heap_top;        /* top of heap                  */
extern BPLONG_PTR  trail_top;       /* top of trail stack           */
extern BPLONG_PTR  hbreg;           /* heap backtrack point         */
extern BPLONG_PTR  sfreg;           /* latest suspension frame      */
extern BPLONG_PTR  gc_b;

extern BPLONG_PTR  triggeredCs[MAXTRIGGERS];
extern int  event_flag[MAXTRIGGERS];
extern BPLONG  event_object[MAXTRIGGERS];
extern BPLONG_PTR  triggering_frame[MAXTRIGGERS];
extern BPLONG  trigger_no;      /* list of woken frames         */
extern SYM_REC_PTR  hash_table[BUCKET_CHAIN];
extern BPLONG  number_of_symbols;
extern BPULONG     toam_signal_vec;
extern int	   user_signal;
extern int      in_critical_region;
extern BPLONG       fd_region_low;
extern BPLONG       fd_region_up;

extern BPLONG  curr_line_no;

extern int      disassem;
extern BPLONG       num_line;        /* print instruction addresses on trace
				     and disassem */
extern int      trace_toam;

extern  double  double_v;

extern BPLONG      global_var_num;
extern int     number_var_exception;
extern BPLONG      global_unnumbervar_max;
extern BPLONG_PTR global_unnumbervar_ptr;
extern BPLONG_PTR global_unnumbervar_watermark;

extern BPLONG      exception;
extern SYM_REC_PTR  list_psc;
extern BPLONG list_psc_int;
extern BPLONG list_psc_hashcode;
extern SYM_REC_PTR  error_psc;
extern SYM_REC_PTR  v2_psc; /* v(_,_) */
extern SYM_REC_PTR  v3_psc; /* v(_,_,_) */
extern SYM_REC_PTR  v4_psc; /* v(_,_,_,_) */
extern SYM_REC_PTR  findall_result0;
extern SYM_REC_PTR  findall_result1;
extern SYM_REC_PTR  findall_result2;
extern SYM_REC_PTR  interval_psc;
extern SYM_REC_PTR  call_cleanup_psc;
extern SYM_REC_PTR  hshtb_psc;
extern SYM_REC_PTR  set_constructor_psc;
extern SYM_REC_PTR  comma_psc;
extern SYM_REC_PTR  float_psc;
extern SYM_REC_PTR  address_psc;
extern SYM_REC_PTR  socket_psc;
extern SYM_REC_PTR  stream_psc;
extern SYM_REC_PTR  bigint_psc;
extern SYM_REC_PTR  timer_psc;
extern SYM_REC_PTR  enter_dyn_call;
extern SYM_REC_PTR  enter_catch_call;
extern SYM_REC_PTR  damon_load;
extern SYM_REC_PTR  slash_psc;
extern SYM_REC_PTR handle_exception_psc;
extern SYM_REC_PTR forward_exception_psc;
extern SYM_REC_PTR vv_eq_c_arc;
extern SYM_REC_PTR v_eq_vc_arc;
extern SYM_REC_PTR uu_eq_c_arc;
extern SYM_REC_PTR u_eq_uc_arc;
extern SYM_REC_PTR c_object_ref_sym;
extern SYM_REC_PTR register_event_listener;

extern BPLONG unknown_exception;
extern BPLONG true_atom;
extern BPLONG dec_ref_atom;
extern BPLONG dec_retr_atom;
extern BPLONG dummy_event_object;
extern BPLONG failure_atom;
extern BPLONG empty_set;
extern BPLONG  not_care_symbol;
extern BPLONG  nil_sym, period_sym,varid;
extern BPLONG  atom_expected;
extern BPLONG  integer_expected;
extern BPLONG  invalid_number_format;
extern BPLONG  number_expected;
extern BPLONG  arith_expr_expected;
extern BPLONG  list_expected;
extern BPLONG  structure_expected;
extern BPLONG  char_code_expected;
extern BPLONG  float_format_expected;
extern BPLONG  operator_type_expected;
extern BPLONG  variable_expected;
extern BPLONG  nonvariable_expected;
extern BPLONG  could_not_open_stream;
extern BPLONG  file_does_not_exist;
extern BPLONG  divide_by_zero;
extern BPLONG  illegal_expression;
extern BPLONG  illegal_timer;
extern BPLONG  illegal_list;
extern BPLONG  illegal_mode_string;
extern BPLONG  illegal_predicate;
extern BPLONG  not_modifiable;
extern BPLONG  out_of_range;
extern BPLONG  out_of_memory_atom;
extern BPLONG  out_of_memory_stack_heap;
extern BPLONG  out_of_memory_table;
extern BPLONG  undefined_predicate;
extern BPLONG  invalid_byte_file;
extern BPLONG  code_area_overflow;
extern BPLONG  permission_error;
extern BPLONG  control_stack_overflow;
extern BPLONG  trail_stack_overflow;
extern BPLONG  integer_overflow;
extern BPLONG  illegal_arguments;
extern BPLONG  out_of_bound;
extern BPLONG  illegal_event;
extern BPLONG  output_mode_error;
extern BPLONG  java_exception;
extern BPLONG  bp_initialization_error;
extern BPLONG  run_time_error;
extern BPLONG  call_damon_load_atom;
extern BPLONG dots_atom;

extern int  debugging_susp;

extern BPLONG main_args;

extern BPLONG  lastc;

/* for threads */
extern SYM_REC_PTR thread_psc;
extern SYM_REC_PTR thread_start_psc;
extern BPLONG attr_cfd_atm;
extern BPLONG attr_neq_atm;
extern SYM_REC_PTR cfd_func_pair;
extern SYM_REC_PTR hashtable_psc;
extern BPLONG thread_table_overflow;
extern BPLONG_PTR bp_threads;
extern BPLONG thread_table_size;
extern BPLONG thread_count;
extern BPLONG thread_current_no;
extern BPLONG thread_interruptable;

/* extern void *malloc(); */

#ifdef LINUX
extern BPLONG addr_top_bit;
#endif

#ifdef LINUX
#define BP_MALLOC(ptr,size){					\
  ptr = (BPLONG_PTR)(malloc(size*sizeof(BPLONG)));			\
  if (ptr!=NULL){							\
    if (addr_top_bit == -1L){						\
      addr_top_bit = ((BPLONG)ptr & TOP_BIT);				\
    } else {								\
      if (addr_top_bit!=((BPLONG)ptr & TOP_BIT)){			\
	free(ptr);							\
	ptr = NULL;							\
      }									\
    }									\
  }									\
  }
#else
#define BP_MALLOC(ptr,size){					\
    ptr = (BPLONG_PTR)(malloc(size*sizeof(BPLONG)));			\
    if (ptr!=NULL){							\
      if ((BPLONG)ptr & TOP_BIT){					\
	free(ptr);							\
	ptr = NULL;							\
      }									\
    }									\
  }
#endif

extern BPLONG use_gl_getline;
extern BPLONG num_stack_expansions;
extern BPLONG num_trail_expansions;
extern BPLONG num_parea_expansions;

extern SYM_REC_PTR 
  str_BUILTIN_ERROR1,
  str_BUILTIN_ERROR2,
  str_BUILTIN_ERROR3,
  str_BUILTIN_ERROR4,
  str_DOMAIN_ERROR,
  str_EVALUATION_ERROR,
  str_EXISTENCE_ERROR,
  str_PERMISSION_ERROR,
  str_REPRESENTATION_ERROR,
  str_RESOURCE_ERROR,
  str_TYPE_ERROR,
  str_SYNTAX_ERROR;

extern BPLONG 
  et_OUT_OF_MEMORY,
  et_OUT_OF_MEMORY_STACK,
  et_OUT_OF_MEMORY_TABLE,
  et_ACCESS, 
  et_ATOM,
  et_ATOMIC,
  et_BINARY_STREAM,
  et_BYTE,
  et_CALLABLE,
  et_CHARACTER,
  et_CHARACTER_CODE,
  et_COMPOUND,
  et_CREATE, 
  et_EVALUABLE,
  et_FLAG,
  et_FLAG_VALUE,
  et_FLOAT_OVERFLOW,
  et_INPUT, 
  et_INTEGER,
  et_INT_OVERFLOW,
  et_IN_BYTE,
  et_IN_CHARACTER,
  et_IN_ESCCHARACTER,
  et_IN_CHARACTER_CODE,
  et_INSTANTIATION_ERROR,
  et_IO_MODE,
  et_LIST,
  et_MAX_ARITY,
  et_MAX_INTEGER,
  et_MODIFY, 
  et_NON_EMPTY_LIST,
  et_NOT_LESS_THAN_ZERO,
  et_NUMBER,
  et_OPEN, 
  et_OPERATOR,
  et_OPERATOR_PRIORITY,
  et_OUTPUT, 
  et_PAST_END_OF_STREAM,
  et_PREDICATE_INDICATOR,
  et_PRIVATE_PROCEDURE,
  et_PROCEDURE,
  et_PROLOG_FLAG,
  et_READ_OPTION,
  et_REPOSITION, 
  et_SOURCE_SINK,
  et_STATIC_PROCEDURE,
  et_STREAM,
  et_STREAM_OPTION,
  et_STREAM_OR_ALIAS,
  et_STREAM_PROPERTY,
  et_TEXT_STREAM,
  et_UNDEFINED,
  et_UNDERFLOW,
  et_VARIABLE,
  et_WRITE_OPTION,
  et_ZERO_DIVISOR,
  et_BIGINT_TOO_LONG;

#endif	
