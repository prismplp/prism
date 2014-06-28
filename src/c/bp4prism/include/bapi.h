/* bapi.h */
#include "frame.h"

#define BP_TYPE_VARIABLE 0x100L
#define BP_TYPE_INTEGER  0x101L
#define BP_TYPE_FLOAT    0x102L
#define BP_TYPE_ATOM     0x103L
#define BP_TYPE_COMPOUND 0x104L

#define BP_pred_ref SYM_REC_PTR

#define BP_term_ref BPLONG

#define IS_HEAP_REFERENCE(ptr) (ptr>=stack_low_addr && ptr<=heap_top)
#define IS_STACK_REFERENCE(ptr) (ptr>=local_top && ptr<=stack_up_addr)
#define IS_STACK_OR_HEAP_REFERENCE(ptr) ((BPLONG_PTR)ptr>=stack_low_addr && (BPLONG_PTR)ptr<=stack_up_addr)

#define INSIDE_HEAP_TOP_SEGMENT(ptr) (ptr>=hbreg && ptr<heap_top)

#define BP_check_integer(A) {			\
    if (!ISINT(A)){				\
      exception = illegal_arguments;		\
      return BP_ERROR;				\
    }						\
  }

#define BP_check_float(A) {			\
    if (!ISFLOAT(A)){				\
      exception = illegal_arguments;		\
      return BP_ERROR;				\
    }						\
  }

#define BP_check_atom(A) {			\
    if (!ISATOM(A)){				\
      exception = illegal_arguments;		\
      return BP_ERROR;				\
    }						\
  }

#define BP_DOUBLE_VAL(op) (ISINT(op) ? (double)INTVAL(op) : (ISFLOAT(op) ? floatval(op) : bp_bigint_to_double(op)))

#define PUSHTRAIL(val)							\
  if (((BPLONG_PTR)(val) > breg) || ((BPLONG_PTR)(val) < hbreg)) {	\
    PUSHTRAILC(val,val);}

#define PUSHTRAIL_h(val)			\
  if ((BPLONG_PTR)(val) < hbreg) {		\
    PUSHTRAILC(val,val);}

#define PUSHTRAIL_s(val)			\
  if (((BPLONG_PTR)(val) > breg))		\
    PUSHTRAILC(val,val)

#define PUSHTRAIL_S_NONATOMIC(p,val)			\
  if ((BPLONG_PTR)(p) > B)				\
    PUSHTRAILC(ADDTAG3(p,TRAIL_VAL_NONATOMIC),val)

#define PUSHTRAIL_H_NONATOMIC(p,cont)			\
  if ((BPLONG_PTR)p<hbreg) {				\
    PUSHTRAILC(ADDTAG3(p,TRAIL_VAL_NONATOMIC),cont);}

#define PUSHTRAIL_H_BIT_VECTOR(p,cont)			\
  if ((BPLONG_PTR)p<hbreg) {				\
    PUSHTRAILC(ADDTAG3(p,TRAIL_BIT_VECTOR),cont);}

#define PUSHTRAIL_H_ATOMIC(p,cont)			\
  if ((BPLONG_PTR)p<hbreg) {				\
    PUSHTRAILC(ADDTAG3(p,TRAIL_VAL_ATOMIC),cont);}

#define PUSHTRAILC_ATOMIC(p,cont)		\
  PUSHTRAILC(ADDTAG3(p,TRAIL_VAL_ATOMIC),cont);

#define PUSHTRAIL_UC_0(addr) PUSHTRAILC(addr,addr)

#define PUSHTRAILC(addr,val) {			\
    if (trail_top<=trail_water_mark0){		\
      trail_top = expand_trail(trail_top,breg); \
    };						\
    *trail_top-- = (BPLONG)(val);		\
    *trail_top-- = (BPLONG)(addr);}

#define POPTRAIL(t){				\
    ++t; op1 = UNTAGGED3(FOLLOW(t));		\
    ++t; *(BPLONG_PTR)op1 = FOLLOW(t);		\
  }

#define UNDO_TRAILING {				\
    register BPLONG op1;			\
    while (trail_top != trail_top0) {		\
      POPTRAIL(trail_top);			\
    }}

#define ARG(i,arity) *(arreg+arity-i+1)

#define NEW_VAR(op)				\
  op = (BPLONG)heap_top;			\
  NEW_HEAP_FREE;

#define ASSIGN_STRUCT(op1,op2)				\
  *(BPLONG_PTR)(op1) = ADDTAG(heap_top,STR);		\
		     PUSHTRAIL(op1);			\
		     NEW_HEAP_NODE((BPLONG)op2);

#define ASSIGN_LIST(op)				\
  *(BPLONG_PTR)(op) = ADDTAG(heap_top,LST);	\
		    PUSHTRAIL(op);

#define MOVE_STRUCT(op1,op2)			\
  op1 = ADDTAG(heap_top,STR);			\
  NEW_HEAP_NODE((BPLONG)op2);

#define ASSIGN_VALUE(var,value)			\
  *(BPLONG_PTR)var = value;			\
  PUSHTRAIL(var)

#define ASSIGN_TRAIL_VALUE(var,value)           \
  *(BPLONG_PTR)var = (BPLONG)(value);		\
  PUSHTRAILC(var,var)

#define ASSIGN(var,value)			\
  *(BPLONG_PTR)var = value

#define LOCAL_OVERFLOW_CHECK(src)		\
  if (local_top < heap_top+SMALL_MARGIN) {	\
    myquit(STACK_OVERFLOW,src);			\
  }

#define LOCAL_OVERFLOW_CHECK_WITH_MARGIN(src,margin)		\
  if (local_top < heap_top+margin) {				\
    exception = et_OUT_OF_MEMORY;				\
    return BP_ERROR;						\
  }

#define QUIT_IF_OVERFLOW_HEAP(margin) if (local_top-margin < heap_top){ \
    quit("Control stack overflow. (domain.c) \n");			\
  }

#define BUILD_VALUE(op1,lab1)				\
  SWITCH_OP_VAR(op1,lab1,				\
		{if ((BPLONG_PTR)op1<heap_top)		\
		    NEW_HEAP_NODE(op1);			\
		  else {				\
		    FOLLOW(op1) = (BPLONG)heap_top;	\
		    PUSHTRAIL(op1);			\
		    NEW_HEAP_FREE;}},			\
		{NEW_HEAP_NODE(UNTAGGED_ADDR(op1));},	\
		{NEW_HEAP_NODE(op1);})

#define GET_CAR(op) FOLLOW((BPLONG_PTR)UNTAGGED_ADDR(op))
#define GET_CDR(op) FOLLOW((BPLONG_PTR)UNTAGGED_ADDR(op)+1)
#define GET_ARG(op,n) FOLLOW((BPLONG_PTR)UNTAGGED_ADDR(op)+n)

#define LIST_NEXT(op) *(op+1)

/* tail and new_list maybe the same */
#define NEW_LIST_NODE(elm,tail,new_list){	\
    BPLONG tmp;					\
    tmp = ADDTAG(heap_top,LST);			\
    FOLLOW(heap_top++) = elm;			\
    FOLLOW(heap_top++) = tail;			\
    new_list = tmp;				\
  }

extern BPLONG no_gcs;

#define ALLOCATE_NEW_PAREA_BLOCK(size,success) {			\
    BPLONG_PTR old_parea_low_addr,tmp_ptr;				\
    old_parea_low_addr = parea_low_addr;				\
    BP_MALLOC(tmp_ptr,size);						\
    if (tmp_ptr!=NULL){							\
      success=1;							\
      num_parea_expansions++;						\
      parea_low_addr = tmp_ptr;						\
      FOLLOW(parea_low_addr) = (BPLONG)old_parea_low_addr;		\
      FOLLOW(parea_low_addr+1) = size;					\
      parea_up_addr = parea_low_addr + size;				\
      parea_water_mark =  parea_up_addr - SMALL_MARGIN;			\
      curr_fence     = (CHAR_PTR)(parea_low_addr+2);			\
    } else { 								\
      success=0;							\
    }									\
  }

#define GET_GLOBAL_SYM(name,arity,sym_ptr){				\
    DEREF(name);DEREF(arity);						\
    if (!ISATOM(name) || !ISINT(arity)){				\
      exception = illegal_arguments; return BP_ERROR;			\
    }									\
    sym_ptr = GET_ATM_SYM_REC(name);					\
    if (arity!=BP_ZERO)  sym_ptr = insert(GET_NAME(sym_ptr),GET_LENGTH(sym_ptr),INTVAL(arity)); \
  }


#define NUMBERED_TERM_BLOCK_SIZE 1000000

#define ADD_NEW_NUMBERED_TERM_AREA_BLOCK(area_record_ptr,success) { \
    BPLONG_PTR tmp_ptr;							\
    BP_MALLOC(tmp_ptr,NUMBERED_TERM_BLOCK_SIZE);			\
    if (tmp_ptr!=NULL){							\
      success=1;							\
      FOLLOW(tmp_ptr) = (BPLONG)(area_record_ptr->low_addr);		\
      area_record_ptr->low_addr = tmp_ptr;				\
      area_record_ptr->up_addr = tmp_ptr+NUMBERED_TERM_BLOCK_SIZE;	\
      area_record_ptr->top = tmp_ptr+1;					\
    } else { 								\
      success=0;							\
    }									\
  }

#define ALLOCATE_FROM_NUMBERED_TERM_AREA(area_record_ptr,ptr,size){	\
    if (area_record_ptr->top + size >= area_record_ptr->up_addr){	\
      int success;							\
      ADD_NEW_NUMBERED_TERM_AREA_BLOCK(area_record_ptr,success);	\
      if (success==0){							\
	ptr = NULL;							\
      } else {								\
	area_record_ptr->num_expansions++;				\
	ptr = area_record_ptr->top;					\
	area_record_ptr->top += size;					\
      }									\
    } else {								\
      ptr = area_record_ptr->top;					\
      area_record_ptr->top += size;					\
    }									\
  }


