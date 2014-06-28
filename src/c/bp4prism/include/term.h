/* term.h */

#define REF               0x0L   /* 00 */
#define LST               0x3L   /* 11 */
#define STR               0x1L   /* 01 */
#define ATM               0x2L   /* 10 */
#define EPSILON           1.0e-12
#define MASK2             0x2L

#define MASK_LOW28        0xfffffffL

#ifdef M64BITS
#define SUSP              0x8000000000000001L
#define TOP_BIT_MASK      0x8000000000000000L
#define TOP_BIT           0x8000000000000000L
#define TAG_MASK          0x8000000000000003L
#define INT_TAG           0x8000000000000002L
#define INT_TAG32         0x80000002L
#define VAL_MASK1         0xfffffffffffffffcL
#define VAL_MASK0         0x7ffffffffffffffcL
#define FFFF              0xffffffffffffffffL
#define MASK_FF           0xffffffffffffffffL
#define HASH_BITS         0x000000000fffffffL
#define MASK_LOW16        0x000000000000ffffL
#else
#define SUSP              0x80000001L
#define TOP_BIT_MASK      0x80000000L
#define TOP_BIT           0x80000000L
#define TAG_MASK          0x80000003L
#define INT_TAG32         0x80000002L
#define INT_TAG           0x80000002L
#define VAL_MASK1         0xfffffffcL
#define VAL_MASK0         0x7ffffffcL
#define FFFF              0xffffffffL
#define MASK_FF           0xffffffffL
#define HASH_BITS         0x0fffffffL
#define MASK_LOW16        0x0000ffffL
#endif

/****************************************************************************/
#define FOLLOW(op)        (*(BPLONG_PTR)(op))     /* return what op points to */

#define TAG(op)           (((BPLONG)op) & 0x3L)
#define ISREF(op)         (!(TAG(op)))
#define ISVAR(op)         (!(TAG(op)))
#define ISNUM(op)         (ISINT(op) || (ISSTRUCT(op) && (IS_FLOAT_PSC(op) || IS_BIGINT_PSC(op))))
#define ISLIST(op)        (((op) & TAG_MASK) == LST)
#define ISSTRUCT(op)      ((TAG_MASK & (op)) == STR)
#define IS_SUSP_VAR(op)   (((op) & TAG_MASK) == SUSP)
#define ISINT(op)         (((op) & TAG_MASK) == INT_TAG)
#define IS_SMALLINT(op)   (((op) & TAG_MASK) == INT_TAG)
#define ISFLOAT(op)       (ISSTRUCT(op) && FOLLOW(UNTAGGED_ADDR(op))==(BPLONG)float_psc)
#define IS_FLOAT_PSC(op)   (FOLLOW(UNTAGGED_ADDR(op))==(BPLONG)float_psc)
#define ISADDR(op)        (ISSTRUCT(op) && FOLLOW(UNTAGGED_ADDR(op))==(BPLONG)address_psc)
#define IS_BIGINT(op)     (ISSTRUCT(op) && FOLLOW(UNTAGGED_ADDR(op))==(BPLONG)bigint_psc)
#define IS_BIGINT_PSC(op) (FOLLOW(UNTAGGED_ADDR(op))==(BPLONG)bigint_psc)
#define ISNONVAR(op)      (TAG(op))
#define ISFREE(var)       ((BPLONG)var == FOLLOW(var)) /* must be a ref */
#define ISATOM(op)        (((op) & TAG_MASK) == ATM)
#define ISNIL(op)         ((op) == nil_sym)
#define ISCOMPOUND(op)    (((op) & 0x1L) == STR && (op)>0)
#define IS_VAR_OR_STRUCT(op) (((op) & MASK2) == 0)

#define MAKEINT32(op)       (((int)(op) << 2) | INT_TAG32)
#define MAKEINT(op)       (((BPLONG)(op) << 2) | INT_TAG)
#define MAKE_INT_OR_BIGINT(op) (BP_IN_SMALLINT_RANGE(op) ? MAKEINT(op) : bp_int_to_bigint(op))
#define INTVAL(op)        (((BPLONG)(op) <<1) >> 3)
#define VALOF_INT_OR_BIGINT(op)  (ISINT(op) ? INTVAL(op) : bp_bigint_to_int(op))
#define NUMVAL(op)        (ISINT(op) ? INTVAL(op) : floatval(op))

#define BP_ZERO MAKEINT(0)
#define BP_ONE  MAKEINT(1)
#define BP_TWO  MAKEINT(2)
#define BP_THREE  MAKEINT(3)
#define BP_MONE  MAKEINT(-1L)


#define ADDTAG3(op,tag)    ((BPLONG)(op) | tag)
#define UNTAGGED3(op)      (((BPLONG)(op)) & ~0x3L) /* fffffffc */
#define UNTAGGED_CONT(op)      (((BPLONG)(op)) & VAL_MASK0) /* 7ffffffc */
/*
#define POS_OVERFLOW(op)  ((BPLONG)(op) & 0xf8000000)
#define NEG_OVERFLOW(op)  (((BPLONG)(op) & 0xf8000000) != 0xf8000000)
#define POS_ANS(op1,op2)  (((BPLONG)(op1)&0x40000000) == ((BPLONG)(op2)&0x40000000))
*/

#ifdef LINUX
#ifdef M64BITS
#define ADDTAG(op,tag)    ((BPLONG)(op) | tag)
#define UNTAG_ADDR(op)         op &= VAL_MASK0
#define UNTAGGED_ADDR(op)      (((BPLONG)op) & VAL_MASK0)
#define UNTAGGED_TOPON_ADDR(op) UNTAGGED_ADDR(op)
#else
#define ADDTAG(op,tag)    (((BPLONG)(op) & ~TOP_BIT) | tag)
#define UNTAG_ADDR(op)         op = ((op & VAL_MASK0) | addr_top_bit)
#define UNTAGGED_ADDR(op)      ((((BPLONG)(op)) & VAL_MASK0) | addr_top_bit)
#define UNTAGGED_TOPON_ADDR(op) UNTAGGED_ADDR(op)
#endif
#else
#define ADDTAG(op,tag)    ((BPLONG)(op) | tag)
#define UNTAG_ADDR(op)         op &= VAL_MASK0
#define UNTAGGED_ADDR(op)      (((BPLONG)(op)) & VAL_MASK0)
#define UNTAGGED_TOPON_ADDR(op) UNTAGGED_ADDR(op)
#endif

/*
#ifdef LARGE_ADDR_SPACE
#define ADDTAG(op,tag)    (((BPLONG)(op) & ~TOP_BIT) | tag)
#define UNTAG_ADDR(op)         op = ((op & VAL_MASK0) | TOP_BIT)
#define UNTAGGED_ADDR(op)      (((BPLONG)op) & VAL_MASK0 | TOP_BIT)
#define UNTAGGED_TOPON_ADDR(op) (((BPLONG)op) & ~0x3L)
#else
#define ADDTAG(op,tag)    ((BPLONG)(op) | tag)
#define UNTAG_ADDR(op)         op &= VAL_MASK0
#define UNTAGGED_ADDR(op)      (((BPLONG)op) & VAL_MASK0)
#define UNTAGGED_TOPON_ADDR(op) (((BPLONG)op) & VAL_MASK0)
#endif
*/

#define GET_ATM_SYM_REC(op)    ((SYM_REC_PTR)UNTAGGED_ADDR(op))
#define GET_STR_SYM_REC(op)    ((SYM_REC_PTR)(FOLLOW(UNTAGGED_ADDR(op))))
#define GET_SYM_REC(op)    (ISATOM(op) ? ((SYM_REC_PTR)UNTAGGED_ADDR(op)) : ((SYM_REC_PTR)(FOLLOW(UNTAGGED_ADDR(op)))))

#define GET_SYM_ARITY(op)  GET_ARITY(GET_SYM_REC(op))
#define GET_STR_SYM_ARITY(op)  GET_ARITY(GET_STR_SYM_REC(op))

#define GET_SYM_LENGTH(op) GET_LENGTH(GET_SYM_REC(op))

#define DEREF(op)					    \
  while (ISREF(op)) {					    \
    top =  (BPLONG_PTR)FOLLOW(op);			    \
    if ((BPLONG)top==op)				    \
      break;						    \
    op = (BPLONG)top;					    \
  }

#define DEREF2(op,VarCode)				    \
  while (ISREF(op)) {					    \
    top =  (BPLONG_PTR)FOLLOW(op);			    \
    if ((BPLONG)top==op){VarCode;}			    \
    op = (BPLONG)top;					    \
  }


#define DEREF_NONVAR(op) while (ISREF(op)) \
                            op = FOLLOW(op)  \
 
#define DEREF_SUSP_VAR(op,dv_ptr) while (ISREF(op)){ \
                            dv_ptr = op; \
                            op = FOLLOW(op);}  \
 
#define NDEREF(op,labl)  top = (BPLONG_PTR)FOLLOW(op);    \
			 if ((BPLONG)top!=op) { \
                          op = (BPLONG)top;   \
			    goto labl;        \
			 }

#define IsNumberedVar(term) (((term) & TAG_MASK)==TAG_MASK)
#define NumberVar(num) (((BPLONG)(num) << 2) | TAG_MASK)

/****************************************************************************/
/* The following are macros for setting heap values. */

#define MAKE_FREE(type,var)  (var) = (type)&(var)
/* must pass a simple pointer, not an expression */

#define NEW_HEAP_FREE *heap_top = (BPLONG)heap_top; heap_top++

/* make a free variable on the top of the heap */

#define NEW_HEAP_NODE(x)          *heap_top++ = x
/* make a new heap node with value x (one word type) */

/*****************************************************************************/
#define GET_ETYPE(ptr)  ((ptr)->etype)
#define GET_SPY(ptr)  ((ptr)->spy)
#define GET_ARITY(ptr)  ((ptr)->arity)
#define GET_LENGTH(ptr) ((ptr)->length)
#define GET_EP(ptr)     ((ptr)->ep)
#define GET_NAME(ptr)   ((ptr)->nameptr)
#define GET_NEXT(ptr)   ((ptr)->next)
/*
#define IS_PRED(sym)    (GET_ETYPE(sym) == T_PRED)
#define IS_CPRED(sym)   (GET_ETYPE(sym) == C_PRED)
#define IS_DYNA(sym)    (GET_ETYPE(sym) == T_DYNA)
#define IS_ORDI(sym)    (GET_ETYPE(sym) == T_ORDI)
*/

#define BP_NEW_SYM(name,arity) insert(name,strlen(name),arity)

#define IHASH(val,size)  (((unsigned long int)val >>2 ) % size)

#ifdef  M64BITS
#define ALIGN(type,ptr)  ptr = (type)((BPULONG)((CHAR_PTR)ptr + 7) & 0xfffffffffffffff8L)
#else
#define ALIGN(type,ptr)  ptr = (type)((BPULONG)((CHAR_PTR)ptr + 3) & 0xfffffffc)
#endif

#define ASSIGN_f_atom(op,value){FOLLOW(op)=value;}
#define ASSIGN_v_heap_term(op,value){FOLLOW(op)=value;PUSHTRAIL(op);}
#define ASSIGN_sv_heap_term(op,value){FOLLOW(op)=value;PUSHTRAIL_s(op);}

/*
#define SWITCH_OP(op,nderef,VarCode,AtmCode,LstCode,StrCode,SuspCode) \
        nderef: DEREF2(op,VarCode); \
   	 if (TAG(op)==ATM){ \
	      AtmCode; \
	 } else if (TAG(op)==LST) { \
	      LstCode; \
	 } else if (ISSTRUCT(op)){ \
	      StrCode; \
	 } else { \
              SuspCode; \
        }

*/
#define SWITCH_OP(op,nderef,VarCode,AtmCode,LstCode,StrCode,SuspCode) \
 nderef: switch (TAG(op)) { \
	    case REF: \
	      NDEREF(op,nderef); \
   	      VarCode; \
              break; \
            case ATM: \
	      AtmCode; \
              break; \
	    case LST: \
	      LstCode; \
              break; \
	    case STR: \
	      if (op>0) StrCode \
              else SuspCode;break;}

#define SWITCH_OP_NOSUSP(op,nderef,VarCode,AtmCode,LstCode,StrCode) \
 nderef: switch (TAG(op)) { \
	    case REF: \
	      NDEREF(op,nderef); \
   	      VarCode; \
              break; \
            case ATM: \
	      AtmCode; \
              break; \
	    case LST: \
	      LstCode; \
              break; \
	    case STR: \
              StrCode;break;}

/*
#define SWITCH_OP_STRUCT(op,nderef,VarCode,StrCode,SuspCode) \
    nderef:if (ISSTRUCT(op)) \
              StrCode \
           else if (ISREF(op)){ \
 	      NDEREF(op,nderef); \
   	      VarCode;} \
           else if (op<0 && TAG(op) == STR) \
              SuspCode; \
*/
#define SWITCH_OP_STRUCT(op,nderef,VarCode,StrCode,SuspCode) \
    nderef:if (ISREF(op)){ \
 	      NDEREF(op,nderef); \
   	      VarCode;} \
           else if (ISSTRUCT(op)){ \
              StrCode \
           } else if (IS_SUSP_VAR(op)) \
              SuspCode; \
 
#define SWITCH_OP_INT(op,nderef,VarCode,IntCode,OtherCode) \
    nderef:if (ISINT(op)) \
              IntCode \
           else if (ISREF(op)){ \
 	      NDEREF(op,nderef); \
   	      VarCode;} \
           else \
              OtherCode; \
/*
#define SWITCH_OP_INT(op,nderef,VarCode,IntCode,OtherCode) \
    nderef:if (ISREF(op)){ \
 	      NDEREF(op,nderef); \
   	      VarCode;} \
           else if (ISINT(op)) \
              IntCode \
           else \
              OtherCode; \
*/

/*
#define SWITCH_OP_ATM(op,nderef,VarCode,AtmCode,SuspCode) \
 nderef: if (IS_VAR_OR_STRUCT(op)){ \
           if (ISREF(op)){ \
	      NDEREF(op,nderef); \
   	      VarCode;} \
           else if (op<0) \
              SuspCode;} \
           AtmCode
*/
#define SWITCH_OP_ATM(op,nderef,VarCode,AtmCode,SuspCode) \
   DEREF2(op,VarCode); \
   if (TAG(op)==ATM) { \
        AtmCode; \
   } else if (op<0){ \
        SuspCode; \
   }

#define SWITCH_OP_LST(op,nderef,VarCode,LstCode,SuspCode) {\
 nderef: if (ISREF(op)){ \
            NDEREF(op,nderef); \
   	    VarCode;} \
         else if (ISLIST(op)) \
            LstCode \
         else if (IS_SUSP_VAR(op)) \
            SuspCode;}

#define SWITCH_OP_NIL(op,nderef,VarCode,NilCode,SuspCode) \
 nderef: if (ISREF(op)){ \
            NDEREF(op,nderef); \
   	    VarCode;} \
         else if (ISNIL(op)) \
            NilCode \
         else if (IS_SUSP_VAR(op)) \
            SuspCode

#define SWITCH_OP_LST_NIL(op,nderef,VarCode,LstCode,NilCode,SuspCode) \
 nderef:  if (ISREF(op)){ \
            NDEREF(op,nderef); \
   	    VarCode;} \
          else if (ISLIST(op)) \
            LstCode \
         else if (ISNIL(op)) \
            NilCode \
         else if (IS_SUSP_VAR(op)) \
            SuspCode

/*
#define SWITCH_OP_VAR(op,nderef,VarCode,SuspCode,OtherCode) \
       if (ISREF(op)){ \
            op = FOLLOW(op); \
            nderef: if (ISREF(op)){ \
	        NDEREF(op,nderef); \
   	        VarCode;} \
            else if (IS_SUSP_VAR(op)) \
              SuspCode \
            else OtherCode;} \
         else OtherCode

#define SWITCH_OP_VAR(op,nderef,VarCode,SuspCode,OtherCode) \
     nderef: DEREF2(op,VarCode); \
     if (IS_SUSP_VAR(op)){ \
        SuspCode;\
     } else { \
        OtherCode; \
     }
*/

#ifdef M64BITS
#define BP_HASH_CODE1(op,hashcode,lab){					\
  SWITCH_OP(op,lab,							\
	    {hashcode = 0;},						\
	    {hashcode = ((op & HASH_BITS)>>2);},			\
	    {hashcode = (ISLIST(op)) ? list_psc_hashcode : 0;}, \
	    {hashcode = ((BPLONG)GET_STR_SYM_REC(op) & HASH_BITS)>>3;}, \
	    {hashcode = 0;});						\
  }
#else
#define BP_HASH_CODE1(op,hashcode,lab){					\
  SWITCH_OP(op,lab,							\
	    {hashcode = 0;},						\
	    {hashcode = ((op & HASH_BITS)>>2);},			\
	    {hashcode = (ISLIST(op)) ? list_psc_hashcode : 0;}, \
	    {hashcode = ((BPLONG)GET_STR_SYM_REC(op) & HASH_BITS)>>2;}, \
	    {hashcode = 0;});						\
  }
#endif

#ifdef M64BITS
#define BP_HASH_KEY1_CODE1(op,key,hashcode,lab){			\
  SWITCH_OP(op,lab,							\
	    {key = BP_ZERO; hashcode = 0;},				\
	    {key = op; hashcode = ((op & HASH_BITS)>>2);},		\
	    {if (ISLIST(op)){key = list_psc_int; hashcode = list_psc_hashcode;} else {key = BP_ZERO; hashcode = 0;}}, \
	    {SYM_REC_PTR sym_ptr = GET_STR_SYM_REC(op); key = ADDTAG((BPLONG)sym_ptr,INT_TAG); hashcode = ((BPLONG)sym_ptr & HASH_BITS)>>3;}, \
	    {key = BP_ZERO; hashcode = 0;});				\
  }
#else
#define BP_HASH_KEY1_CODE1(op,key,hashcode,lab){			\
  SWITCH_OP(op,lab,							\
	    {key = BP_ZERO; hashcode = 0;},				\
	    {key = op; hashcode = ((op & HASH_BITS)>>2);},		\
	    {if (ISLIST(op)){key = list_psc_int; hashcode = list_psc_hashcode;} else {key = BP_ZERO; hashcode = 0;}}, \
	    {SYM_REC_PTR sym_ptr = GET_STR_SYM_REC(op); key = ADDTAG((BPLONG)sym_ptr,INT_TAG); hashcode = ((BPLONG)sym_ptr & HASH_BITS)>>2;}, \
	    {key = BP_ZERO; hashcode = 0;});				\
  }
#endif

#define BP_HASH_KEY1(op,key,lab){						\
    SWITCH_OP(op,lab,							\
	      {key = BP_ZERO;},						\
	      {key = op;},						\
	      {if (ISLIST(op)) key = list_psc_int; else key = BP_ZERO;}, \
	      {key = ADDTAG((BPLONG)GET_STR_SYM_REC(op),INT_TAG);},	\
	      {key = BP_ZERO;});					\
  }

#define SWITCH_OP_VAR(op,nderef,VarCode,SuspCode,OtherCode) \
 nderef: if (IS_VAR_OR_STRUCT(op)){ \
             if (ISREF(op)){ \
	        NDEREF(op,nderef); \
   	        VarCode;} \
             else if (op<0) \
              SuspCode \
             else OtherCode;} \
          else OtherCode

#define SWITCH_OP_STACK_VAR(op,nderef,VarCode,OtherCode) \
 nderef: if (ISREF(op)){ \
             if ((BPLONG_PTR)op>LOCAL_TOP){ \
	        NDEREF(op,nderef); \
   	        VarCode;} \
             else OtherCode;} \
          else OtherCode

#define TRIGGER_ON 0x1L
#define INTERRUPT 0x2L
#define EVENT_POOL_NONEMPTY 0x4L
#define USER_INTERRUPT 0x8L

#define INSERT_TRIGGER_var_ins(dv_ptr) {\
  if (DV_ins_cs(dv_ptr) != (BPLONG)nil_sym) { \
    if (trigger_no>=MAXTRIGGERS) quit("Event queue overflowed (ins)\n"); \
    else { \
      toam_signal_vec |= TRIGGER_ON; \
      triggeredCs[++trigger_no] = A_DV_ins_cs(dv_ptr); \
      event_flag[trigger_no] = EVENT_VAR_INS; }}}

#define INSERT_TRIGGER_dvar_ins(dv_ptr) {\
  if (DV_ins_cs(dv_ptr) != (BPLONG)nil_sym) { \
    if (trigger_no>=MAXTRIGGERS) quit("Event queue overflowed (ins)\n"); \
    else { \
     toam_signal_vec |= TRIGGER_ON; \
     triggeredCs[++trigger_no] = A_DV_ins_cs(dv_ptr); \
     event_flag[trigger_no] = EVENT_DVAR_INS; \
     triggering_frame[trigger_no] = NULL; }}}

#define INSERT_TRIGGER_minmax(dv_ptr) \
  if (DV_minmax_cs(dv_ptr) != (BPLONG)nil_sym) { \
    if (trigger_no>=MAXTRIGGERS) quit("Event queue overflowed (bound)\n"); \
    else { \
       toam_signal_vec |= TRIGGER_ON; \
       triggeredCs[++trigger_no] = A_DV_minmax_cs(dv_ptr); \
       triggering_frame[trigger_no] = NULL; \
       event_flag[trigger_no] = EVENT_DVAR_MINMAX; \
    }}

/* record the triggering frame also */
#define INSERT_TRIGGER_minmax_checkseed(dv_ptr,fp) \
  if (DV_minmax_cs(dv_ptr) != (BPLONG)nil_sym) { \
    if (trigger_no>=MAXTRIGGERS) quit("Event queue overflowed (bound1)\n"); \
    else { \
       toam_signal_vec |= TRIGGER_ON; \
       triggeredCs[++trigger_no] = A_DV_minmax_cs(dv_ptr); \
       event_flag[trigger_no] = EVENT_DVAR_MINMAX; \
       triggering_frame[trigger_no] = fp; \
    }}

/* Notice that agents watching dom_any events are also attached to DV_dom_cs */
#define INSERT_TRIGGER_dom(dv_ptr,elm) \
  if (DV_dom_cs(dv_ptr) != (BPLONG)nil_sym) { \
    if (trigger_no>=MAXTRIGGERS) quit("Event queue overflowed (dom)\n"); \
    else { \
       toam_signal_vec |= TRIGGER_ON; \
       triggeredCs[++trigger_no] = A_DV_dom_cs(dv_ptr); \
       event_flag[trigger_no] = EVENT_DVAR_DOM; \
       event_object[trigger_no] = elm;  \
    } \
  }

#define INSERT_TRIGGER_outer_dom0(dv_ptr,elm) \
    if (trigger_no>=MAXTRIGGERS) quit("Event queue overflowed (dom_any)\n"); \
    else { \
       toam_signal_vec |= TRIGGER_ON; \
       triggeredCs[++trigger_no] = A_DV_outer_dom_cs(dv_ptr); \
       event_flag[trigger_no] = EVENT_DVAR_OUTER_DOM; \
       event_object[trigger_no] = elm;  \
    }

#define INSERT_TRIGGER_outer_dom(dv_ptr,elm) \
  if (DV_outer_dom_cs(dv_ptr) != (BPLONG)nil_sym) { \
    INSERT_TRIGGER_outer_dom0(dv_ptr,elm);}

#define INSERT_TRIGGER_event(dv_ptr,elm) \
  if (DV_dom_cs(dv_ptr) != (BPLONG)nil_sym) { \
    if (trigger_no>=MAXTRIGGERS) quit("Event queue overflowed (general)\n"); \
    else { \
       toam_signal_vec |= TRIGGER_ON; \
       triggeredCs[++trigger_no] = A_DV_dom_cs(dv_ptr); \
       event_flag[trigger_no] = EVENT_GENERAL; \
       event_object[trigger_no] = elm;  \
    } \
  }

/***************************************************************************/
/* Macros for manipulating domain variables */
#define A_DV_var(dv_ptr) dv_ptr
#define DV_var(dv_ptr) FOLLOW(dv_ptr)

#define A_DV_type(dv_ptr) (dv_ptr+1)
#define DV_type(dv_ptr) FOLLOW(dv_ptr+1)

#define A_DV_bit_vector_ptr(dv_ptr) (dv_ptr+1)
#define DV_bit_vector_ptr(dv_ptr) FOLLOW(dv_ptr+1)
/*
   bit_vector_ptr points to a variable-sized structure of the following fields
   low_ptr
   up_ptr
   low_ptr : word
   ...
   up_ptr : word
*/

#define A_DV_ins_cs(dv_ptr) (dv_ptr+2)
#define DV_ins_cs(dv_ptr) FOLLOW(dv_ptr+2)

#define A_DV_attached(dv_ptr) (dv_ptr+3)
#define DV_attached(dv_ptr) FOLLOW(dv_ptr+3)

#define A_DV_size(dv_ptr) (dv_ptr+4)
#define DV_size(dv_ptr) FOLLOW(dv_ptr+4)

#define A_DV_first(dv_ptr) (dv_ptr+5)
#define DV_first(dv_ptr) FOLLOW(dv_ptr+5)

#define A_DV_last(dv_ptr) (dv_ptr+6)
#define DV_last(dv_ptr) FOLLOW(dv_ptr+6)

#define A_DV_minmax_cs(dv_ptr) (dv_ptr+7)
#define DV_minmax_cs(dv_ptr) FOLLOW(dv_ptr+7)

#define A_DV_dom_cs(dv_ptr) (dv_ptr+8)
#define DV_dom_cs(dv_ptr) FOLLOW(dv_ptr+8)

#define A_DV_outer_dom_cs(dv_ptr) (dv_ptr+9)
#define DV_outer_dom_cs(dv_ptr) FOLLOW(dv_ptr+9)

#define SIZE_OF_DV 10

#define BV_low_val(bv_ptr) FOLLOW(bv_ptr)

#define BV_up_val(bv_ptr) FOLLOW(bv_ptr+1)

#define BV_base_ptr(bv_ptr) (bv_ptr+2)


#define IT_DOMAIN   1
#define UN_DOMAIN   0

#ifdef LINUX
#define IS_BV_DOMAIN(dv_ptr)  (DV_type(dv_ptr)!=IT_DOMAIN && DV_type(dv_ptr)!=UN_DOMAIN)
#else
#define IS_BV_DOMAIN(dv_ptr)  ((BPULONG)DV_type(dv_ptr)>1)
#endif

/****************** bit vector **************************/
/*
   WORD_NUMBER(val)
    val              word_number
   32..63            1
   0..31             0
   -32..-1           -1
*/
#define WORD_NUMBER(val) (val>=0) ? (val/NBITS_IN_LONG) : -(-val+NBITS_IN_LONG-1)/NBITS_IN_LONG


#define WORD_OFFSET(bv_ptr,elm,w,w_ptr,offset){ \
   offset = elm-BV_low_val(bv_ptr); \
   w_ptr = (BPLONG_PTR)BV_base_ptr(bv_ptr) + offset/NBITS_IN_LONG; \
   offset = offset % NBITS_IN_LONG; \
   w = FOLLOW(w_ptr); }

#define NEXT_IN_WORD(from,w,w_ptr,offset,mask){ \
    mask = (MASK_FF << offset); \
    if ((w & mask)==0L){		\
      w_ptr++; \
      w = FOLLOW(w_ptr); \
      from = from + NBITS_IN_LONG - offset; \
      offset = 0L; \
      while (w==0L){ \
         w_ptr++; \
         w = FOLLOW(w_ptr); \
         from = from + NBITS_IN_LONG; \
      } \
    }	\
  }


#define  NEXT_IN_ELM(elm,w,offset,mask){ \
    while ((w & (0xffL << offset))==0){offset += 8; elm += 8;} \
    mask = (0x1L << offset);						\
    while (!(w & mask)){						\
      elm++;								\
      mask <<= 1;							\
      offset++;								\
    }									\
  }

/* elm is the next element that is in the domain, mask is the mask for the elm */
#define BV_NEXT_IN(bv_ptr,elm,w,w_ptr,offset,mask){ \
  WORD_OFFSET(bv_ptr,elm,w,w_ptr,offset); \
  NEXT_IN_WORD(elm,w,w_ptr,offset,mask); \
  NEXT_IN_ELM(elm,w,offset,mask); \
}

#define PREV_IN_WORD(elm,w,w_ptr,offset,mask){ \
    mask = (MASK_FF >> (NBITS_IN_LONG-1-offset)); \
    while ((w & mask) == 0){ \
      w_ptr--; \
      w = FOLLOW(w_ptr); \
      elm = elm - offset -1; \
      offset = NBITS_IN_LONG-1; \
      mask = MASK_FF; \
     } \
   }

#define  PREV_IN_ELM(elm,w,offset,mask){ \
  mask = (0x1L << offset); \
  while ((w & mask) ==0){  \
    elm--; \
    mask >>= 1; \
 } \
}

#define BV_PREV_IN(bv_ptr,elm,w,w_ptr,offset,mask){ \
  WORD_OFFSET(bv_ptr,elm,w,w_ptr,offset); \
  PREV_IN_WORD(elm,w,w_ptr,offset,mask); \
  PREV_IN_ELM(elm,w,offset,mask); \
}

/********************************************/
#define IS_IT_DOMAIN(dv_ptr)  (DV_type(dv_ptr)==IT_DOMAIN)
#define IS_UN_DOMAIN(dv_ptr)  (DV_type(dv_ptr)==UN_DOMAIN)

#define TRAIL_VAR 0
#define TRAIL_VAL_ATOMIC 1
#define TRAIL_VAL_NONATOMIC 2
#define TRAIL_BIT_VECTOR 3

#define PREPARE_NUMBER_TERM(num) { \
    global_var_num = num; \
}

#define PREPARE_UNNUMBER_TERM(ptr) {					\
    global_unnumbervar_max = -1;					\
    global_unnumbervar_watermark = global_unnumbervar_ptr = ptr;	\
  }

/* a must be positive */
#define UP_DIV(V,X,a){ \
    if (X>=0) { \
        V = X/a; \
        if (V*a != X) V++; \
    } else { \
        V = X/a; \
    } \
}

#define LOW_DIV(V,X,a){ \
    if (X>=0) { \
      V = X/a; \
    } else { \
      V = X/a; \
      if (V*a!=X) V--; \
    } \
}


