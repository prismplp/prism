#define AR_AR_ADDR(ar)                  ar
#define AR_CPS_ADDR(ar)                 (ar-1)
#define AR_TOP_ADDR(ar)                 (ar-2)
#define AR_BTM_ADDR(ar)                 (ar-3)
#define AR_B_ADDR(ar)                   (ar-4)
#define AR_CPF_ADDR(ar)                 (ar-5)
#define AR_H_ADDR(ar)                   (ar-6)
#define AR_T_ADDR(ar)                   (ar-7)
#define AR_SF_ADDR(ar)                  (ar-8)

#define AR_REEP_ADDR(ar)                (ar-4)
#define AR_PREV_ADDR(ar)                (ar-5)
#define AR_STATUS_ADDR(ar)              (ar-6)
#define AR_OUT_ADDR(ar)                 (ar-7)

#define AR_AR(ar)                  *ar
#define AR_CPS(ar)                 *(ar-1)
#define AR_TOP(ar)                 *(ar-2)
#define AR_BTM(ar)                 *(ar-3)
#define AR_B(ar)                   *(ar-4)
#define AR_CPF(ar)                 *(ar-5)
#define AR_H(ar)                   *(ar-6)
#define AR_T(ar)                   *(ar-7)
#define AR_SF(ar)                  *(ar-8)

#define AR_REEP(ar)                *(ar-4)
#define AR_PREV(ar)                *(ar-5)
#define AR_STATUS(ar)              *(ar-6)
#define AR_OUT(ar)                 *(ar-7)

#define AR_STATUS_STAMP(F) UNTAGGED3(AR_STATUS(F))

#define AR_PRIORITY(ar)            0


#define AR_SUBGOAL_TABLE(ar)       *(ar-9)
#define GET_AR_SUBGOAL_TABLE(ar)   (*(ar-9) & VAL_MASK1)
#define SET_AR_SUBGOAL_TABLE(ar,st) *(ar-9) = st
#define AR_CURR_ANSWER(ar)         *(ar-10)

#define AR_THREAD_ID(ar)           FOLLOW((BPLONG_PTR)AR_TOP(ar)+1)

#define AR_TABLE_NEW_BITS(ar)    *(ar-11)

/************************************************/
#define FLAT_FRAME_TAG    0x0L
#define FLAT_B_FRAME_TAG  0x1L
#define SUSP_FRAME_TAG    0x2L
#define TEMP_FRAME_TAG    0x3L

#ifdef M64BITS
#define CATCHER_FRAME_TAG        0x8000000000000001L
#define NONDET_FRAME_TAG         0x8000000000000002L
#define CATCHER_NONDET_FRAME_TAG 0x8000000000000003L
#define TABLE_FRAME_TAG          0x8000000000000001L
#else
#define CATCHER_FRAME_TAG        0x80000001L
#define NONDET_FRAME_TAG         0x80000002L
#define CATCHER_NONDET_FRAME_TAG 0x80000003L
#define TABLE_FRAME_TAG          0x80000001L
#endif

/* A catcher frame is a nondet frame that needs special treatment by the cut.
   It's a tabled frame, or a frame for catch/3 or call_cleanup/2
*/
#define IS_CATCHER_FRAME(f) ((AR_BTM(f) & CATCHER_FRAME_TAG)==CATCHER_FRAME_TAG)
#define IS_NONDET_FRAME(f) ((AR_BTM(f) & NONDET_FRAME_TAG)==NONDET_FRAME_TAG)
#define IS_SUSP_FRAME(f) ((AR_BTM(f) & TAG_MASK)==SUSP_FRAME_TAG)
#define IS_TABLE_FRAME(f) ((AR_BTM(f) & TAG_MASK)==TABLE_FRAME_TAG)
#define IS_FLAT_B_FRAME(f) ((AR_BTM(f) & TAG_MASK)==FLAT_B_FRAME_TAG)
#define IS_FLAT_FRAME(f) ((AR_BTM(f) & TAG_MASK)==FLAT_FRAME_TAG)

#define NO_RESERVED_SLOTS(f,no){				\
    switch((AR_BTM(f) & TAG_MASK)){				\
    case FLAT_FRAME_TAG: no = 4; break;				\
    case FLAT_B_FRAME_TAG: no = 5; break;			\
    case SUSP_FRAME_TAG: no = SUSP_FRAME_SIZE; break;		\
    case NONDET_FRAME_TAG:					\
    case CATCHER_NONDET_FRAME_TAG: no = 9; break;		\
    case TABLE_FRAME_TAG:  no = 13; break;			\
    }								\
  }

#define FLAT_FRAME_SIZE 4
#define FLAT_B_FRAME_SIZE 5
#define SUSP_FRAME_SIZE 8
#define NONDET_FRAME_SIZE 9
#define TABLE_FRAME_SIZE 13

/************************************************
subgoalTable -> bucket[0]
                 ...
       	        bucket[subgoalTableBucketSize-1]
subgoalTableEntry ->
                NEXT
                ANSWER_TABLE
                ANSWER_TABLE_FIRST0 // very beginning of the answer list
                ANSWER_TABLE_FIRST1 // beginning of the new answers for the last round
                ANSWER_TABLE_FIRST2 // beginning of the new answers for the current round
		STATE           // looping or normal or revised
                MASTER_AR       // point to the frame of the master. If
                                // complete, MASTER_AR becomes SUBGOAL_COMPLETE
		SCC_ROOT        // root subgoal of the SCC this subgoal belongs to
		SCC_ELMS        // list of nodes in the SCC rooted at this node
                SYM
                args

   The STATE slot stores the following information
   ...  4           3           2          1         0
   -----------------------------------------------------------
      promoted | looping | evaluated  | iteration | revised   |
   ------------------------------------------------------------
    iteration -- 1: iteration; 0: normal
    evaluated - 1: the subgoal has been evaluated; 0: not evaluated
    looping -- 1: a looping node 0: not a looping node

    The information on master/slave is not stored. It is possible to
    distinguish masters from slaves. Slaves only execute the
    table_use_answer instruction while masters execute table_check_competion
    after a round of execution.
*/
#define SUBGOAL_NEW_REGION_SET_BIT 0x10L
#define SUBGOAL_LOOPING_BIT 0x8L
#define SUBGOAL_EVALUATED_BIT 0x4L
#define SUBGOAL_ITERATION_BIT 0x2L
#define SUBGOAL_ANS_REVISED_BIT 0x1L

#define SUBGOAL_COMPLETE 0x1L
#define SUBGOAL_TEMP_COMPLETE 0x2L

/************************************************/
#define GT_NEXT_ADDR(tab) (BPLONG_PTR)tab
#define GT_NEXT(tab) FOLLOW((BPLONG_PTR)tab)
#define GT_ANSWER_TABLE(tab) FOLLOW((BPLONG_PTR)(tab)+1)
#define GT_ANSWER_TABLE_FIRST0(tab) ANSWERTABLE_FIRST(GT_ANSWER_TABLE(tab))
#define GT_STATE(tab) FOLLOW((BPLONG_PTR)(tab)+2)
#define GT_TOP_AR(tab) FOLLOW((BPLONG_PTR)(tab)+3)
#define GT_SCC_ROOT(tab) FOLLOW((BPLONG_PTR)(tab)+4)
#define GT_SCC_ELMS(tab) FOLLOW((BPLONG_PTR)(tab)+5)
#define GT_SYM(tab) FOLLOW((BPLONG_PTR)(tab)+6)
#define GT_ARG_ADDR(tab) ((BPLONG_PTR)(tab)+7)
#define GT_RECORD_SIZE 7

#define AR_IS_SCC_ROOT(ar,subgoal_entry) (GT_TOP_AR((BPLONG_PTR)GT_SCC_ROOT(subgoal_entry)) == (BPLONG)ar)

#define SUBGOAL_IS_ITERATION(subgoal_entry) (GT_STATE(subgoal_entry) & SUBGOAL_ITERATION_BIT)
#define SUBGOAL_START_ITERATION(subgoal_entry) GT_STATE(subgoal_entry) = (SUBGOAL_ITERATION_BIT | SUBGOAL_LOOPING_BIT)
#define SUBGOAL_START_NORMAL(subgoal_entry) GT_STATE(subgoal_entry) = 0

#define SET_SUBGOAL_EVALUATED(subgoal_entry) GT_STATE(subgoal_entry) = GT_STATE(subgoal_entry) | SUBGOAL_EVALUATED_BIT
#define SUBGOAL_EVALUATED(subgoal_entry) (GT_STATE(subgoal_entry) & SUBGOAL_EVALUATED_BIT)

#define SET_SUBGOAL_LOOPING(subgoal_entry)				\
  if (GT_STATE(subgoal_entry) & SUBGOAL_LOOPING_BIT);			\
  else									\
    {GT_STATE(subgoal_entry) = GT_STATE(subgoal_entry) | SUBGOAL_LOOPING_BIT;}
#define SUBGOAL_IS_LOOPING(subgoal_entry) (GT_STATE(subgoal_entry) & SUBGOAL_LOOPING_BIT)

#define SET_SUBGOAL_ANS_REVISED(subgoal_entry) GT_STATE(subgoal_entry) = GT_STATE(subgoal_entry) | SUBGOAL_ANS_REVISED_BIT;

#define SUBGOAL_ANS_IS_REVISED(subgoal_entry) (GT_STATE(subgoal_entry) & SUBGOAL_ANS_REVISED_BIT)

#define SUBGOAL_IS_COMPLETE(subgoal_entry) (GT_TOP_AR(subgoal_entry) == SUBGOAL_COMPLETE)

#define MARK_SUBGOAL_NEW_REGION_IS_SET(subgoal_entry) GT_STATE(subgoal_entry) = GT_STATE(subgoal_entry) | SUBGOAL_NEW_REGION_SET_BIT
#define SUBGOAL_NEW_REGION_IS_SET(subgoal_entry) (GT_STATE(subgoal_entry) & SUBGOAL_NEW_REGION_SET_BIT)

/* a subgoal is in its own scc */
#define AllocateSubgoalTableEntry(ptr,arity,sym){	\
    AllocateFromTableArea(ptr,arity+GT_RECORD_SIZE);	\
  }

#define InitializeSubgoalTableEntry(ptr,sym){		\
    GT_ANSWER_TABLE(ptr) = (BPLONG)NULL;			\
    GT_STATE(ptr) = 0;						\
    GT_TOP_AR(ptr) = (BPLONG)NULL;				\
    GT_SYM(ptr) = (BPLONG)sym;					\
    GT_SCC_ELMS(ptr) = (BPLONG)NULL;				\
    GT_SCC_ROOT(ptr) = (BPLONG)ptr;				\
  }

#define TABLE_FREE_CELL(ptr) {FOLLOW(ptr+1) = (BPLONG)table_free_cells_ptr;table_free_cells_ptr=ptr;}


/************************************************
answerTable ->  SIZE
 	        FIRST
	        LAST
                COUNT
                BUCKET_PTR

BUCKET_PTR  ->  bucket[0]
                 ...
       	        bucket[SIZE-1]
************************************************/
#define ANSWERTABLE_BUCKET_SIZE(tab) FOLLOW((BPLONG_PTR)(tab))
#define ANSWERTABLE_FIRST(tab) FOLLOW((BPLONG_PTR)(tab)+1)
#define ANSWERTABLE_FIRST_ADDR(tab) ((BPLONG_PTR)(tab)+1)
#define ANSWERTABLE_LAST_ADDR(tab) ((BPLONG_PTR)(tab)+2)
#define ANSWERTABLE_LAST(tab) FOLLOW((BPLONG_PTR)(tab)+2)
#define ANSWERTABLE_COUNT(tab) FOLLOW((BPLONG_PTR)(tab)+3)
#define ANSWERTABLE_BUCKET_PTR(tab) FOLLOW((BPLONG_PTR)(tab)+4)

#define ANSWERTABLE_RECORD_SIZE 5

/************************************************
   answer -> next_in_table
          -> next_in_chain
          -> A1
          -> ...
	  -> An
************************************************/
#define ANSWER_NEXT_IN_TABLE(answer) FOLLOW((BPLONG_PTR)(answer))
#define ANSWER_NEXT_IN_TABLE_ADDR(answer) (answer)
#define ANSWER_NEXT_IN_CHAIN(answer) FOLLOW((BPLONG_PTR)(answer)+1)
#define ANSWER_NEXT_IN_CHAIN_ADDR(answer) (answer+1)
#define ANSWER_ARG_ADDR(answer) ((BPLONG_PTR)(answer)+2)

#define TABLE_MODE_BITS(ep) FOLLOW(ep+6)
#define TABLE_MODE_OPT_ARG(ep) FOLLOW(ep+7)
#define TABLE_MODE_CARDG(ep) FOLLOW(ep+8)
