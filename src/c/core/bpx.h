#ifndef BPX_H
#define BPX_H

#include "bprolog.h"
#include "stuff.h"

/*====================================================================*/

#define NULL_TERM ((TERM)(0))

/*--------------------------------*/

/* These are the safer versions of DEREF and NDEREF macros.           */

#define XDEREF(op) \
	do { if(TAG(op) || (op) == FOLLOW(op)) { break; } (op) = FOLLOW(op); } while(1)
#define XNDEREF(op, label) \
	do { if(TAG(op) || (op) == FOLLOW(op)) { break; } (op) = FOLLOW(op); goto label; } while(1)

/*--------------------------------*/

/* This low-level macro provides more detailed information about the  */
/* type of a given term than TAG(op).                                 */

#define XTAG(op) ((op) & TAG_MASK)

#define REF0 0x0L
#define REF1 TOP_BIT
#define INT  INT_TAG
#define NVAR TAG_MASK

/*--------------------------------*/

/* The following macros are the same as IsNumberedVar and NumberVar  */
/* respectively, provided just for more consistent naming.           */

#define IS_NVAR(op) ( ((op) & TAG_MASK) == NVAR )
#define MAKE_NVAR(id) ( (((BPLONG)(id)) << 2) | NVAR )

/*--------------------------------*/

/* This macro is redefined to reduce warnings on GCC 4.x.            */

#if defined LINUX && ! defined M64BITS
#undef  UNTAGGED_ADDR
#define UNTAGGED_ADDR(op) ( (((BPLONG)(op)) & VAL_MASK0) | addr_top_bit )
#endif

/*====================================================================*/

bool        bpx_is_var(TERM);
bool        bpx_is_atom(TERM);
bool        bpx_is_integer(TERM);
bool        bpx_is_float(TERM);
bool        bpx_is_nil(TERM);
bool        bpx_is_list(TERM);
bool        bpx_is_structure(TERM);
bool        bpx_is_compound(TERM);
bool        bpx_is_unifiable(TERM, TERM);
bool        bpx_is_identical(TERM, TERM);

TERM        bpx_get_call_arg(BPLONG, BPLONG);

BPLONG      bpx_get_integer(TERM);
double      bpx_get_float(TERM);
const char* bpx_get_name(TERM);
int         bpx_get_arity(TERM);
TERM        bpx_get_arg(BPLONG, TERM);
TERM        bpx_get_car(TERM);
TERM        bpx_get_cdr(TERM);

TERM        bpx_build_var(void);
TERM        bpx_build_integer(BPLONG);
TERM        bpx_build_float(double);
TERM        bpx_build_atom(const char *);
TERM        bpx_build_list(void);
TERM        bpx_build_nil(void);
TERM        bpx_build_structure(const char *, BPLONG);

bool        bpx_unify(TERM, TERM);
int         bpx_compare(TERM, TERM);

TERM        bpx_string_2_term(char *);
char*       bpx_term_2_string(TERM);

int         bpx_call_term(TERM);
int         bpx_call_string(char *);
int         bpx_mount_query_term(TERM);
int         bpx_mount_query_string(char *);
int         bpx_next_solution(void);

void        bpx_write(TERM);

int         bpx_printf(const char *, ...);

/*====================================================================*/

#endif /* BPX_H */
