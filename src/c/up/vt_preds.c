/* -*- c-basic-offset: 4 ; tab-width: 4 -*- */

/*------------------------------------------------------------------------*/

#include "bprolog.h"
#include "up/up.h"
#include "up/em_aux.h"
#include "up/em_aux_ml.h"
#include "up/em_aux_vb.h"
#include "up/vt.h"
#include "up/vt_ml.h"
#include "up/vt_vb.h"
#include "up/util.h"

/*------------------------------------------------------------------------*/

/* mic.c (B-Prolog) */
NORET myquit(BPLONG, char *);

/*------------------------------------------------------------------------*/

int pc_prism_vt_4(void) {
	struct VT_Engine vt_eng;

	RET_ON_ERR(check_smooth(&vt_eng.smooth));
	RET_ON_ERR(run_vt(&vt_eng));

	return
	    bpx_unify(bpx_get_call_arg(1,4), bpx_build_integer(vt_eng.iterate   )) &&
	    bpx_unify(bpx_get_call_arg(2,4), bpx_build_float  (vt_eng.lambda    )) &&
	    bpx_unify(bpx_get_call_arg(3,4), bpx_build_float  (vt_eng.likelihood)) &&
	    bpx_unify(bpx_get_call_arg(4,4), bpx_build_integer(vt_eng.smooth    )) ;
}

int pc_prism_vbvt_2(void) {
	struct VBVT_Engine vbvt_eng;

	RET_ON_ERR(check_smooth_vb());
	RET_ON_ERR(run_vbvt(&vbvt_eng));

	return
	    bpx_unify(bpx_get_call_arg(1,2), bpx_build_integer(vbvt_eng.iterate)) &&
	    bpx_unify(bpx_get_call_arg(2,2), bpx_build_float(vbvt_eng.free_energy));
}

int pc_prism_both_vt_2(void) {
	struct VBVT_Engine vbvt_eng;

	RET_ON_ERR(check_smooth_vb());
	RET_ON_ERR(run_vbvt(&vbvt_eng));

	get_param_means();

	return
	    bpx_unify(bpx_get_call_arg(1,2), bpx_build_integer(vbvt_eng.iterate)) &&
	    bpx_unify(bpx_get_call_arg(2,2), bpx_build_float(vbvt_eng.free_energy));
}
