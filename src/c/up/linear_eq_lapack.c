#include "up/up.h"
#include "up/flags.h"
#include "lapacke.h"
#include "bprolog.h"

int lapack_solve(void);

// need to declare this function before
double bpx_get_float(TERM t);

int pc_solve_linear_system_4(void) {
	// get arguments
	TERM pr_n, pr_A, pr_b, pr_x;
	pr_n = bpx_get_call_arg(1,4); // first argument: vector size
	pr_A = bpx_get_call_arg(2,4); // second argument: matrix A
	pr_b = bpx_get_call_arg(3,4); // third argument: RHS vector b
	pr_x = bpx_get_call_arg(4,4); // fourth argument: solution x

	// transform to c types
	int n = bpx_get_integer(pr_n);
	double* A = (double*) malloc(n*n*sizeof(double));
	double* b = (double*) malloc(n*sizeof(double));
	int* ipiv = (int*) malloc(n*sizeof(int));

	int i, j;
	for(i = 0; i < n; i++) {
		b[i] = bpx_get_float(bpx_get_car(pr_b));
		pr_b = bpx_get_cdr(pr_b);
		for(j = 0; j < n; j++) {
			A[i*n + j] = bpx_get_float(bpx_get_car(pr_A));
			pr_A = bpx_get_cdr(pr_A);
		}
	}
#ifdef PREFIX_DEBUG
	for(i = 0; i < n; i++) {
		for(j = 0; j < n; j++) {
			printf("(%3.3f)",A[i*n + j]);
		}
		printf("\n");
	}
	for(i = 0; i < n; i++) {
		printf("(%3.3f)",b[i]);
	}
	printf("\n");
#endif


	// check for correct conversion
	if(exception) {
		return BP_FALSE;
	}

	// call lapack
	int info = LAPACKE_dgesv(LAPACK_ROW_MAJOR, n, 1, A, n, ipiv, b, 1);
	// check result
	if(info > 0) {
		// A singular, no solution
		return BP_FALSE;
	}

	// transform result back to prolog list
	for(i = 0; i < n; i++) {
		TERM nl = bpx_build_list();
		TERM car = bpx_get_car(nl);
		TERM cdr = bpx_get_cdr(nl);

		bpx_unify(car, bpx_build_float(b[i]));
		bpx_unify(pr_x, nl);
		pr_x = cdr;
	}
	bpx_unify(pr_x, bpx_build_nil());

	// check for correct conversion
	if(exception) {
		return BP_FALSE;
	}

#ifdef PREFIX_DEBUG
	for(i = 0; i < n; i++) {
		printf("(%3.3f)",b[i]);
	}
	printf("\n");
#endif

	// clean up
	free(A);
	free(b);
	free(ipiv);

	return BP_TRUE;
}

