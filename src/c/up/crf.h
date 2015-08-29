/* -*- c-basic-offset: 4 ; tab-width: 4 -*- */

#ifndef __CRF_GRAD_H__
#define __CRF_GRAD_H__

/*------------------------------------------------------------------------*/

#define DEFAULT_MAX_ITERATE (10000)

/*------------------------------------------------------------------------*/

struct CRF_Engine {
	double  likelihood;         /* [out] log likelihood */
	int     iterate;            /* [out] number of iterations */

	/* Functions called during computation. */
	int     (* compute_feature         )(void);
	int     (* compute_crf_probs       )(void);
	double  (* compute_likelihood     )(void);
	int     (* compute_gradient       )(void);
	int     (* update_lambdas         )(double);
};

typedef struct CRF_Engine   * CRF_ENG_PTR;

/*------------------------------------------------------------------------*/

#define SHOW_PROGRESS(n)                                                \
    do {                                                                \
        if(!verb_em && em_message > 0 && (n) % em_progress == 0) {      \
            if((n) % (em_progress * 10) == 0)                           \
                prism_printf("%d", n);                                  \
            else                                                        \
                prism_printf(".");                                      \
        }                                                               \
    } while (0)

#define SHOW_PROGRESS_HEAD(str, r)                                      \
    do {                                                                \
        if(num_restart > 1) {                                           \
            if(verb_em)                                                 \
                prism_printf("<<<< RESTART #%d >>>>\n", r);             \
            else if(em_message > 0)                                     \
                prism_printf("[%d] ", r);                               \
        }                                                               \
        if(!verb_em && em_message > 0)                                  \
            prism_printf("%s: ", str);                                  \
    } while (0)

#define SHOW_PROGRESS_TAIL(converged, n, x)                             \
    do {                                                                \
        const char *str =                                               \
            converged ? "Converged" : "Stopped";                        \
                                                                        \
        if(verb_em)                                                     \
            prism_printf("* %s (%.9f)\n", str, x);                      \
        else if(em_message > 0)                                         \
            prism_printf("(%d) (%s: %.9f)\n", n, str, x);               \
    } while (0)

#define SHOW_PROGRESS_TEMP(x)                                           \
    do {                                                                \
        if(verb_em)                                                     \
            prism_printf("* Temperature = %.3f\n", x);                  \
        else if(em_message > 0 && show_itemp)                           \
            prism_printf("<%.3f>", x);                                  \
        else if(em_message > 0)                                         \
            prism_printf("*");                                          \
    } while (0)

#define SHOW_PROGRESS_INTR()                                            \
    do {                                                                \
        if(verb_em)                                                     \
            prism_printf("* Interrupted\n");                            \
        else if(em_message > 0)                                         \
            prism_printf("(Interrupted)\n");                            \
    } while (0)

#define REACHED_MAX_ITERATE(n)                                          \
    ((max_iterate == -1 && (n) >= DEFAULT_MAX_ITERATE) ||               \
     (max_iterate >= +1 && (n) >= max_iterate))

/*------------------------------------------------------------------------*/

#endif /* __CRF_GRD_H__ */
