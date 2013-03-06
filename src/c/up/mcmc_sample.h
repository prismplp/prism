#ifndef MCMC_SAMPLE_H
#define MCMC_SAMPLE_H

extern SW_COUNT_PTR **mh_state_sw_count;
extern SW_COUNT_PTR *prop_state_sw_count;
extern SW_COUNT_PTR **smp_val_sw_count;
extern int *mh_state_sw_count_size;
extern int prop_state_sw_count_size;
extern int *smp_val_sw_count_size;

extern int *mh_switch_table;

extern double *stored_params;
extern double *stored_hparams;
extern int num_stored_params;

extern int num_observed_goals;
extern int *observed_goals;
extern int num_samples;

extern double logV0;

extern int trace_sw_id;
extern int trace_step;
extern int postparam_sw_id;
extern int postparam_step;

void clean_sampled_values(void);
void clean_mh_state_sw_count(void);
void release_mh_occ_switches(void);

void alloc_mh_switch_table(void);
void clean_mh_switch_table(void);

void init_mh_sw_tab(void);
double log_dirichlet_Z(int);
double log_dirichlet_pdf_value(void);

int loop_mh(int, int, int);

#endif
