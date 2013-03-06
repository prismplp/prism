#ifndef MCMC_EML_H
#define MCMC_EML_H

double mh_marginal(void);
void return_state(int);
void add_last_state_counts(void);
void restore_preserved_params(void);
double log_avg(double[], int);

#endif

