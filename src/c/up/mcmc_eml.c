#include "up/mcmc.h"
#include "up/mcmc_sample.h"

static double *logVs = NULL;

/*------[estimate marginal log-likelihood]----------------------*/
/*                                                              */
/* Chib's idea (1995) + Rao-Blackwellization                    */
/* P(G|As) = P(theta*|As)P(G|theta*)/P(theta*|G)                */
/* P(theta*|G)                                                  */
/*      =  \sum_e P(theta*|e)P(e|G)                             */
/*      = T^{-1} (P(theta*|e_1)+..+ P(theta*|e_T))              */
/*            where e_i ~ P(e|G)                                */
/* theta* = average params of posterior Dirichlet learned by VB */
/*--------------------------------------------------------------*/

void return_state(int index) {
	int i,sw_ins_id;

	for (i=0; i<smp_val_sw_count_size[index]; i++) {
		sw_ins_id = smp_val_sw_count[index][i]->sw_ins_id;
		switch_instances[sw_ins_id]->count -= smp_val_sw_count[index][i]->count;
	}
}

static void comp_all_sampled_values(void) {
	int i;

	logVs[num_samples] = log_dirichlet_pdf_value();

	for (i=num_samples-1; i>=0; i--) {
		if (smp_val_sw_count[i]==NULL) {
			logVs[i] = logVs[i+1];
		} else {
			return_state(i);
			logVs[i] = log_dirichlet_pdf_value();
		}
	}
}

void add_last_state_counts(void) {
	int i,j;

	for (i=0; i<sw_ins_tab_size; i++) {
		switch_instances[i]->count = 0;
	}
	for (i=0; i<num_observed_goals; i++) {
		for (j=0; j<mh_state_sw_count_size[i]; j++) {
			switch_instances[mh_state_sw_count[i][j]->sw_ins_id]->count += mh_state_sw_count[i][j]->count;
		}
	}
}

void restore_preserved_params(void) {
	int i;

	for (i=0; i<num_stored_params; i++) {
		switch_instances[i]->inside = stored_params[i];
		switch_instances[i]->smooth_prolog = stored_hparams[i];
	}
}

double log_avg(double logVs[], int size) {
	int i;
	double logMax,avg;

	logMax = logVs[0];

	for (i=1; i<size; i++) {
		if (logMax<logVs[i]) {
			logMax = logVs[i];
		}
	}

	avg = 0.0;
	for (i=0; i<size; i++) {
		avg += exp(logVs[i]-logMax);
	}

	avg /= size;

	return logMax + log(avg);
}

double mh_marginal(void) {
	double logV,eml;

	logVs = (double *)MALLOC((num_samples + 1) * sizeof(double));

	/* restore the parameters learned by VB */
	restore_preserved_params();

	/* restore the last sample */
	add_last_state_counts();

	/* restore the samples backward to compute P(theta*|e_T),....,P(theta*|e_1) */
	comp_all_sampled_values();

	logV = log_avg(logVs, (num_samples + 1));
	eml = logV0 - logV;

	FREE(logVs);

	return eml;
}
