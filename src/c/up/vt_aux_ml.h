#ifndef VT_AUX_ML_H
#define VT_AUX_ML_H

void count_occ_sws_aux(EG_PATH_PTR, int);
void count_occ_sws(void);
int examine_likelihood(void);
double compute_vt_likelihood(void);
int update_vt_params(void);
int update_vt_params_smooth(void);

#endif /* VT_AUX_ML_H */

