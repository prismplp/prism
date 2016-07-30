#ifndef CRF_AUX_H
#define CRF_AUX_H
void initialize_lambdas_noisy_uniform(void);
void initialize_lambdas_random(void);
void initialize_lambdas_zero(void);
void initialize_lambdas(void);
void set_visited_flags(EG_NODE_PTR node_ptr);
void initialize_visited_flags(void);
void count_complete_features(EG_NODE_PTR node_ptr, int count);
void initialize_crf_count(void);
int compute_gradient_scaling_none(void);
int compute_gradient_scaling_log_exp(void);
double compute_log_likelihood_scaling_none(void);
double compute_log_likelihood_scaling_log_exp(void);
int compute_crf_probs_scaling_none(void);
int compute_crf_probs_scaling_log_exp(void);
int update_lambdas(double tmp_epsilon);
void save_current_params(void);
void restore_current_params(void);
double compute_gf_sd(void);
double compute_gf_sd_LBFGS(void);
double compute_phi_alpha(CRF_ENG_PTR crf_ptr, double alpha);
double compute_phi_alpha_LBFGS(CRF_ENG_PTR crf_ptr, double alpha);
double line_search(CRF_ENG_PTR crf_ptr, double alpha0, double rho, double c1, double likelihood, double gf_sd);
double line_search_LBFGS(CRF_ENG_PTR crf_ptr, double alpha0, double rho, double c1, double likelihood, double gf_sd);
double golden_section(CRF_ENG_PTR crf_ptr,double a, double b);
double golden_section_LBFGS(CRF_ENG_PTR crf_ptr,double a, double b);
void initialize_LBFGS(void);
void clean_LBFGS(void);
void compute_hessian(int iterate);
void restore_old_gradient(void);
void initialize_LBFGS_q(void);
void compute_LBFGS_y_rho(void);
int update_lambdas_LBFGS(double tmp_epsilon);
int run_grd(CRF_ENG_PTR crf_ptr);
void config_crf(CRF_ENG_PTR crf_ptr);
void restart_LBFGS(void);

#endif /*CRF_AUX_H */

