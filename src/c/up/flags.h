#ifndef FLAGS_H
#define FLAGS_H

/*========================================================================*/

int pc_set_daem_1(void);
int pc_set_em_message_1(void);
int pc_set_em_progress_1(void);
int pc_set_error_on_cycle_1(void);
int pc_set_explicit_empty_expls_1(void);
int pc_set_fix_init_order_1(void);
int pc_set_init_method_1(void);
int pc_set_itemp_init_1(void);
int pc_set_itemp_rate_1(void);
int pc_set_log_scale_1(void);
int pc_set_max_iterate_1(void);
int pc_set_mcmc_message_1(void);
int pc_set_mcmc_progress_1(void);
int pc_set_num_restart_1(void);
int pc_set_prism_epsilon_1(void);
int pc_set_show_itemp_1(void);
int pc_set_std_ratio_1(void);
int pc_set_verb_em_1(void);
int pc_set_verb_graph_1(void);
int pc_set_warn_1(void);
int pc_set_debug_level_1(void);
int pc_set_annealing_weight_1(void);
int pc_set_crf_learning_rate_1(void);
int pc_set_crf_epsilon_1(void);
int pc_set_crf_golden_b_1(void);
int pc_set_crf_init_method_1(void);
int pc_set_crf_learn_mode_1(void);
int pc_set_crf_ls_rho_1(void);
int pc_set_crf_ls_c1_1(void);
int pc_set_crf_penalty_1(void);
int pc_set_sgd_learning_rate_1(void);
int pc_set_sgd_penalty_1(void);
int pc_set_scc_debug_level_1(void);
int pc_set_sgd_adam_beta_1(void);
int pc_set_sgd_adam_gamma_1(void);
int pc_set_sgd_adam_epsilon_1(void);
int pc_set_sgd_adadelta_gamma_1(void);
int pc_set_sgd_adadelta_epsilon_1(void);
int pc_set_rank_loss_1(void);
int pc_set_rank_loss_c_1(void);

/*========================================================================*/

extern int     daem;
extern int     em_message;
extern int     em_progress;
extern int     error_on_cycle;
extern int     explicit_empty_expls;
extern int     fix_init_order;
extern int     init_method;
extern double  itemp_init;
extern double  itemp_rate;
extern int     log_scale;
extern int     max_iterate;
extern int     mcmc_message;
extern int     mcmc_progress;
extern int     num_restart;
extern double  prism_epsilon;
extern int     show_itemp;
extern double  std_ratio;
extern int     verb_em;
extern int     verb_graph;
extern int     warn;
extern int     debug_level;
extern double annealing_weight;
extern int     crf_learning_rate;
extern double  crf_epsilon;
extern double  crf_golden_b;
extern int     crf_init_method;
extern int     crf_learn_mode;
extern double  crf_ls_rho;
extern double  crf_ls_c1;
extern double  crf_penalty;
extern int     scc_debug_level;
extern int     sgd_optimizer;
extern double  sgd_learning_rate;
extern double  sgd_penalty;
extern int     num_minibatch;
extern double  sgd_adam_beta;
extern double  sgd_adam_gamma;
extern double  sgd_adam_epsilon;
extern double  sgd_adadelta_gamma;
extern double  sgd_adadelta_epsilon;
extern double  rank_loss_c;
extern int     rank_loss;

#endif /* FLAGS_H */
