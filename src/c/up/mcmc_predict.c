#include "up/mcmc.h"
#include "up/mcmc_sample.h"
#include "up/mcmc_eml.h"

int num_predict_goals = 0;

static int *num_candidates = NULL;
static double ***sampledVs = NULL;
static SW_COUNT_PTR ***cand_expl_sw_count = NULL; /* candidates used in reranking */
static int **cand_expl_sw_count_size = NULL;

/*--------------[ Prediction ]------------------------------------------------------*/
/* Bayes predication with reranking                                                 */
/*       OGs = observed goals                                                       */
/*       PGs = goals seeking for viterbi groundings and expls                       */
/* ViterbiEG = argmax_{E:Es for PG} estimated_log(P(E|PG,OGs))                      */
/*           = argmax_{E:Es for PG} estimated_log(P(E|OGs))                         */
/*   where                                                                          */
/*   Es = PG's candidate expls for reranking computed by n-viterbig using theta*,   */
/*   estimated P(E|OGs) = T^{-1} sum_{t} P(E|E_all^t) such that                     */
/*   E_all^t = state of M-H MCMC for P(E_all|OGs) at t step                         */
/*   theta* = average params of posterior Dirichlet learned by VB                   */
/*----------------------------------------------------------------------------------*/

static void clean_candidate_expls(void)
{
    int i,j,k;

    for (i=0; i<num_predict_goals; i++) {
        for (j=0; j<num_candidates[i]; j++) {
            for (k=0; k<cand_expl_sw_count_size[i][j]; k++) {
                FREE(cand_expl_sw_count[i][j][k]);
            }
            FREE(cand_expl_sw_count[i][j]);
        }
        FREE(cand_expl_sw_count[i]);
        FREE(cand_expl_sw_count_size[i]);
    }

    FREE(cand_expl_sw_count);
    FREE(cand_expl_sw_count_size);
    FREE(num_candidates);
}

static int alloc_candidate_expls_aux(V_ENT_PTR ent_ptr)
{
    int i,sw_ins_id,index;
    int num_occ_sw_ins=0;
    EG_PATH_PTR path_ptr;

    path_ptr = ent_ptr->path_ptr;
    for (i=0; i<path_ptr->sws_len; i++) {
        sw_ins_id = path_ptr->sws[i]->id;
        mh_switch_table[sw_ins_id]++;
        if (mh_switch_table[sw_ins_id] == 1) {
            num_occ_sw_ins++;
        }
    }
    for (i=0; i<path_ptr->children_len; i++) {
        index = ent_ptr->top_n_index[i];
        if (path_ptr->children[i]->top_n != NULL) {
            num_occ_sw_ins += alloc_candidate_expls_aux(path_ptr->children[i]->top_n[index]);
        }
    }

    return num_occ_sw_ins;
}

/* scan the top_n explanations and create the count table on the candidates for reranking */
static void alloc_candidate_expls(int *prediction_goals, int n)
{
    int i,j,k,l,g_id;
    EG_NODE_PTR eg_ptr;
    V_ENT_PTR ent_ptr;

    cand_expl_sw_count = (SW_COUNT_PTR ***)MALLOC(num_predict_goals * sizeof(SW_COUNT_PTR **));
    num_candidates = (int *)MALLOC(num_predict_goals * sizeof(int));
    cand_expl_sw_count_size = (int **)MALLOC(num_predict_goals * sizeof(int *));

    for (i=0; i<num_predict_goals; i++) {
        g_id = prediction_goals[i];
        eg_ptr = expl_graph[g_id];
        num_candidates[i] = eg_ptr->top_n_len;

        cand_expl_sw_count[i] = (SW_COUNT_PTR **)MALLOC(num_candidates[i] * sizeof(SW_COUNT_PTR *));
        cand_expl_sw_count_size[i] = (int *)MALLOC(num_candidates[i] * sizeof(int));
        for (j=0; j<num_candidates[i]; j++) {
            ent_ptr = eg_ptr->top_n[j];
            init_mh_sw_tab();
            cand_expl_sw_count_size[i][j] = alloc_candidate_expls_aux(ent_ptr);
            cand_expl_sw_count[i][j] = (SW_COUNT_PTR *)MALLOC(cand_expl_sw_count_size[i][j] * sizeof(SW_COUNT_PTR));
            k = 0;
            for (l=0; l<sw_ins_tab_size; l++) {
                if (mh_switch_table[l]!=0) {
                    cand_expl_sw_count[i][j][k] = (SW_COUNT_PTR)MALLOC(sizeof(struct SwitchCount));
                    cand_expl_sw_count[i][j][k]->sw_ins_id = l;
                    cand_expl_sw_count[i][j][k]->count = mh_switch_table[l];
                    k++;
                }
            }
        }
    }
}

static double log_prod_i_prediction(int PG_index, int can_index)
{
    int i,j,sw_ins_id,occ_flag = 0;
    double log_prod_is = 0.0;
    SW_INS_PTR sw_ptr;

    for (i=0; i<sw_tab_size; i++) {
        sw_ptr = switches[i];
        while (sw_ptr != NULL) {
            sw_ins_id = sw_ptr->id;
            for (j=0; j<cand_expl_sw_count_size[PG_index][can_index]; j++) {
                if (sw_ins_id == cand_expl_sw_count[PG_index][can_index][j]->sw_ins_id) {
                    occ_flag = 1;
                    break;
                }
            }
            if (occ_flag == 1) {
                log_prod_is += log_dirichlet_Z(switches[i]->id);
                occ_flag = 0;
                break;
            }
            sw_ptr = sw_ptr->next;
        }
    }
    return log_prod_is;
}

static void add_candidate_counts(int PG_index, int can_index)
{
    int i,sw_ins_id;

    for (i=0; i<cand_expl_sw_count_size[PG_index][can_index]; i++) {
        sw_ins_id = cand_expl_sw_count[PG_index][can_index][i]->sw_ins_id;
        switch_instances[sw_ins_id]->count += cand_expl_sw_count[PG_index][can_index][i]->count;
    }
}

static void subtract_candidate_counts(int PG_index, int can_index)
{
    int i,sw_ins_id;

    for (i=0; i<cand_expl_sw_count_size[PG_index][can_index]; i++) {
        sw_ins_id = cand_expl_sw_count[PG_index][can_index][i]->sw_ins_id;
        switch_instances[sw_ins_id]->count -= cand_expl_sw_count[PG_index][can_index][i]->count;
    }
}

static void comp_all_sampled_values_prediction_aux(int t)
{
    int i,j;
    double log_prod_is1,log_prod_is2;

    for (i=0; i<num_predict_goals; i++) {
        for (j=0; j<num_candidates[i]; j++) {
            log_prod_is2 = log_prod_i_prediction(i,j);
            add_candidate_counts(i,j);
            log_prod_is1 = log_prod_i_prediction(i,j);
            subtract_candidate_counts(i,j);
            sampledVs[i][j][t] = log_prod_is1 - log_prod_is2;
        }
    }
}

static void comp_all_sampled_values_prediction(void)
{
    int i,j,k;

    comp_all_sampled_values_prediction_aux(num_samples);

    for (i=num_samples-1; i>=0; i--) {
        if (smp_val_sw_count[i]==NULL) {
            for (j=0; j<num_predict_goals; j++) {
                for (k=0; k<num_candidates[j]; k++) {
                    sampledVs[j][k][i] = sampledVs[j][k][i+1];
                }
            }
        }
        else {
            return_state(i);
            comp_all_sampled_values_prediction_aux(i);
        }
    }
}

static void comp_mcmc_value_prediction(PR_ANS_PTR *ans)
{
    int i,j;
    PR_ANS_PTR temp,next,ans_ptr;
    double logAVP;

    for (i=0; i<num_predict_goals; i++) {
        ans[i] = (PR_ANS_PTR)MALLOC(sizeof(struct PredictAns));
        ans[i]->next = NULL;
        for (j=0; j<num_candidates[i]; j++) {
            ans_ptr = (PR_ANS_PTR)MALLOC(sizeof(struct PredictAns));
            ans_ptr->rank = j;
            logAVP = log_avg(sampledVs[i][j], (num_samples + 1));
            ans_ptr->logP = logAVP;
            temp = ans[i];
            while (temp->next != NULL && temp->next->logP > logAVP) {
                temp = temp->next;
            }
            next = temp->next;
            temp->next = ans_ptr;
            ans_ptr->next = next;
        }
    }
}

void mh_predict(int *prediction_goals, int n, PR_ANS_PTR *ans)
{
    int i,j;

    /* restore the parameters learned by VB */
    restore_preserved_params();

    /* restore the last sample */
    add_last_state_counts();

    /* Compute the candidates Es for reranking by n-Viterbi */
    compute_n_max(n);

    alloc_mh_switch_table();
    alloc_candidate_expls(prediction_goals,n);

    sampledVs = (double ***)MALLOC(num_predict_goals * sizeof(double **));
    for (i=0; i<num_predict_goals; i++) {
        sampledVs[i] = (double **)MALLOC(num_candidates[i] * sizeof(double *));
        for (j=0; j<num_candidates[i]; j++) {
            sampledVs[i][j] = (double *)MALLOC((num_samples + 1) * sizeof(double));
        }
    }

    comp_all_sampled_values_prediction();
    comp_mcmc_value_prediction(ans);

    clean_mh_switch_table();
    clean_candidate_expls();

    release_occ_switches();
    release_num_sw_vals();
}
