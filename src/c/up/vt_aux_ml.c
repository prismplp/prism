/* -*- c-basic-offset: 2; tab-width: 8 -*- */

/*------------------------------------------------------------------------*/

#include "bprolog.h"
#include "up/up.h"
#include "up/graph.h"
#include "up/util.h"
#include "up/flags.h"

/*------------------------------------------------------------------------*/

void count_occ_sws_aux(EG_PATH_PTR path_ptr,int count)
{
    int i;

    if(path_ptr!=NULL) {
        for(i=0; i<path_ptr->children_len; i++) {
            count_occ_sws_aux(path_ptr->children[i]->max_path,count);
        }
        for(i=0; i<path_ptr->sws_len; i++) {
            path_ptr->sws[i]->count += count;
        }
    }
}

void count_occ_sws(void)
{
    int i,root_id;
    SW_INS_PTR ptr;

    for(i=0; i<occ_switch_tab_size; i++) {
        ptr = occ_switches[i];
        while(ptr!=NULL) {
            ptr->count = 0;
            ptr = ptr->next;
        }
    }

    for(i=0; i<num_roots; i++) {
        root_id = roots[i]->id;
        count_occ_sws_aux(expl_graph[root_id]->max_path,roots[i]->count);
    }
}

/*------------------------------------------------------------------------*/

int examine_likelihood(void)
{
    SW_INS_PTR ptr;
    int i;

    for(i = 0; i < occ_switch_tab_size; i++) {
        ptr = occ_switches[i];
        while(ptr!=NULL) {
            if(ptr->count == 0 && ptr->inside < TINY_PROB) {
                emit_error("Parameter being zero -- %s",prism_sw_ins_string(ptr->id)); //FIXME:error message
                RET_ERR(err_underflow);
            }
            ptr = ptr->next;
        }
    }

    return BP_TRUE;
}

/*------------------------------------------------------------------------*/

double compute_vt_likelihood(void)
{
    double likelihood = 0.0;
    SW_INS_PTR ptr;
    int i;

    for(i = 0; i < occ_switch_tab_size; i++) {
        ptr = occ_switches[i];
        while(ptr!=NULL) {
            likelihood += ptr->count * log(ptr->inside);
            ptr = ptr->next;
        }
    }

    return likelihood;
}

/*------------------------------------------------------------------------*/

int update_vt_params(void)
{
    int i;
    SW_INS_PTR ptr;
    double sum;

    for (i = 0; i < occ_switch_tab_size; i++) {
        ptr = occ_switches[i];
        sum = 0.0;
        while (ptr != NULL) {
            ptr = ptr->next;
        }
        if (sum != 0.0) {
            ptr = occ_switches[i];
            if (ptr->fixed > 0) continue;
            while (ptr != NULL) {
                if (ptr->fixed == 0) ptr->inside = ptr->count / sum;
                if (log_scale && ptr->inside < TINY_PROB) {
                    emit_error("Parameter being zero (-inf in log scale) -- %s",
                               prism_sw_ins_string(ptr->id));
                    RET_ERR(err_underflow);
                }
                ptr = ptr->next;
            }
        }
    }

    return BP_TRUE;
}

int update_vt_params_smooth(void)
{
    int i;
    SW_INS_PTR ptr;
    double sum;

    for (i = 0; i < occ_switch_tab_size; i++) {
        ptr = occ_switches[i];
        sum = 0.0;
        while (ptr != NULL) {
            sum += ptr->count + ptr->smooth;
            ptr = ptr->next;
        }
        if (sum != 0.0) {
            ptr = occ_switches[i];
            if (ptr->fixed > 0) continue;
            while (ptr != NULL) {
                if (ptr->fixed == 0)
                    ptr->inside = (ptr->count + ptr->smooth) / sum;
                ptr = ptr->next;
            }
        }
    }

    return BP_TRUE;
}

/*------------------------------------------------------------------------*/
