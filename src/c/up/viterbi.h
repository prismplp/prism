#ifndef VITERBI_H
#define VITERBI_H

int pc_compute_viterbi_5(void);
int pc_compute_n_viterbi_3(void);
int pc_compute_n_viterbi_rerank_4(void);

void compute_max(void);
void compute_n_max(int);

void get_only_nth_most_likely_path(int, int, TERM *);

void compute_n_crf_max(int);/*[D-PRISM]*/
void get_n_most_likely_path(int,int,TERM *);
void get_most_likely_path(int,TERM *,TERM *,TERM *,double *);
void get_n_most_likely_path_rerank(int , int , int,TERM *);

#endif /* VITERBI_H */


