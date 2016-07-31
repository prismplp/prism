#ifndef CRF_RANK_H
#define CRF_RANK_H

int pc_crf_rank_prepare_5(void);
int pc_crf_rank_learn_grd_2(void);

typedef struct {
	int goal_id;
} RankData;
typedef struct {
	int num_ranks;
	RankData* ranks;
} RankList;
extern int num_rank_lists;
extern RankList* rank_lists;
#endif /*CRF_RANK_H */

