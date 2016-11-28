#ifndef __RANK_H__
#define __RANK_H__

/*
   ranking table
   ROOT (up/up.h) is not used in rank learning
 */
typedef struct RankNode *RNK_NODE_PTR;
struct RankNode{
	int goal_count;
	int* goals;
	RNK_NODE_PTR next;
};

int pc_rank_learn_7(void);
int pc_set_goal_rank_1(void);
int pc_clear_goal_rank_0(void);

#endif /* __RANK_H__ */

