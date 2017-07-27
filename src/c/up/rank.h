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
typedef struct ExplRankMinibatch *RankMinibatchPtr;
typedef struct ExplRankMinibatch RankMinibatch;
struct ExplRankMinibatch{
	RNK_NODE_PTR* roots;
	int num_roots;
	EG_NODE_PTR* egraph;
	int* egraph_count;
	int egraph_size;
	int batch_size;
	int count;
};

enum RankLoss{
	RANK_LOSS_HINGE=0,
	RANK_LOSS_SQUARE=1,
	RANK_LOSS_EXP=2,
	RANK_LOSS_LOG=3,
};



int pc_rank_learn_7(void);
int pc_set_goal_rank_1(void);
int pc_clear_goal_rank_0(void);

#endif /* __RANK_H__ */

