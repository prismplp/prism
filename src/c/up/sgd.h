#ifndef __SGD_H__
#define __SGD_H__
#include "up/up.h"

void initialize_sgd_weights(void);
void initialize_parent_switch(void);
int update_sgd_weights(int iterate);
int update_sgd_params(void);
typedef struct ExplMinibatch* MinibatchPtr;
typedef struct ExplMinibatch Minibatch;
struct ExplMinibatch{
	ROOT* roots;
	int num_roots;
	EG_NODE_PTR* egraph;
	int* egraph_count;
	int egraph_size;
	int batch_size;
	int count;
};

enum SGDOptimizer{
	OPTIMIZER_SGD=0,
	OPTIMIZER_ADADELTA=1,
	OPTIMIZER_ADAM=2,
};


#endif /* __SGD_H__ */

