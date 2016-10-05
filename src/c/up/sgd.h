#ifndef __SGD_H__
#define __SGD_H__
#include "up/up.h"

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



#endif /* __SGD_H__ */

