#ifndef SAVE_EXPL_GRAPH_H
#define SAVE_EXPL_GRAPH_H

enum SaveFormat{
	FormatJson=0,
	FormatPb=1,
	FormatPbTxt=2,
	FormatHDF5=3,
};

int pc_prism_save_2(void);

#endif /* SAVE_EXPL_GRAPH_H */
