/* -*- c-basic-offset: 4 ; tab-width: 4 -*- */

#ifndef MP_H
#define MP_H

/*-------------------------------------------------------------------------*/

#include <mpi.h>

/* MS-MPI uses a non-null sentinel when receive status is not needed. */
#if defined(PRISM_MINGW) && PRISM_MINGW == 1 && defined(__MINGW32__)
#define MP_IGNORE_STATUS MPI_STATUS_IGNORE
#else
#define MP_IGNORE_STATUS NULL
#endif

/*-------------------------------------------------------------------------*/

#define TAG_GOAL_REQ   (1)
#define TAG_GOAL_LEN   (2)
#define TAG_GOAL_STR   (3)

#define TAG_SWITCH_REQ (4)
#define TAG_SWITCH_RES (5)

/*-------------------------------------------------------------------------*/

#endif /* MP_H */
