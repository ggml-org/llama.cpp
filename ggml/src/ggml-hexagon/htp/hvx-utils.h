#ifndef HVX_UTILS_H
#define HVX_UTILS_H

#include "hex-utils.h"

#include "hvx-types.h"
#include "hvx-copy.h"
#include "hvx-scale.h"
#include "hvx-exp.h"
#include "hvx-inverse.h"
#include "hvx-reduce.h"
#include "hvx-sigmoid.h"
#include "hvx-sqrt.h"
#include "hvx-arith.h"
#include "hvx-div.h"
#include "hvx-base.h"

#ifndef GATHER_TYPE
#    ifdef __hexagon__
#        define GATHER_TYPE(_a) (uint32_t) (_a)
#    else
#        define GATHER_TYPE(_a) (_a)
#    endif
#endif

#if defined(__hexagon__)
#    define SCATTER_TYPE(_a) (intptr_t) _a
#else
#    define SCATTER_TYPE(_a) (HVX_Vector *) _a
#endif

#endif /* HVX_UTILS_H */
