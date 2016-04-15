/**
 * @file maths.h
 * @brief Prototypes for maths.
 *
 * @author misaka-10032 (longqic@andrew.cmu.edu)
 */

#ifndef HALSTM_MATHS_H
#define HALSTM_MATHS_H

#include "hblas/halide_blas.h"

namespace halstm {
  inline int halstm_sgemm(bool transA, bool transB, float a, buffer_t *A, buffer_t *B, float b, buffer_t *C);
}
#endif // HALSTM_MATHS_H
