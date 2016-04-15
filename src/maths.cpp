/**
 * @file maths.cpp
 * @brief Impl of math
 *
 * @author misaka-10032 (longqic@andrew.cmu.edu)
 */

#include "maths.h"
#include "hblas/halide_blas.h"

namespace halstm {

  inline int sgemm(bool transA, bool transB, float a, buffer_t *A, buffer_t *B, float b, buffer_t *C) {
    return halide_sgemm(transA, transB, a, A, B, b, C);
  }

}