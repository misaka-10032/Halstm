/**
 * @file maths.h
 * @brief Prototypes for maths.
 *
 * @author misaka-10032 (longqic@andrew.cmu.edu)
 */

#ifndef HALSTM_MATHS_H
#define HALSTM_MATHS_H

#include <Halide.h>
using namespace Halide;

namespace halstm {
  void hal_gemm(bool transA, bool transB, int M, int N, int K,
                float alpha, Func &A, Func &B, float beta, Func &C);
}

#endif // HALSTM_MATHS_H
