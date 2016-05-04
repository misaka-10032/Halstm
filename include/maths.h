/**
 * @file maths.h
 * @brief Prototypes for maths.
 *
 * @author misaka-10032 (longqic@andrew.cmu.edu)
 */

#ifndef HALSTM_MATHS_H
#define HALSTM_MATHS_H

#include "Halide.h"
using namespace Halide;

// void hal_gemm(bool transA, bool transB, int M, int N, int K,
// float alpha, Func& A, Func& B, float beta, Func& c);

namespace halstm {

  //operate on 2-D matrix
  Func matrix_dot(bool RA, bool RB, bool transA, bool transB,
                  Func& A, Func& B, int M, int N, int dim_k);

//  Func matrix_add_2d(Func& m1, Func& m2);

  Func matrix_add_3d(Func& m1, Func& m2);

  // Elementwise matrix multiplication
  Func matrix_mul(Func& m1, Func& m2);

  // m0 <- m1 + m2
  void matrix_add_2d(Func& m0, int m0offx, int m0offy,
                     Func& m1, int m1offx, int m1offy,
                     Func& m2, int m2offx, int m2offy,
                     RDom&& range);

  // m0 <- m1 * m2
  void matrix_mul_2d(Func& m0, int m0offx, int m0offy,
                     Func& m1, int m1offx, int m1offy,
                     Func& m2, int m2offx, int m2offy,
                     RDom&& range);

  // in <- tanh(in)
  void Tanh_2d(Halide::Func& input, RDom &&range);

  // in <- sigmoid(in)
  void Sigmoid_2d(Halide::Func &input, RDom &&range);


}

#endif // HALSTM_MATHS_H
