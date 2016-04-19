/**
 * @file maths.h
 * @brief Prototypes for maths.
 *
 * @author misaka-10032 (longqic@andrew.cmu.edu)
 */

#ifndef HALSTM_MATHS_H
#define HALSTM_MATHS_H

#include "Halide.h"

// void hal_gemm(bool transA, bool transB, int M, int N, int K,
// float alpha, Func& A, Func& B, float beta, Func& c);

namespace halstm {

    //currently operate on 2-D array
    //TODO1: for now transpose is not considered, the parameter must be adjusted by user
    //TODO2: need scheduling
    //TODO3: need to determine way to organize the matrix layout: column major or row major
    Halide::Func define_hal_gemm(bool transA, bool transB, int M, int N, int K,
                  float alpha, Halide::Func& A, Halide::Func& B, float beta, Halide::Func& C){
      Halide::Func result("result");

      const Halide::Expr num_rows = M;
      const Halide::Expr num_cols = N;
      const Halide::Expr sum_size = K;

      Halide::Func A_("A_"), B_("B_"), C_("C_");

      // temp var
      Halide::Var i, j;

      Halide::Var k("k");
      Halide::Func prod;

      // matrix multiplication AxB
      prod(k, i, j) = A(i, k) * B(k, j);
      Halide::Func AB("AB");
      Halide::RDom rv(0, sum_size);
      AB(i, j) += prod(rv, i, j);

      result(i, j) = (alpha * AB(i, j) + beta * C(i, j));

      return result;
    }
}
#endif // HALSTM_MATHS_H
