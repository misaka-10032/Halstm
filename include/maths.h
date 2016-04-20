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

    //operate on 2-D matrix
    //TODO: add scheduling
    //TODO: Figure out how to encapsulate this to avoid user from freaking out
    Halide::Func define_matrix_dot(Halide::Func& A, Halide::Func& B, int dim_k){
      Halide::Func result("Matrix_Dot");
      Halide::Expr sum_size = dim_k;

      // temp var
      Halide::Var i, j;

      // matrix multiplication AxB
      Halide::Var k("k");
      Halide::Func prod;
      // Express all the products we need to do a matrix multiply as a 3D Func.
      prod(k, j, i) = A(k, i) * B(j, k);

      Halide::RDom rv(0, sum_size);
      Halide::Func AB("AB");
      AB(j, i) += prod(rv, j, i);
      result(j, i) = AB(j, i);

      //scheduling
      //(*result).bound(i, 0, num_rows).bound(j, 0, num_cols);
      //result.output_buffer().set_bounds(0, 0, num_rows).set_bounds(1, 0, num_cols);
      return result;
    }

    Halide::Func* define_matrix_add(Halide::Func& m1, Halide::Func& m2, int M, int N){
      Halide:: Func* result = new Halide::Func("Matrix_Add");

      Halide::Var i, j;

      (*result)(i, j) = m1(i, j) + m2(i, j);
      //scheduling

      (*result).parallel(j);
      //result.vectorize(i, 4);
      return result;
    }
}
#endif // HALSTM_MATHS_H
