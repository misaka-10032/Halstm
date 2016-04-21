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
    Halide::Func matrix_dot(bool RA, bool RB, bool transA, bool transB, Halide::Func& A, Halide::Func& B, int M, int N, int dim_k){
      Halide::Func result("Matrix_Dot");
      Halide::Expr sum_size = dim_k;

      // temp var
      Halide::Var i("i"), j("j"), r("r");

      Halide::Func A_("A_"), B_("B_");

      if(transA){
        if(RA) {
          A_(j, i, r) = A(i, j, r);
        }else{
          A_(j, i) = A(i, j);
        }
      }else{
        if(RA){
          A_(j, i, r) = A(j, i, r);
        }else{
          A_(j, i) = A(j, i);
        }
      }

      if(transB){
        if(RB){
          B_(j, i, r) = B(i, j, r);
        }else{
          B_(j, i) = B(i, j);
        }
      }else{
        if(RB){
          B_(j, i, r) = B(j, i, r);
        }else{
          B_(j, i) = B(j, i);
        }
      }

      Halide::Expr e = 0.0f;
      // matrix multiplication AxB
      Halide::Var k("k");
      Halide::Func prod("prod");
      Halide::RDom rv(0, sum_size);
      Halide::Func AB("AB");

      if (RA) {
        if(RB){
          // if RA and RB are both set, then they must be equal
          prod(k, j, i, r) = A_(k, i, r) * B_(j, k, r);
          result(j, i, r) += prod(rv, j, i, r);
        }else {
          prod(k, j, i, r) = A_(k, i, r) * B_(j, k);
          result(j, i, r) += (prod(rv, j, i, r));
        }
      }else if(RB){
        prod(k, j, i, r) = A_(k, i) * B_(j, k, r);
        result(j, i, r) += prod(rv, j, i, r);
      }else{
        prod(k, j, i) = A_(k, i) * B_(j, k);
        result(j, i) += prod(rv, j, i);
      }

      //scheduling
      //(*result).bound(i, 0, num_rows).bound(j, 0, num_cols);
      //result.output_buffer().set_bounds(0, 0, num_rows).set_bounds(1, 0, num_cols);
      return result;
    }

    Halide::Func matrix_add(bool R, Halide::Func& m1, Halide::Func& m2, int M, int N){
      Halide:: Func result("Matrix_Add");
      Halide::Var i, j, k;

      if(R) {
        result(i, j, k) = m1(i, j, k) + m2(i, j, k);
      }else{
        result(i, j) = m1(i, j) + m2(i, j);
      }


      return result;
    }

    // Elementwise matrix multiplication
    Halide::Func matrix_mul(Halide::Func& m1,Halide::Func& m2){
      Halide::Func result("Matrix_Mul");

      Halide::Var i, j;

      result(i, j) = m1(i, j) * m2 (i, j);

      //scheduling
      result.parallel(j);
      result.vectorize(i, 4);

      return result;
    }
}

#endif // HALSTM_MATHS_H
