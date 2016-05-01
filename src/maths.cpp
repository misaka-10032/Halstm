/**
 * @file maths.cpp
 * @brief Impl of math
 *
 * @author misaka-10032 (longqic@andrew.cmu.edu)
 */

#include "maths.h"

namespace halstm {

  Halide::Func matrix_dot(bool RA, bool RB, bool transA, bool transB,
                          Func& A, Func& B, int M, int N, int dim_k){
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

//  Halide::Func matrix_add_2d(Halide::Func& m1, Halide::Func& m2){
//    Halide::Func result("Matrix_Add_2d");
//    Halide::Var i("madd_i"), j("madd_j");
//
////    printf("m1's dimension: %d\n", m1.dimensions());
////    printf("m2's dimension: %d\n", m2.dimensions());
//
//    result(i, j) = m1(i, j) + m2(i, j);
//
//    return result;
//  }

  Halide::Func matrix_add_3d(Halide::Func& m1, Halide::Func& m2){
    Halide::Func result("Matrix_Add_3d");
    Halide::Var i, j, k;

    result(i, j, k) = m1(i, j, k) + m2(i, j, k);

    return result;
  }

  // Elementwise matrix multiplication
//  Halide::Func matrix_mul(Halide::Func& m1,Halide::Func& m2){
//    Halide::Func result("Matrix_Mul");
//
//    Halide::Var i, j;
//
//    result(i, j) = m1(i, j) * m2 (i, j);
//
//    //scheduling
//    result.parallel(j);
//    result.vectorize(i, 4);
//
//    return result;
//  }

  void matrix_add_2d(Func& m0, int m0offx, int m0offy,
                     Func& m1, int m1offx, int m1offy,
                     Func& m2, int m2offx, int m2offy,
                     RDom&& range) {
    m0(range.x + m0offx, range.y + m0offy) =
        m1(range.x + m1offx, range.y + m1offy) + m2(range.x + m2offx, range.y + m2offy);
  }

  void matrix_mul_2d(Func& m0, int m0offx, int m0offy,
                     Func& m1, int m1offx, int m1offy,
                     Func& m2, int m2offx, int m2offy,
                     RDom&& range) {
    m0(range.x + m0offx, range.y + m0offy) =
        m1(range.x + m1offx, range.y + m1offy) * m2(range.x + m2offx, range.y + m2offy);
  }

  /*
   * Get a instance of tanh activation function
   * - input: the last stage Halide Func in pipeline
   */
  void Tanh_2d(Halide::Func& input, RDom &&range) {
    input(range.x, range.y) = Halide::tanh(input(range.x, range.y));
  }

  /*
   * Get a instance of sigmoid activation function
   * - input: the last stage Halide Func in pipeline
   */
  void Sigmoid_2d(Halide::Func &input, RDom &&range) {
    input(range.x, range.y) =
        1.0f / (1.0f + Halide::fast_exp(-input(range.x, range.y)));
  }

}
