/**
 * @file maths.cpp
 * @brief Impl of math
 *
 * @author misaka-10032 (longqic@andrew.cmu.edu)
 */

#include "maths.h"

namespace halstm {

  void Dot_2dx2d(bool transA, bool transB, Func &A, Func &B,
                 Var &x, Var &y, int rsize, Func &C) {
    // TODO: schedule
    Var r("r");
    Func A_("A_"), B_("B_"), C_("C_");
    C(x, y) = (float) 0;
    A_(x, y) = transA ? A(y, x) : A(x, y);
    B_(x, y) = transB ? B(y, x) : B(x, y);
    C_(r, x, y) = B_(x, r) * A_(r, y);
    C(x, y) += C_(RDom(0, rsize), x, y);
  }

  void Dot_3dx2d(bool transA, bool transB, Func &A, Func &B,
                 Var &x, Var &y, Var &z, int rsize, Func &C) {
    // TODO: schedule
    Var r("r");
    Func A_("A_"), B_("B_"), C_("C_");
    C(x, y, z) = (float) 0;
    A_(x, y, z) = transA ? A(z, y, x) : A(x, y, z);
    B_(x, y) = transB ? B(y, x) : B(x, y);
    C_(r, x, y, z) = B_(x, r) * A_(r, y, z);
    C(x, y, z) += C_(RDom(0, rsize), x, y, z);
  }

  /*
   * Get a instance of tanh activation function
   * - input: the last stage Halide Func in pipeline
   */
  void Tanh_2d(RDom &&range, Func& input, Func& output) {
    output(range.x, range.y) = Halide::tanh(input(range.x, range.y));
  }

  /*
   * Get a instance of sigmoid activation function
   * - input: the last stage Halide Func in pipeline
   */
  void Sigmoid_2d(RDom &&range, Func &input, Func &output) {
    output(range.x, range.y) =
        1.0f / (1.0f + Halide::fast_exp(-input(range.x, range.y)));
  }

  void Set_2d(RDom &&range, float v, Func &func) {
    func(range.x, range.y) = v;
  }
}
