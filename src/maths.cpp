/**
 * @file maths.cpp
 * @brief Impl of math
 *
 * @author misaka-10032 (longqic@andrew.cmu.edu)
 */

#include <Halide.h>
#include "maths.h"


namespace halstm {

  void Dot_2dx2d(bool transA, bool transB, Func &A, Func &B,
                 Var &x, Var &y, int rsize, Func &C) {
    Var r("r");
    Func A_("A_"), B_("B_"), C_("C_");
    C(x, y) = (float) 0;
    A_(x, y) = transA ? A(y, x) : A(x, y);
    B_(x, y) = transB ? B(y, x) : B(x, y);
    C_(r, x, y) = B_(x, r) * A_(r, y);
    C(x, y) += C_(RDom(0, rsize), x, y);


    // scheduling
    Var i, j, ii, ji, jii, iii, io, jo, t;
    Var ti[3], tj[3];
    int vec = 1;
    const int s = vec * 2;

//    C.tile(x, y, ti[1], tj[1], x, y, 2*s, 2*s);
//    C.tile(x, y, ii, ji, s, 4).tile(x, y, ti[0], tj[0], x, y, 1, s/4);
//    C.fuse(tj[1], ti[1], t).parallel(t);
//    C.rename(tj[0], t);
//    A_.compute_root()
//            .split(y, jo, ji, s)
//            .unroll(x).vectorize(ji).parallel(jo, 4);
//    B_.compute_at(C, t)
//            .tile(x, y, ii, ji, 8, 8)
//            .vectorize(ii).unroll(ji);
//    C.unroll(j).vectorize(x)
//            .update().vectorize(x);
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

    C_.compute_at(C, x);
    C_.parallel(z);
    C.parallel(z);
  }

  /*
   * Get a instance of tanh activation function
   * - input: the last stage Halide Func in pipeline
   */
  void Tanh_2d(RDom &&range, Func& input, Func& output) {
    output(range.x, range.y) = Halide::tanh(input(range.x, range.y));

    //scheduling
  }

  /*
   * Get a instance of sigmoid activation function
   * - input: the last stage Halide Func in pipeline
   */
  void Sigmoid_2d(RDom &&range, Func &input, Func &output) {
    output(range.x, range.y) =
        1.0f / (1.0f + Halide::fast_exp(-input(range.x, range.y)));

    // scheduling
  }

  void Set_2d(RDom &&range, float v, Func &func) {
    func(range.x, range.y) = v;

    // scheduling
  }
}
