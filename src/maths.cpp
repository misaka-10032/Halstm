/**
 * @file maths.cpp
 * @brief Impl of math
 *
 * @author misaka-10032 (longqic@andrew.cmu.edu)
 */

#include <Halide.h>
#include "maths.h"
#include "schedule.h"

namespace halstm {

  void Dot_2dx2d(bool transA, bool transB, const Func& A, const Func& B,
                 const Var& x, const Var& y, int rsize, Func &C) {
    Func A_("A_"), B_("B_");
    A_(x, y) = transA ? A(y, x) : A(x, y);
    B_(x, y) = transB ? B(y, x) : B(x, y);

    RDom rdom(0, rsize);
    Var xi("xi"), xo("xo"), yi("yi"), yo("yo");
    Func C_("C_");
    C_(x, y) = 0.f;
    C_(x, y) += B_(x, rdom) * A_(rdom, y);
    C(x, y) = C_(x, y);
    C_.compute_at(C, x).vectorize(x);
    C_.update().reorder(x, y, rdom).vectorize(x).unroll(y);
    C.tile(x, y, xi, yi, 16, 4)
        .vectorize(xi).unroll(yi).parallel(y);
//    C.print_loop_nest();

//    Var r("r");
//    Func C_("C_");
//    C_(r, x, y) = B_(x, r) * A_(r, y);
//    C(x, y) += C_(RDom(0, rsize), x, y);

  }

  void Dot_3dx2d(bool transA, bool transB, const Func& A, const Func& B,
                 const Var& x, const Var& y, const Var& z, int rsize, Func &C) {
    // TODO: schedule
    Var r("r");
    Func A_("A_"), B_("B_"), C_("C_");
    C(x, y, z) = (float) 0;
    A_(x, y, z) = transA ? A(z, y, x) : A(x, y, z);
    B_(x, y) = transB ? B(y, x) : B(x, y);
    C_(r, x, y, z) = B_(x, r) * A_(r, y, z);
    C(x, y, z) += C_(RDom(0, rsize), x, y, z);

    // scheduling
    C.compute_root();
    C.parallel(z);
  }

  /*
   * Get a instance of tanh activation function
   * - input: the last stage Halide Func in pipeline
   */
  void Tanh_2d(RDom&& range, const Func& input, Func& output) {
    output(range.x, range.y) = Halide::tanh(input(range.x, range.y));

    //scheduling
  }

  /*
   * Get a instance of sigmoid activation function
   * - input: the last stage Halide Func in pipeline
   */
  void Sigmoid_2d(RDom&& range, const Func& input, Func& output) {
    output(range.x, range.y) =
        1.0f / (1.0f + Halide::fast_exp(-input(range.x, range.y)));

    // scheduling
  }

  void Set_2d(RDom&& range, float v, Func &func) {
    func(range.x, range.y) = v;

    // scheduling
  }
}
