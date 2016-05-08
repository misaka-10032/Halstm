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

    Func AA("AA"), BB("BB");
    Var xi("xi"), xo("xo"), yi("yi"), yo("yo");
    AA(xi, xo, yi, yo) = A_(xi+xo*TILE_SZ, yi+yo*TILE_SZ);
    BB(xi, xo, yi, yo) = B_(xi+xo*TILE_SZ, yi+yo*TILE_SZ);
    RDom rd(0, TILE_SZ, 0, rsize/TILE_SZ);
    Func C_("C_");
    C_(xi, xo, yi, yo) = 0.f;
    C_(xi, xo, yi, yo) += BB(xi, xo, rd.x, rd.y) * AA(rd.x, rd.y, yi, yo);
    C_.fuse(yo, yi, y).parallel(y);
    C_.update()
        .reorder({rd.x, xi, yi, rd.y, xo, yo})
        .parallel(yo);
    C_.compute_root();

    C(x, y) = C_(x%TILE_SZ, x/TILE_SZ, y%TILE_SZ, y/TILE_SZ);
    C.parallel(y);
//    C.print_loop_nest();

//    Var r("r");
//    Func C_("C_");
//    //RDom rd(0, rsize);
//    C_(x, y, r) = B_(x, r) * A_(r, y);
//    C(x, y) += C_(x, y, r);
//    //C.parallel(rd.x);
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
