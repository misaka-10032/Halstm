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
                 const Var& x, const Var& y, int rsize, Func &C,
                 bool schedule) {
    Func A_("A_"), B_("B_");
    A_(x, y) = transA ? A(y, x) : A(x, y);
    B_(x, y) = transB ? B(y, x) : B(x, y);

    if (schedule) {
      Func AA("AA"), BB("BB");
      Var xi("xi"), xo("xo"), yi("yi"), yo("yo"), xy("xy");
      AA(xi, yi, xo, yo) = A_(xi + xo * TILE_SZ, yi + yo * TILE_SZ);
      BB(xi, yi, xo, yo) = B_(xi + xo * TILE_SZ, yi + yo * TILE_SZ);
      AA.fuse(xo, yo, xy).parallel(xy).compute_root();
      BB.fuse(xo, yo, xy).parallel(xy).compute_root();

      RDom rd(0, TILE_SZ, 0, rsize / TILE_SZ);
      Func CC("CC");
      CC(xi, xo, yi, yo) = 0.f;
      CC.fuse(yo, yi, y).parallel(y);
      CC(xi, xo, yi, yo) += BB(xi, rd.x, xo, rd.y) * AA(rd.x, yi, rd.y, yo);
      CC.update()
          .reorder({rd.x, xi, yi, rd.y, xo, yo})
          .parallel(yo).vectorize(xi);
      CC.compute_root();

      C(x, y) = CC(x % TILE_SZ, x / TILE_SZ, y % TILE_SZ, y / TILE_SZ);
      C.parallel(y);
    } else {
      // non-scheduled version is suitable for small matrices
      // baseline
      Var r("r");
      Func C_("C_");
      C_(x, y, r) = B_(x, r) * A_(r, y);
      C(x, y) = 0.f;
      C(x, y) += C_(x, y, RDom(0, rsize));
    }
  }

  /**
   * op(A): (rsize, ysize, _)
   * op(B): (_, rsize)
   * C:     (_, ysize, _)
   */
  void Dot_3dx2d(bool transA, bool transB, const Func& A, const Func& B,
                 const Var& x, const Var& y, const Var& z,
                 int rsize, int ysize, Func &C,
                 bool schedule) {
    Func A_("A_"), B_("B_");
    A_(x, y, z) = transA ? A(z, y, x) : A(x, y, z);
    B_(x, y) = transB ? B(y, x) : B(x, y);

    if (schedule) {
      Func AA("AA"), BB("BB");
      Var xi("xi"), xo("xo"), yi("yi"), yo("yo"), xy("xy");
      AA(xi, yi, xo, yo) = A_(xi + xo * TILE_SZ,
                              (yi + yo * TILE_SZ) % ysize,
                              (yi + yo * TILE_SZ) / ysize);
      BB(xi, yi, xo, yo) = B_(xi + xo * TILE_SZ, yi + yo * TILE_SZ);
      AA.fuse(xo, yo, xy).parallel(xy).compute_root();
      BB.fuse(xo, yo, xy).parallel(xy).compute_root();

      RDom rd(0, TILE_SZ, 0, rsize / TILE_SZ);
      Func CC("CC");
      CC(xi, xo, yi, yo) = 0.f;
      CC.fuse(yo, yi, y).parallel(y);
      CC(xi, xo, yi, yo) += BB(xi, rd.x, xo, rd.y) * AA(rd.x, yi, rd.y, yo);
      CC.update()
          .reorder({rd.x, xi, yi, rd.y, xo, yo})
          .parallel(yo).vectorize(xi);
      CC.compute_root();

      Var yz("yz");
      Func Cf("Cf");
      Cf(x, y) = CC(x % TILE_SZ, x / TILE_SZ, y % TILE_SZ, y / TILE_SZ);
      C(x, y, z) = Cf(x, y+ysize*z);
      C.fuse(y, z, yz).parallel(yz);
    } else {
      Var r("r");
      Func C_("C_");
      C(x, y, z) = 0.f;
      C_(r, x, y, z) = B_(x, r) * A_(r, y, z);
      C(x, y, z) += C_(RDom(0, rsize), x, y, z);
    }
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
