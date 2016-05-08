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

  void Dot_2dx2d(bool transA, bool transB, const Func &A, const Func &B,
                 const Var &x, const Var &y, int rsize, Func &C,
                 bool schedule=true);

  void Dot_3dx2d(bool transA, bool transB, const Func &A, const Func &B,
                 const Var &x, const Var &y, const Var &z,
                 int rsize, int ysize, Func &C,
                 bool schedule=true);

  void Tanh_2d(RDom &&range, const Func& input, Func& output);

  void Sigmoid_2d(RDom &&range, const Func& input, Func& output);

  void Set_2d(RDom &&range, float v, Func &func);
}

#endif // HALSTM_MATHS_H
