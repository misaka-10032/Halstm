/**
 * @file test_matrix.h
 * @brief Prototypes for test_matrix.
 *
 * @author misaka-10032 (longqic@andrew.cmu.edu)
 */

#ifndef HALSTM_TEST_MATRIX_H
#define HALSTM_TEST_MATRIX_H

#include <cblas.h>
#include <cxxtest/TestSuite.h>
#include "maths.h"
#include "CycleTimer.h"
#include "utils.h"

typedef struct {
  float v[16];
} float16;

const int M = 256;
const int N = 256;
const int K = 512;

using namespace halstm;

class TestLstmLayer : public CxxTest::TestSuite {
public:
  void init(float* A, float* B, float* C,
            Image<float>& iA, Image<float>& iB, Image<float>& iC) {
    for (int i = 0; i < M*K; i++) {
      float r = static_cast<float> (rand()) / static_cast <float> (RAND_MAX);
      A[i] = r;
      iA.data()[i] = r;
    }
    for (int i = 0; i < K*N; i++) {
      float r = static_cast<float> (rand()) / static_cast <float> (RAND_MAX);
      B[i] = r;
      iB.data()[i] = r;
    }
    for (int i = 0; i < M*N; i++) {
      C[i] = 0.f;
      iC.data()[i] = 0.f;
    }
  }

  void _test_dot_2dx2d(bool transA, bool transB) {
    float* A = new float[M*K];
    float* B = new float[K*N];
    float* C = new float[M*N];
    Image<float> iA, iB, iC(M, N, "iC");
    if (transA)
      iA = Image<float>(M, K, "iA");
    else
      iA = Image<float>(K, M, "iA");
    if (transB)
      iB = Image<float>(K, N, "iB");
    else
      iB = Image<float>(N, K, "iB");

    init(A, B, C, iA, iB, iC);
    cblas_sgemm(CblasRowMajor,
                transA ? CblasTrans : CblasNoTrans,
                transB ? CblasTrans : CblasNoTrans,
                M, N, K, 1.f,
                A, transA ? M: K,
                B, transB ? K: N,
                1.f, C, N);
    Var x("x"), y("y");
    Func fA(iA), fB(iB), fC("fC");
    Dot_2dx2d(transA, transB, fA, fB, x, y, K, fC);
    fC.realize(iC);

    for (int i = 0; i < M*N; i++)
      TS_ASSERT(fabs(C[i] - iC.data()[i]) < EPSILON);

    delete[] A;
    delete[] B;
    delete[] C;
  }

  void TestDot2dx2d_ff() {
    _test_dot_2dx2d(false, false);
    TS_TRACE("TestDot2dx2d_ff pass!");
  }

  void TestDot2dx2d_ft() {
   _test_dot_2dx2d(false, true);
    TS_TRACE("TestDot2dx2d_ft pass!\n");
  }

  void TestDot2dx2d_tf() {
    _test_dot_2dx2d(true, false);
    TS_TRACE("TestDot2dx2d_tf pass!\n");
  }

  void TestDot2dx2d_tt() {
    _test_dot_2dx2d(true, true);
    TS_TRACE("TestDot2dx2d_tt pass!\n");
  }

  void TestDot3dx2d_ff() {
    Var x("x"), y("y"), z("z");
    Func fa;
    fa(x, y, z) = cast<float>(x + y + z);
    Image<float> A = fa.realize(2, 2, 2);

    Func fb;
    fb(x, y) = cast<float>(10 + x + y);
    Image<float> B = fb.realize(4, 2);

    Image<float> C0(4, 4);
    float16 f16 = {11., 12., 13., 14., 32., 35., 38., 41.,
                   32., 35., 38., 41., 53., 58., 63., 68.};
    *((float16*) C0.data()) = f16;

    Func fA = Func(A);
    Func fB = Func(B);
    Func fC("fC");
    Dot_3dx2d(false, false, fA, fB, x, y, z, 2, fC);
    Image<float> C = fC.realize(4, 2, 2);

    for (int ii = 0; ii < 16; ii++) {
      TS_ASSERT_EQUALS(C.data()[ii], C0.data()[ii]);
    }
  }

  void TestDot3dx2d_ft() {
    Var x("x"), y("y"), z("z");
    Func fa;
    fa(x, y, z) = cast<float>(x + y + z);
    Image<float> A = fa.realize(2, 2, 2);

    Func fb;
    fb(x, y) = cast<float>(10 + x + y);
    Image<float> B = fb.realize(2, 4);

    Image<float> C0(4, 4);
    float16 f16 = {11., 12., 13., 14., 32., 35., 38., 41.,
                   32., 35., 38., 41., 53., 58., 63., 68.};
    *((float16*) C0.data()) = f16;

    Func fA = Func(A);
    Func fB = Func(B);
    Func fC("fC");
    Dot_3dx2d(false, true, fA, fB, x, y, z, 2, fC);
    Image<float> C = fC.realize(4, 2, 2);

    for (int ii = 0; ii < 16; ii++) {
      TS_ASSERT_EQUALS(C.data()[ii], C0.data()[ii]);
    }
  }
};

#endif // HALSTM_TEST_MATRIX_H
