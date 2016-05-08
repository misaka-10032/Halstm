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

const int R = 64;
const int M = 256;
const int N = 256;
const int K = 512;

using namespace halstm;

class TestLstmLayer : public CxxTest::TestSuite {
public:
  void init(int nA, int nB, int nC, float* A, float* B, float* C,
            Image<float>& iA, Image<float>& iB, Image<float>& iC) {
    for (int i = 0; i < nA; i++) {
      float r = static_cast<float> (rand()) / static_cast <float> (RAND_MAX);
      A[i] = r;
      iA.data()[i] = r;
    }
    for (int i = 0; i < nB; i++) {
      float r = static_cast<float> (rand()) / static_cast <float> (RAND_MAX);
      B[i] = r;
      iB.data()[i] = r;
    }
    for (int i = 0; i < nC; i++) {
      C[i] = 0.f;
      iC.data()[i] = 0.f;
    }
  }

  void _test_dot_2dx2d(bool transA, bool transB) {
    float* A = new float[M*K];
    float* B = new float[K*N];
    float* C = new float[M*N];
    Image<float> iA, iB, iC(N, M, "iC");
    if (transA)
      iA = Image<float>(M, K, "iA");
    else
      iA = Image<float>(K, M, "iA");
    if (transB)
      iB = Image<float>(K, N, "iB");
    else
      iB = Image<float>(N, K, "iB");

    init(M*K, K*N, M*N, A, B, C, iA, iB, iC);
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

  void _test_dot_3dx2d(bool transA, bool transB) {
    float* A = new float[R*M*K];
    float* B = new float[K*N];
    float* C = new float[R*M*N];
    Image<float> iA, iB, iC(N, M, R, "iC");
    if (transA)
      iA = Image<float>(R, M, K, "iA");
    else
      iA = Image<float>(K, M, R, "iA");
    if (transB)
      iB = Image<float>(K, N, "iB");
    else
      iB = Image<float>(N, K, "iB");
    init(R*M*K, K*N, R*M*N, A, B, C, iA, iB, iC);

    cblas_sgemm(CblasRowMajor,
                transA ? CblasTrans : CblasNoTrans,
                transB ? CblasTrans : CblasNoTrans,
                R*M, N, K, 1.f,
                A, transA ? R*M: K,
                B, transB ? K: N,
                1.f, C, N);
    Var x("x"), y("y"), z("z");
    Func fA(iA), fB(iB), fC("fC");
    Dot_3dx2d(transA, transB, fA, fB, x, y, z, K, M, fC);
    fC.realize(iC);

    for (int i = 0; i < R*M*N; i++)
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
    _test_dot_3dx2d(false, false);
    TS_TRACE("TestDot3dx2d_ff pass!\n");
  }

  void TestDot3dx2d_ft() {
    _test_dot_3dx2d(false, true);
    TS_TRACE("TestDot3dx2d_ft pass!\n");
  }

  // TODO: TestDot3dx2d_tf/tt with sgemm
//  void TestDot3dx2d_tf() {
//    _test_dot_3dx2d(true, false);
//    TS_TRACE("TestDot3dx2d_tf pass!\n");
//  }
//
//  void TestDot3dx2d_tt() {
//    _test_dot_3dx2d(true, true);
//    TS_TRACE("TestDot3dx2d_tt pass!\n");
//  }
};

#endif // HALSTM_TEST_MATRIX_H
