/**
 * @file test_matrix.h
 * @brief Prototypes for test_matrix.
 *
 * @author misaka-10032 (longqic@andrew.cmu.edu)
 */

#ifndef HALSTM_TEST_MATRIX_H
#define HALSTM_TEST_MATRIX_H

#include <cxxtest/TestSuite.h>
#include "maths.h"

typedef struct {
  float v[16];
} float16;

using namespace halstm;

class TestLstmLayer : public CxxTest::TestSuite {
public:
  void TestDot2d_ffff() {
    Var i, j;
    Func fa;
    fa(i, j) = cast<float>(i + j);
    Image<float> A = fa.realize(2, 4);

    Func fb;
    fb(i, j) = cast<float>(10 + i + j);
    Image<float> B = fb.realize(4, 2);

    Image<float> C0(4, 4);
    float16 f16 = {11., 12., 13., 14., 32., 35., 38., 41.,
                   53., 58., 63., 68., 74., 81., 88., 95.};
    *((float16*) C0.data()) = f16;

    Func fA = Func(A);
    Func fB = Func(B);
    Func fC = matrix_dot(false, false, false, false,
                         fA, fB, 4, 4, 2);
    Image<float> C = fC.realize(4, 4);

    for (int i = 0; i < 16; i++) {
      TS_ASSERT_EQUALS(C.data()[i], C0.data()[i]);
    }
  }

  void TestDot2d_ffft() {
    Var i, j;
    Func fa;
    fa(i, j) = cast<float>(i + j);
    Image<float> A = fa.realize(2, 4);

    Func fb;
    fb(i, j) = cast<float>(10 + i + j);
    Image<float> B = fb.realize(2, 4);

    Image<float> C0(4, 4);
    float16 f16 = {11., 12., 13., 14., 32., 35., 38., 41.,
                   53., 58., 63., 68., 74., 81., 88., 95.};
    *((float16*) C0.data()) = f16;

    Func fA = Func(A);
    Func fB = Func(B);
    Func fC = matrix_dot(false, false, false, true,
                         fA, fB, 4, 4, 2);
    Image<float> C = fC.realize(4, 4);

    for (int i = 0; i < 16; i++) {
      TS_ASSERT_EQUALS(C.data()[i], C0.data()[i]);
    }
  }

  void TestDot2d_fftf() {
    Var i, j;
    Func fa;
    fa(i, j) = cast<float>(i + j);
    Image<float> A = fa.realize(4, 2);

    Func fb;
    fb(i, j) = cast<float>(10 + i + j);
    Image<float> B = fb.realize(4, 2);

    Image<float> C0(4, 4);
    float16 f16 = {11., 12., 13., 14., 32., 35., 38., 41.,
                   53., 58., 63., 68., 74., 81., 88., 95.};
    *((float16*) C0.data()) = f16;

    Func fA = Func(A);
    Func fB = Func(B);
    Func fC = matrix_dot(false, false, true, false,
                         fA, fB, 4, 4, 2);
    Image<float> C = fC.realize(4, 4);

    for (int i = 0; i < 16; i++) {
      TS_ASSERT_EQUALS(C.data()[i], C0.data()[i]);
    }
  }

  void TestDot2d_fftt() {
    Var i, j;
    Func fa;
    fa(i, j) = cast<float>(i + j);
    Image<float> A = fa.realize(4, 2);

    Func fb;
    fb(i, j) = cast<float>(10 + i + j);
    Image<float> B = fb.realize(2, 4);

    Image<float> C0(4, 4);
    float16 f16 = {11., 12., 13., 14., 32., 35., 38., 41.,
                   53., 58., 63., 68., 74., 81., 88., 95.};
    *((float16*) C0.data()) = f16;

    Func fA = Func(A);
    Func fB = Func(B);
    Func fC = matrix_dot(false, false, true, true,
                         fA, fB, 4, 4, 2);
    Image<float> C = fC.realize(4, 4);

    for (int i = 0; i < 16; i++) {
      TS_ASSERT_EQUALS(C.data()[i], C0.data()[i]);
    }
  }

  void TestDot2d_tfff() {
    Var i, j, k;
    Func fa;
    fa(i, j, k) = cast<float>(i + j + k);
    Image<float> A = fa.realize(2, 2, 2);

    Func fb;
    fb(i, j) = cast<float>(10 + i + j);
    Image<float> B = fb.realize(4, 2);

    Image<float> C0(4, 4);
    float16 f16 = {11., 12., 13., 14., 32., 35., 38., 41.,
                   32., 35., 38., 41., 53., 58., 63., 68.};
    *((float16*) C0.data()) = f16;

    Func fA = Func(A);
    Func fB = Func(B);
    Func fC = matrix_dot(true, false, false, false,
                         fA, fB, 4, 4, 2);
    Image<float> C = fC.realize(4, 2, 2);

    for (int i = 0; i < 16; i++) {
      TS_ASSERT_EQUALS(C.data()[i], C0.data()[i]);
    }
  }

};

#endif // HALSTM_TEST_MATRIX_H
