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
#include "CycleTimer.h"

typedef struct {
  float v[16];
} float16;

using namespace halstm;

class TestLstmLayer : public CxxTest::TestSuite {
public:
  void TestDot2dx2d_ff() {
    Var x, y, z;
    Func fa;
    fa(x, y) = cast<float>(x + y);
    Image<float> A = fa.realize(2, 4);

    Func fb;
    fb(x, y) = cast<float>(10 + x + y);
    Image<float> B = fb.realize(4, 2);

    Image<float> C0(4, 4);
    float16 f16 = {11., 12., 13., 14., 32., 35., 38., 41.,
                   53., 58., 63., 68., 74., 81., 88., 95.};
    *((float16*) C0.data()) = f16;

    Func fA = Func(A);
    Func fB = Func(B);
    Func fC("fC");
    Dot_2dx2d(false, false, fA, fB, x, y, 2, fC);
    Image<float> C = fC.realize(4, 4);

    for (int i = 0; i < 16; i++) {
      TS_ASSERT_EQUALS(C.data()[i], C0.data()[i]);
    }
  }

  void TestDot2dx2d_ft() {
    Var x, y, z;
    Func fa;
    fa(x, y) = cast<float>(x + y);
    Image<float> A = fa.realize(2, 4);

    Func fb;
    fb(x, y) = cast<float>(10 + x + y);
    Image<float> B = fb.realize(2, 4);

    Image<float> C0(4, 4);
    float16 f16 = {11., 12., 13., 14., 32., 35., 38., 41.,
                   53., 58., 63., 68., 74., 81., 88., 95.};
    *((float16*) C0.data()) = f16;

    Func fA = Func(A);
    Func fB = Func(B);
    Func fC("fC");
    Dot_2dx2d(false, true, fA, fB, x, y, 2, fC);
    Image<float> C = fC.realize(4, 4);

    for (int i = 0; i < 16; i++) {
      TS_ASSERT_EQUALS(C.data()[i], C0.data()[i]);
    }
  }

  void TestDot2dx2d_tf() {
    Var x, y, z;
    Func fa;
    fa(x, y) = cast<float>(x + y);
    Image<float> A = fa.realize(4, 2);

    Func fb;
    fb(x, y) = cast<float>(10 + x + y);
    Image<float> B = fb.realize(4, 2);

    Image<float> C0(4, 4);
    float16 f16 = {11., 12., 13., 14., 32., 35., 38., 41.,
                   53., 58., 63., 68., 74., 81., 88., 95.};
    *((float16*) C0.data()) = f16;

    Func fA = Func(A);
    Func fB = Func(B);
    Func fC("fC");
    Dot_2dx2d(true, false, fA, fB, x, y, 2, fC);
    Image<float> C = fC.realize(4, 4);

    for (int i = 0; i < 16; i++) {
      TS_ASSERT_EQUALS(C.data()[i], C0.data()[i]);
    }
  }

  void TestDot2dx2d_tt() {
    Var x, y, z;
    Func fa;
    fa(x, y) = cast<float>(x + y);
    Image<float> A = fa.realize(4, 2);

    Func fb;
    fb(x, y) = cast<float>(10 + x + y);
    Image<float> B = fb.realize(2, 4);

    Image<float> C0(4, 4);
    float16 f16 = {11., 12., 13., 14., 32., 35., 38., 41.,
                   53., 58., 63., 68., 74., 81., 88., 95.};
    *((float16*) C0.data()) = f16;

    Func fA = Func(A);
    Func fB = Func(B);
    Func fC("fC");
    Dot_2dx2d(true, true, fA, fB, x, y, 2, fC);

    Image<float> C = fC.realize(4, 4);
    for (int i = 0; i < 16; i++) {
      TS_ASSERT_EQUALS(C.data()[i], C0.data()[i]);
    }
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

  void TestDotPerf2dx2d(){
    int m=1000, n=1000, k=1000;
    Var x("x"), y("y"), z("z");
    Func fa, fb;
    fa(x, y) = cast<float> (x + y);
    fb(x, y) = cast<float>(10 + x + y);
    Image<float> A = fa.realize(k, m);
    Image<float> B = fa.realize(n, k);

    Func fA = Func(A);
    Func fB = Func(B);
    Func fC("fC");

    printf("\nTesting 2dx2d Matrix Dot Performance\n");
    double startTime = CycleTimer::currentSeconds();
    Dot_2dx2d(false, false, fA, fB, x, y, k, fC);
    Image<float> C = fC.realize(n, m);
    double endTime = CycleTimer::currentSeconds();
    double time = endTime - startTime;
    printf("(%d x %d x %d) Matrix Dot Takes: %f s\n", m, k, n, time);
  }
};

#endif // HALSTM_TEST_MATRIX_H
