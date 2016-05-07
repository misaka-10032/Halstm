//
// Created by XiaotongSun on 16/5/5.
//


#include <cxxtest/TestSuite.h>
#include "maths.h"
#include "CycleTimer.h"

using namespace halstm;

class TestLstmLayer : public CxxTest::TestSuite{
public:
  void TestDotPerf3dx2d(){
    int t = 20, m=800, n=64, k=100;
    Var x("x"), y("y"), z("z");
    Func fa("fa"), fb("fb");
    fa(x, y, z) = cast<float> (x + y + z);
    fb(x, y) = cast<float>(10 + x + y);
    Image<float> A = fa.realize(k, m, t);
    Image<float> B = fb.realize(n, k);

    Func fA = Func(A);
    Func fB = Func(B);
    Func fC("fC");

    printf("\nTesting 3dx2d Matrix Dot Performance\n");
    double startTime = CycleTimer::currentSeconds();
    Dot_3dx2d(false, false, fA, fB, x, y, z, k, fC);
    Image<float> C = fC.realize(n, m, t);
    double endTime = CycleTimer::currentSeconds();
    double time = endTime - startTime;
    printf("(%d x %d x %d) Matrix Dot Takes: %f s\n", m, k, n, time);
  }
};