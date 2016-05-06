//
// Created by XiaotongSun on 16/5/5.
//


#include <cxxtest/TestSuite.h>
#include "maths.h"
#include "CycleTimer.h"

using namespace halstm;

class TestLstmLayer : public CxxTest::TestSuite{
public:
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