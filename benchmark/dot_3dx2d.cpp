/**
 * @file dot_2dx2d.cpp
 * @brief benchmark for dot_2dx2d
 *
 * @author misaka-10032 (longqic@andrew.cmu.edu)
 */

#include <stdlib.h>
#include <cblas.h>
#include "maths.h"
#include "benchmark.h"

using namespace halstm;

const int R = 64;
const int M = 1024;
const int N = 1024;
const int K = 2048;
const int RUNS = 10;

void init(int nA, int nB, float* A, float* B,
          Image<float>& iA, Image<float>& iB) {
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
}

void compare(bool transA, bool transB) {
  float *A = new float[R*M*K];
  float *B = new float[K*N];
  Image<float> iA, iB, iC(N, M, R, "iC");
  if (transA)
    iA = Image<float>(R, M, K);
  else
    iA = Image<float>(K, M, R);
  if (transB)
    iB = Image<float>(K, N);
  else
    iB = Image<float>(N, K);
  init(R*M*K, K*N, A, B, iA, iB);

  double time = benchmark(RUNS, 1, [&]() {
    float *C = new float[R*M*N];
    memset(C, 0, R*M*N*sizeof(float));
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                R*M, N, K, 1.f, A, K, B, N, 1.f, C, N);
    delete[] C;
  });
  printf("(%d, %d, %d)%s dot (%d, %d)%s by vecLib takes: %f ms\n",
         R, M, K, transA ? ".T" : "", K, N, transB ? ".T" : "", time);
  delete[] A;
  delete[] B;

  Var x("x"), y("y"), z("z");
  Func fA(iA), fB(iB);
  time = benchmark(RUNS, 1, [&]() {
    Func fC("fC");
    Dot_3dx2d(transA, transB, fA, fB, x, y, z, K, M, fC);
    fC.realize(iC);
  });
  printf("(%d, %d, %d)%s dot (%d, %d)%s by Halide takes: %f ms\n",
         R, M, K, transA ? ".T" : "", K, N, transB ? ".T" : "", time);
}

int main(int argc, char** argv) {
  printf("A dot B\n");
  compare(false, false);
  printf("\n");

  printf("A dot B.T\n");
  compare(false, true);
  printf("\n");

  // TODO: write 3dx2d_tf/tt with sgemm
//  printf("A.T dot B\n");
//  compare(true, false);
//  printf("\n");
//
//  printf("A.T dot B.T\n");
//  compare(true, true);
//  printf("\n");

  return 0;
}
