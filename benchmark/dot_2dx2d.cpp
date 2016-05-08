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

const int M = 2048;
const int N = 2048;
const int K = 4096;
const int RUNS = 10;

void compare(bool transA, bool transB) {
  float *A = new float[M * K];
  float *B = new float[K * N];
  for (int i = 0; i < M*K; i++)
    A[i] = static_cast<float> (rand()) / static_cast <float> (RAND_MAX);
  for (int i = 0; i < K*N; i++)
    B[i] = static_cast<float> (rand()) / static_cast <float> (RAND_MAX);
  const int ldc = N;
  double time = benchmark(RUNS, 1, [&]() {
    float *C = new float[M * N];
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, 1.f, A, K, B, N, 1.f, C, ldc);
    delete[] C;
  });
  printf("(%d, %d)%s dot (%d, %d)%s by vecLib takes: %f ms\n",
         M, K, transA ? ".T" : "", K, N, transB ? ".T" : "", time);
  delete[] A;
  delete[] B;

  Var x("x"), y("y");
  Func fA("fA"), fB("fB");
  fA(x, y) = static_cast<float> (rand()) / static_cast <float> (RAND_MAX);
  fB(x, y) = static_cast<float> (rand()) / static_cast <float> (RAND_MAX);
  transA ? fA.realize(M, K) : fA.realize(K, M);
  transB ? fB.realize(K, N) : fB.realize(N, K);
  time = benchmark(RUNS, 1, [&]() {
    Func fC("fC");
    Dot_2dx2d(transA, transB, fA, fB, x, y, K, fC);
    fC.realize(N, M);
  });
  printf("(%d, %d)%s dot (%d, %d)%s by Halide takes: %f ms\n",
         M, K, transA ? ".T" : "", K, N, transB ? ".T" : "", time);
}

// That by official site is not faster
void compare1(bool transA, bool transB) {
  float *A = new float[M * K];
  float *B = new float[K * N];
  for (int i = 0; i < M*K; i++)
    A[i] = static_cast<float> (rand()) / static_cast <float> (RAND_MAX);
  for (int i = 0; i < K*N; i++)
    B[i] = static_cast<float> (rand()) / static_cast <float> (RAND_MAX);
  const int ldc = N;
  double time = benchmark(RUNS, 1, [&]() {
    float *C = new float[M * N];
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, 1.f, A, K, B, N, 1.f, C, ldc);
    delete[] C;
  });
  printf("(%d, %d)%s dot (%d, %d)%s by vecLib takes: %f ms\n",
         M, K, transA ? ".T" : "", K, N, transB ? ".T" : "", time);
  delete[] A;
  delete[] B;

  Var x("x"), y("y");
  Func fA("fA"), fB("fB");
  Image<float> iA, iB;
  fA(x, y) = static_cast<float> (rand()) / static_cast <float> (RAND_MAX);
  fB(x, y) = static_cast<float> (rand()) / static_cast <float> (RAND_MAX);
  if (transA) {
    iA = Image<float>(K, M);
  } else {
    iA = Image<float>(M, K);
  }
  fA.realize(iA);
  if (transB) {
    iB = Image<float>(N, K);
  } else {
    iB = Image<float>(K, N);
  }
  fB.realize(iB);

  time = benchmark(RUNS, 1, [&]() {
    Func fC("fC");
    Var i, j, ii, ji, jii, io, jo, t;
    const Expr num_rows = (iA.width()/32)*32;
    const Expr num_cols = (iB.height()/32)*32;
    const Expr sum_size = (iA.height()/32)*32;
    const Expr iA_width = iA.width();
    const Expr iA_height = iA.height();

    const int vec = 4;
    const int s = vec * 2;

    // Instead of transposing B, swap A and B, transpose A, and
    // then transpose AB.
    bool transpose_AB = false;
    bool transpose_A_ = transA;
    bool transpose_B_ = transB;
    if (transB) {
      std::swap(iA, iB);
      transpose_A_ = !transpose_A_;
      transpose_B_ = false;
      transpose_AB = true;
    }

    Var ti[3], tj[3];

    // Swizzle A for better memory order in the inner loop.
    Func Ar("Ar"), Br("Br"), As("As"), Atmp("Atmp");
    Atmp(i, j) = iA(i, j);
    if (transpose_A_) {
      As(i, j, io) = Atmp(j, io*s + i);
    } else {
      As(i, j, io) = Atmp(io*s + i, j);
    }

    Ar(i, j) = As(i % s, j, i / s);
    Br(i, j) = iB(i, j);

    Var k("k");
    Func prod;
    // Express all the products we need to do a matrix multiply as a 3D Func.
    prod(k, i, j) = Ar(i, k) * Br(k, j);

    // Reduce the products along k.
    Func AB("AB");
    RDom rv(0, sum_size);
    AB(i, j) += prod(rv, i, j);

    Func ABt("ABt");
    if (transpose_AB) {
      // Transpose A*B if necessary.
      ABt(i, j) = AB(j, i);
    } else {
      ABt(i, j) = AB(i, j);
    }

    // Do the part that makes it a 'general' matrix multiply.
    fC(i, j) = ABt(i, j);

    if (transpose_AB) {
      fC
          .tile(i, j, ii, ji, 4, s).vectorize(ii).unroll(ji)
          .tile(i, j, ti[0], tj[0], i, j, s/4, 1);
    } else {
      fC
          .tile(i, j, ii, ji, s, 4).vectorize(ii).unroll(ji)
          .tile(i, j, ti[0], tj[0], i, j, 1, s/4);
    }
    fC.tile(ti[0], tj[0], ti[0], tj[0], ti[1], tj[1], 2, 2);

    // If we have enough work per task, parallelize over these tiles.
    fC.specialize(num_rows >= 256 && num_cols >= 256)
        .fuse(tj[0], ti[0], t).parallel(t);

    // Otherwise tile one more time before parallelizing, or don't
    // parallelize at all.
    fC.specialize(num_rows >= 128 && num_cols >= 128)
        .tile(ti[0], tj[0], ti[0], tj[0], ti[2], tj[2], 2, 2)
        .fuse(tj[0], ti[0], t).parallel(t);

    fC.bound(i, 0, num_rows).bound(j, 0, num_cols);

    As.compute_root()
        .split(j, jo, ji, s).reorder(i, ji, io, jo)
        .unroll(i).vectorize(ji)
        .specialize(iA_width >= 256 && iA_height >= 256).parallel(jo, 4);

    Atmp.compute_at(As, io)
        .vectorize(i).unroll(j);

    AB.compute_at(fC, i)
        .unroll(j).vectorize(i)
        .update()
        .reorder(i, j, rv).unroll(j).unroll(rv, 2).vectorize(i);

    if (transpose_AB) {
      ABt.compute_at(fC, i).unroll(i).vectorize(j);
    }

//    A_.set_min(0, 0).set_min(1, 0);
//    B_.set_bounds(0, 0, sum_size).set_min(1, 0);
//    C_.set_bounds(0, 0, num_rows).set_bounds(1, 0, num_cols);
//    fC.output_buffer().set_bounds(0, 0, num_rows).set_bounds(1, 0, num_cols);

//    fC.print_loop_nest();

//    fC.realize(M, N);
    fC.realize(N, M);
  });
  printf("(%d, %d)%s dot (%d, %d)%s by Halide takes: %f ms\n",
         M, K, transA ? ".T" : "", K, N, transB ? ".T" : "", time);
}

int main(int argc, char** argv) {
  printf("A dot B\n");
  compare(false, false);
  printf("\n");

  printf("A dot B.T\n");
  compare(false, true);
  printf("\n");

  printf("A.T dot B\n");
  compare(true, false);
  printf("\n");

  printf("A.T dot B.T\n");
  compare(true, true);
  printf("\n");

  return 0;
}
