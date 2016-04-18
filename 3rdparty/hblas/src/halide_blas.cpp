#include <string.h>
#include <iostream>
#include "halide_blas.h"

#define assert_no_error(func)                                       \
  if (func != 0) {                                                  \
    std::cerr << "ERROR! Halide kernel returned non-zero value.\n"; \
  }                                                                 \

namespace {

void init_scalar_buffer(const float *x, buffer_t *buff) {
    memset((void*)buff, 0, sizeof(buffer_t));
    buff->host = (uint8_t*)const_cast<float*>(x);
    buff->extent[0] = 1;
    buff->stride[0] = 1;
    buff->elem_size = sizeof(float);
}

void init_scalar_buffer(const double *x, buffer_t *buff) {
    memset((void*)buff, 0, sizeof(buffer_t));
    buff->host = (uint8_t*)const_cast<double*>(x);
    buff->extent[0] = 1;
    buff->stride[0] = 1;
    buff->elem_size = sizeof(double);
}

void init_vector_buffer(const int N, const float *x, const int incx, buffer_t *buff) {
    memset((void*)buff, 0, sizeof(buffer_t));
    buff->host = (uint8_t*)const_cast<float*>(x);
    buff->extent[0] = N;
    buff->stride[0] = incx;
    buff->elem_size = sizeof(float);
}

void init_vector_buffer(const int N, const double *x, const int incx, buffer_t *buff) {
    memset((void*)buff, 0, sizeof(buffer_t));
    buff->host = (uint8_t*)const_cast<double*>(x);
    buff->extent[0] = N;
    buff->stride[0] = incx;
    buff->elem_size = sizeof(double);
}

void init_matrix_buffer(const int M, const int N, const float *A, const int lda, buffer_t *buff) {
    memset((void*)buff, 0, sizeof(buffer_t));
    buff->host = (uint8_t*)const_cast<float*>(A);
    buff->extent[0] = M;
    buff->extent[1] = N;
    buff->stride[0] = 1;
    buff->stride[1] = lda;
    buff->elem_size = sizeof(float);
}

void init_matrix_buffer(const int M, const int N, const double *A, const int lda, buffer_t *buff) {
    memset((void*)buff, 0, sizeof(buffer_t));
    buff->host = (uint8_t*)const_cast<double*>(A);
    buff->extent[0] = M;
    buff->extent[1] = N;
    buff->stride[0] = 1;
    buff->stride[1] = lda;
    buff->elem_size = sizeof(double);
}

}

#ifdef __cplusplus
extern "C" {
#endif

//////////
// copy //
//////////

void hblas_scopy(const int N, const float *x, const int incx,
                 float *y, const int incy) {
    buffer_t buff_x, buff_y;
    init_vector_buffer(N, x, incx, &buff_x);
    init_vector_buffer(N, y, incy, &buff_y);
    assert_no_error(halide_scopy(&buff_x, &buff_y));
}

void hblas_dcopy(const int N, const double *x, const int incx,
                 double *y, const int incy) {
    buffer_t buff_x, buff_y;
    init_vector_buffer(N, x, incx, &buff_x);
    init_vector_buffer(N, y, incy, &buff_y);
    assert_no_error(halide_dcopy(&buff_x, &buff_y));
}

//////////
// scal //
//////////

void hblas_sscal(const int N, const float a, float *x, const int incx) {
    buffer_t buff_x;
    init_vector_buffer(N, x, incx, &buff_x);
    assert_no_error(halide_sscal(a, &buff_x));
}

void hblas_dscal(const int N, const double a, double *x, const int incx) {
    buffer_t buff_x;
    init_vector_buffer(N, x, incx, &buff_x);
    assert_no_error(halide_dscal(a, &buff_x));
}

//////////
// axpy //
//////////

void hblas_saxpy(const int N, const float a, const float *x, const int incx,
                 float *y, const int incy) {
    buffer_t buff_x, buff_y;
    init_vector_buffer(N, x, incx, &buff_x);
    init_vector_buffer(N, y, incy, &buff_y);
    assert_no_error(halide_saxpy(a, &buff_x, &buff_y));
}

void hblas_daxpy(const int N, const double a, const double *x, const int incx,
                 double *y, const int incy) {
    buffer_t buff_x, buff_y;
    init_vector_buffer(N, x, incx, &buff_x);
    init_vector_buffer(N, y, incy, &buff_y);
    assert_no_error(halide_daxpy(a, &buff_x, &buff_y));
}

//////////
// dot  //
//////////

float hblas_sdot(const int N, const float *x, const int incx,
                 const float *y, const int incy) {
    float result;
    buffer_t buff_x, buff_y, buff_dot;
    init_vector_buffer(N, x, incx, &buff_x);
    init_vector_buffer(N, y, incy, &buff_y);
    init_scalar_buffer(&result, &buff_dot);
    assert_no_error(halide_sdot(&buff_x, &buff_y, &buff_dot));
    return result;
}

double hblas_ddot(const int N, const double *x, const int incx,
                  const double *y, const int incy) {
    double result;
    buffer_t buff_x, buff_y, buff_dot;
    init_vector_buffer(N, x, incx, &buff_x);
    init_vector_buffer(N, y, incy, &buff_y);
    init_scalar_buffer(&result, &buff_dot);
    assert_no_error(halide_ddot(&buff_x, &buff_y, &buff_dot));
    return result;
}

//////////
// nrm2 //
//////////

float hblas_snrm2(const int N, const float *x, const int incx) {
    float result;
    buffer_t buff_x, buff_nrm;
    init_vector_buffer(N, x, incx, &buff_x);
    init_scalar_buffer(&result, &buff_nrm);
    assert_no_error(halide_sdot(&buff_x, &buff_x, &buff_nrm));
    return std::sqrt(result);
}

double hblas_dnrm2(const int N, const double *x, const int incx) {
    double result;
    buffer_t buff_x, buff_nrm;
    init_vector_buffer(N, x, incx, &buff_x);
    init_scalar_buffer(&result, &buff_nrm);
    assert_no_error(halide_ddot(&buff_x, &buff_x, &buff_nrm));
    return std::sqrt(result);
}

//////////
// asum //
//////////

float hblas_sasum(const int N, const float *x, const int incx) {
    float result;
    buffer_t buff_x, buff_sum;
    init_vector_buffer(N, x, incx, &buff_x);
    init_scalar_buffer(&result, &buff_sum);
    assert_no_error(halide_sasum(&buff_x, &buff_sum));
    return result;
}

double hblas_dasum(const int N, const double *x, const int incx) {
    double result;
    buffer_t buff_x, buff_sum;
    init_vector_buffer(N, x, incx, &buff_x);
    init_scalar_buffer(&result, &buff_sum);
    assert_no_error(halide_dasum(&buff_x, &buff_sum));
    return result;
}

//////////
// gemv //
//////////

void hblas_sgemv(const enum HBLAS_ORDER Order, const enum HBLAS_TRANSPOSE trans,
                 const int M, const int N, const float a, const float *A, const int lda,
                 const float *x, const int incx, const float b, float *y, const int incy) {
    bool t = false;
    switch (trans) {
    case HblasNoTrans:
        t = false; break;
    case HblasConjTrans:
    case HblasTrans:
        t = true; break;
    };

    buffer_t buff_A, buff_x, buff_y;
    init_matrix_buffer(M, N, A, lda, &buff_A);
    if (t) {
        init_vector_buffer(M, x, incx, &buff_x);
        init_vector_buffer(N, y, incy, &buff_y);
    } else {
        init_vector_buffer(N, x, incx, &buff_x);
        init_vector_buffer(M, y, incy, &buff_y);
    }

    assert_no_error(halide_sgemv(t, a, &buff_A, &buff_x, b, &buff_y));
}

void hblas_dgemv(const enum HBLAS_ORDER Order, const enum HBLAS_TRANSPOSE trans,
                 const int M, const int N, const double a, const double *A, const int lda,
                 const double *x, const int incx, const double b, double *y, const int incy) {
    bool t = false;
    switch (trans) {
    case HblasNoTrans:
        t = false; break;
    case HblasConjTrans:
    case HblasTrans:
        t = true; break;
    };

    buffer_t buff_A, buff_x, buff_y;
    init_matrix_buffer(M, N, A, lda, &buff_A);
    if (t) {
        init_vector_buffer(M, x, incx, &buff_x);
        init_vector_buffer(N, y, incy, &buff_y);
    } else {
        init_vector_buffer(N, x, incx, &buff_x);
        init_vector_buffer(M, y, incy, &buff_y);
    }

    assert_no_error(halide_dgemv(t, a, &buff_A, &buff_x, b, &buff_y));
}

//////////
// ger  //
//////////

void hblas_sger(const enum HBLAS_ORDER order, const int M, const int N,
                const float alpha, const float *x, const int incx,
                const float *y, const int incy, float *A, const int lda)
{
    buffer_t buff_x, buff_y, buff_A;
    init_vector_buffer(M, x, incx, &buff_x);
    init_vector_buffer(N, y, incy, &buff_y);
    init_matrix_buffer(M, N, A, lda, &buff_A);

    assert_no_error(halide_sger(alpha, &buff_x, &buff_y, &buff_A));
}

void hblas_dger(const enum HBLAS_ORDER order, const int M, const int N,
                const double alpha, const double *x, const int incx,
                const double *y, const int incy, double *A, const int lda)
{
    buffer_t buff_x, buff_y, buff_A;
    init_vector_buffer(M, x, incx, &buff_x);
    init_vector_buffer(N, y, incy, &buff_y);
    init_matrix_buffer(M, N, A, lda, &buff_A);

    assert_no_error(halide_dger(alpha, &buff_x, &buff_y, &buff_A));
}

//////////
// gemm //
//////////

void hblas_sgemm(const enum HBLAS_ORDER Order, const enum HBLAS_TRANSPOSE TransA,
                 const enum HBLAS_TRANSPOSE TransB, const int M, const int N,
                 const int K, const float alpha, const float *A,
                 const int lda, const float *B, const int ldb,
                 const float beta, float *C, const int ldc) {
    bool tA = false, tB = false;
    switch (TransA) {
    case HblasNoTrans:
        tA = false; break;
    case HblasConjTrans:
    case HblasTrans:
        tA = true; break;
    };

    switch (TransB) {
    case HblasNoTrans:
        tB = false; break;
    case HblasConjTrans:
    case HblasTrans:
        tB = true; break;
    };

    buffer_t buff_A, buff_B, buff_C;
    if (!tA) {
        init_matrix_buffer(M, K, A, lda, &buff_A);
    } else {
        init_matrix_buffer(K, M, A, lda, &buff_A);
    }

    if (!tB) {
        init_matrix_buffer(K, N, B, ldb, &buff_B);
    } else {
        init_matrix_buffer(N, K, B, ldb, &buff_B);
    }

    init_matrix_buffer(M, N, C, ldc, &buff_C);

    assert_no_error(halide_sgemm(tA, tB, alpha, &buff_A, &buff_B, beta, &buff_C));
}

void hblas_dgemm(const enum HBLAS_ORDER Order, const enum HBLAS_TRANSPOSE TransA,
                 const enum HBLAS_TRANSPOSE TransB, const int M, const int N,
                 const int K, const double alpha, const double *A,
                 const int lda, const double *B, const int ldb,
                 const double beta, double *C, const int ldc) {
    bool tA = false, tB = false;
    switch (TransA) {
    case HblasNoTrans:
        tA = false; break;
    case HblasConjTrans:
    case HblasTrans:
        tA = true; break;
    };

    switch (TransB) {
    case HblasNoTrans:
        tB = false; break;
    case HblasConjTrans:
    case HblasTrans:
        tB = true; break;
    };

    buffer_t buff_A, buff_B, buff_C;
    if (!tA) {
        init_matrix_buffer(M, K, A, lda, &buff_A);
    } else {
        init_matrix_buffer(K, M, A, lda, &buff_A);
    }

    if (!tB) {
        init_matrix_buffer(K, N, B, ldb, &buff_B);
    } else {
        init_matrix_buffer(N, K, B, ldb, &buff_B);
    }

    init_matrix_buffer(M, N, C, ldc, &buff_C);

    assert_no_error(halide_dgemm(tA, tB, alpha, &buff_A, &buff_B, beta, &buff_C));
}


#ifdef __cplusplus
}
#endif
