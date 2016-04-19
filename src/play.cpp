//
// Created by XiaotongSun on 16/4/18.
//

#include "Halide.h"
#include "activation.h"
#include "maths.h"

using namespace Halide;

// buffer_t Reference
//typedef struct buffer_t {
//  uint64_t dev;
//  uint8_t* host;
//  int32_t extent[4];
//  int32_t stride[4];
//  int32_t min[4];
//  int32_t elem_size;
//  HALIDE_ATTRIBUTE_ALIGN(1) bool host_dirty;
//  HALIDE_ATTRIBUTE_ALIGN(1) bool dev_dirty;
//  HALIDE_ATTRIBUTE_ALIGN(1) uint8_t _padding[10 - sizeof(void *)];
//} buffer_t;

template <class T>
void populate_data_buffer(buffer_t * buffer , T* host_addr, int32_t stride_0, int32_t stride_1, int32_t extent_0, int32_t extent_1, int32_t elem_size){
  buffer->host = (uint8_t*)host_addr;
  buffer->stride[0] = stride_0;
  buffer->stride[1] = stride_1;
  buffer->extent[0] = extent_0;
  buffer->extent[1] = extent_1;
  buffer->elem_size = elem_size;
}

int main(int argc, char **argv){
  Var x, y;


//--------test activation function-------

//  int row = 2;
//  int col = 2;
//  float raw_data[row*col] ;
//  for(int i = 0; i < row; i++){
//    for(int j = 0; j < col; j++)
//      raw_data[i*row+j] = 5.0;
//  }
//
//  buffer_t data = {0};
//  populate_data_buffer<float>(&data, &raw_data[0], 1, 2, 2, 2, 4);
//  Image<float_t> input(&data, "data");
//  Func in;
//  in(x, y) = input(x, y);


//--------Func tanh_ac = halstm::define_tanh(in);-------
//  tanh_ac.trace_stores();
//  Image<float> r1 = tanh_ac.realize(2, 2);
//
//  Func sigmoid_ac = halstm::define_sigmoid(in);
//  sigmoid_ac.trace_stores();
//  Image<float> r2 = sigmoid_ac.realize(2,2);

  // test matrix multiplication

  int m = 2, k = 2, n = 2;
  float A[m*k]; A[0] = 2.0; A[1] = 3.0; A[2] = 1.0; A[3] = 5.0;
  float B[k*n]; B[0] = 4.0; B[1] = -2.0; B[2] = 3.0; B[3] = 2.0;
  float C[m*n]; C[0] = 0.0; C[1] = 0.0; C[2] = 0.0; C[3] = 0.0;
  float alpha = 1.0f;
  float beta = 1.0f;

  buffer_t mA = {0}, mB = {0}, mC = {0};
  populate_data_buffer(&mA, &A[0], 1,2,2,2,4);
  populate_data_buffer(&mB, &B[0], 1,2,2,2,4);
  populate_data_buffer(&mC, &C[0], 1,2,2,2,4);
  Image<float> inputA(&mA, "matrixA");
  Image<float> inputB(&mB, "matrixB");
  Image<float> inputC(&mC, "matrixC");

  Var i, j;
  Func funcA, funcB, funcC;
  funcA(i, j) = inputA(i, j);
  funcB(i, j) = inputB(i, j);
  funcC(i, j) = inputC(i, j);

  Func sgemm = halstm::define_hal_gemm(false, false, 2,2,2,alpha, funcA, funcB, beta, funcC);
  sgemm.trace_stores();
  sgemm.realize(2,2);

  return 0;
}