//
// Created by XiaotongSun on 16/4/18.
//

#include "Halide.h"
#include "activation.h"
#include "maths.h"
#include "layers.h"

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


/*
 * Should be used later as I/O component
 * Populate buffer_t struct to generate input Image<T>, which is a matrix
 */
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

  //-------------test matrix multiplication

  const int m = 2, l = 2, n = 3;
  float A[m*l] = {0}; //A[0] = -2.0; A[1] = 1.0; A[2] = 2.0; A[3] = 3.0;
  float B[l*n] = {0}; //B[0] = 1.0; B[1] = -2.0; B[2] = 1.0; B[3] = 2.0; B[4] = -2.0; B[5] = -1.0;


  buffer_t mA = {0}, mB = {0}, mC = {0};
  populate_data_buffer(&mA, &A[0], 1,2,2,2,4);
  populate_data_buffer(&mB, &B[0], 1,3,3,2,4);
  Image<float> inputA(&mA, "matrixA");
  Image<float> inputB(&mB, "matrixB");
  printf("A(%d, %d)\n", inputA.width(), inputA.height());
  printf("B(%d, %d)\n", inputB.width(), inputB.height());
  inputA(0, 0) = 3;
  inputA(1, 0) = 1;
  inputA(0, 1) = 4;
  inputA(1, 1) = 2;
  inputB(0, 0) = -1;
  inputB(1, 0) = 1;
  inputB(2, 0) = 0;
  inputB(0, 1) = 1;
  inputB(1, 1) = 2;
  inputB(2, 1) = -1;

  //printf("B(%d, %d) = %f\n", 2, 1, inputB(2,1));

  Var i("i"), j("j"), k("k");
  Func funcA, funcB;
  funcA(i, j) = inputA(i, j);
  funcB(i, j) = inputB(i, j);

//  Func test1("A"), test2("B");
//  test1(i, j) = inputA(i, j);
//  test2(i, j) = inputA(i, j);
//  Func test("C");
//  test(k, j, i) = test1(k, i) * test2(j, k);
//  test.print_loop_nest();
//  test.trace_stores();
//  test.realize(2,2,2);

  Func matrix_dot = halstm::define_matrix_dot(funcA, funcB, 2);
  Image<float> c(3,2);
  matrix_dot.trace_stores();
  matrix_dot.realize(c);

  // -----------------test matrix add

//  int m = 2;
//  int n = 1;
//
//  float A[m*n]; A[0] = -2.0; A[1] = 1.0;
//  float B[m*n]; B[0] = 1.0; B[1] = -3.0;
//  buffer_t mA = {0}, mB = {0};
//  populate_data_buffer(&mA, &A[0],1,1,1,2, 4);
//  populate_data_buffer(&mB, &B[0],1,1,1,2, 4);
//  Image<float> inputA(&mA, "matrixA");
//  Image<float> inputB(&mB, "matrixB");
//  printf("A(%d, %d)\n", inputA.height(), inputA.width());
//  printf("B(%d, %d)\n", inputB.height(), inputB.width());
//
//  Var i, j, l;
//  Func funcA, funcB;
//  funcA(i, j) = inputA(i, j);
//  funcB(i, j) = inputB(i, j);
//
//  Func *matrix_add = halstm::define_matrix_add(funcA, funcB, m, n);
//  Image<float> c(1, 2);
//  matrix_add->trace_stores();
//  matrix_add->realize(c);
//  delete matrix_add;

//  Var i, j;
//  Func *test = new Func("heh");
//  (*test)(i, j) = (i + j);
//  test->trace_stores();
//  test->realize(2,2);

  return 0;
}