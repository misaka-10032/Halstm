//
// Created by XiaotongSun on 16/4/17.
//

#include "linear_algebra/sgemm.h"
#include <stdio.h>

int main(int argc, char ** argv){
//  typedef struct buffer_t {
//    uint64_t dev;
//    uint8_t* host;
//    int32_t extent[4];
//    int32_t stride[4];
//    int32_t min[4];
//    int32_t elem_size;
//    HALIDE_ATTRIBUTE_ALIGN(1) bool host_dirty;
//    HALIDE_ATTRIBUTE_ALIGN(1) bool dev_dirty;
//    HALIDE_ATTRIBUTE_ALIGN(1) uint8_t _padding[10 - sizeof(void *)];
//  } buffer_t;

  float matrix_A[2*2];
  matrix_A[0] = 3.0f;
  matrix_A[1] = 2.0f;
  matrix_A[2] = 1.0f;
  matrix_A[3] = -1.0f;

  float matrix_B[2*2];
  matrix_B[0] = 1.0f;
  matrix_B[1] = 2.0f;
  matrix_B[2] = -1.0f;
  matrix_B[3] = 0.0f;

  float matrix_C[2*2];
  matrix_C[0] = 0.0f;
  matrix_C[1] = 0.0f;
  matrix_C[2] = 0.0f;
  matrix_C[3] = 0.0f;

  float output[2*2];

  float alpha = 1.0f;
  float beta = 1.0f;

  buffer_t A = {0}, B = {0}, C = {0}, output_buf = {0};

  //set host
  A.host = (uint8_t*)&matrix_A[0]; B.host = (uint8_t*)&matrix_B[0]; C.host = (uint8_t*)&matrix_C[0]; output_buf.host = (uint8_t*)&output[0];
  //set stride
  A.stride[0] = B.stride[0] = C.stride[0] = output_buf.stride[0] = 1;
  A.stride[1] = B.stride[1] = C.stride[1] = output_buf.stride[1] = 2;
  //set extent
  A.extent[0] = B.extent[0] = C.extent[0] = output_buf.extent[0] = 2;
  A.extent[1] = B.extent[1] = C.extent[1] = output_buf.extent[1] = 2;
  //set element size
  A.elem_size = B.elem_size = C.elem_size = output_buf.elem_size = 4;

  int error = sgemm(alpha, &A, &B, beta, &C, &output_buf);

//  if (error) {
//    printf("Halide returned an error: %d\n", error);
//    return -1;
//  }

  for(int i = 0; i < 2; i++){
    for (int j = 0; j < 2; j++) {
      float val = output[i*2+j];
      printf("[%d, %d] : %f\n", i, j, val);
    }
  }
};
