//
// Created by XiaotongSun on 16/4/18.
//

#ifndef HALSTM_ACTIVATION_H
#define HALSTM_ACTIVATION_H


/*
 * Implementation of layer's activation function
 */

#include "Halide.h"

using namespace Halide;

class Activation {

public:
  Func input;
  Func sigmoid;

  Var x, y;

  Activation(Func in): input(in){
    sigmoid(x, y) = 1.0f / ( 1.0f + Halide::fast_exp(-input));
  }
};

#endif //HALSTM_ACTIVATION_H
