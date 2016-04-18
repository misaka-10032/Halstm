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

Var i,j;

template <class T>
class Activation {

public:

  Func tanh, sigmoid;
  Halide::Func input;

  Activation(Func in): input(in){
    tanh(i, j) = Halide::tanh(input(i, j));
    sigmoid(i, j) = 1.0f / (1.0f + Halide::fast_exp(input(i, j)));
  }
};

#endif //HALSTM_ACTIVATION_H
