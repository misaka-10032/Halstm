//
// Created by XiaotongSun on 16/4/18.
//

#ifndef HALSTM_ACTIVATION_H
#define HALSTM_ACTIVATION_H


/*
 * Implementation of layer's activation function
 */

#include "Halide.h"

//TODO: currently operate on 2-D array, see if need change

namespace halstm {
/*
 * Get a instance of tanh activation function
 * - input: the last stage Halide Func in pipeline
 */
    Halide::Func Tanh_(Halide::Func input) {
      Halide::Func result;

      Halide::Var x, y;
      result(x, y) = Halide::tanh(input(x, y));

      return result;
    }

/*
 * Get a instance of sigmoid activation function
 * - input: the last stage Halide Func in pipeline
 */
    Halide::Func Sigmoid_(Halide::Func input) {
      Halide::Func result;

      Halide::Var x, y;
      result(x, y) = 1.0f / (1.0f + Halide::fast_exp(-input(x, y)));

      return result;
    }
}
#endif //HALSTM_ACTIVATION_H
