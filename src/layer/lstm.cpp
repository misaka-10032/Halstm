/**
 * @file lstm.cpp
 * @brief Impl of lstm layer
 *
 * @author misaka-10032 (longqic@andrew.cmu.edu)
 */

#include "layers.h"
#include "maths.h"
#include "sgemm.h"
#include "Halide.h"

using namespace Halide;
using namespace halstm;


/**
 * Implementation of Forward Function
 */
void LstmLayer::Forward(Func &in, Func &out) {

}

void LstmLayer::Backward(Func &out, Func &in) {
  // TODO
}
