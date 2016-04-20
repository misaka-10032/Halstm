/**
 * @file layers.h
 * @brief Prototypes for layers.
 *
 * @author misaka-10032 (longqic@andrew.cmu.edu)
 */

#ifndef HALSTM_LAYERS_H
#define HALSTM_LAYERS_H

#include <Halide.h>
#include <vector>
#include <string>
#include <memory>

using namespace Halide;

class Layer {
public:

  virtual void forward(Func& in, Func& out);
  virtual void backward(Func& in, Func& out);
};

class RNNLayer: public Layer{
public:

  int T_;   // sequence length
  int N_;   // batch size
  int H_;   // hidden(output) dimension
  int I_;   // input dimension

  Func Whh;
  Func Wxh;

  RNNLayer(int T, int N, int H, int I):
          T_(T), N_(N), H_(H), I_(I){}

  // in(I_, N_, T_)
  // out(H_, N_, T_)
  void forward(Func &in, Func &out){

//    Var t, n, h, i;

    in();

//    Func trans;  // (N_, H_)
//    for (int t = 0; t < T_; t++) {
//      // in(t): (N_, I_)
//      // Wxh: (I_, H_)
//      out(t) = dot(in(t), Wxh);
//      out(t) += dot(trans, Whh);
//      out(t) = tanh(out(t));
//      trans = out(t);
//    }
  }
};

#endif // HALSTM_LAYERS_H
