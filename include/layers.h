/**
 * @file layers.h
 * @brief Prototypes for layers.
 *
 * @author misaka-10032 (longqic@andrew.cmu.edu)
 */

#ifndef HALSTM_LAYERS_H
#define HALSTM_LAYERS_H

#include <Halide.h>

using namespace Halide;



class Layer {
public:
  virtual void Forward(Func& in, Func& out) = 0;
  virtual void Backward(Func& out, Func& in) = 0;
};

class Criterion : public Layer {
public:
  virtual void Loss(Func& pred, Func& tgt, Func& loss) = 0;
};

class LstmLayer : public Layer {
public:
  LstmLayer(int T, int N, int H) :
      T_(T), N_(N), H_(H) {}
private:
  int T_; // sequence length
  int N_; // batch size
  int H_; // number of units
};

#endif // HALSTM_LAYERS_H
