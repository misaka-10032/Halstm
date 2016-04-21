/**
 * @file layers.h
 * @brief Prototypes for layers.
 *
 * @author misaka-10032 (longqic@andrew.cmu.edu)
 */

#ifndef HALSTM_LAYERS_H
#define HALSTM_LAYERS_H

#include <Halide.h>
#include <maths.h>
#include <vector>
#include <string>
#include <memory>
#include "activation.h"

using namespace Halide;

class Layer {
public:

  virtual void forward(Func& in, Func& out);
  virtual void backward(Func& in, Func& out);
};

//class RNNLayer: public Layer{
//public:
//
//  RNNLayer(int T, int N, int H, int I):
//          T_(T), N_(N), H_(H), I_(I){}
//
//  // in(I_, N_, T_)
//  // out(H_, N_, T_)
//  void forward(Func &in, Func &out){
//
////    Var t, n, h, i;
//
//    in();

//    Func trans;  // (N_, H_)
//    for (int t = 0; t < T_; t++) {
//      // in(t): (N_, I_)
//      // Wxh: (I_, H_)
//      out(t) = dot(in(t), Wxh);
//      out(t) += dot(trans, Whh);
//      out(t) = tanh(out(t));
//      trans = out(t);
//    }
//  }
//};

template  <typename T>
class LSTMLayer: public Layer{
public:
  int I_; // input dimension
  int H_; // num of hidden units;
  int T_; // length of sequence
  int N_; // batch size

  Image<T> weight_i_;    // (4*H x I)
  Image<T> weight_h_;    // (4*H x H)
  Image<T> bias_;        // (1 x 4*H)


  Image<T>  bias_multiplier_;  // (T x N x 1)

  Image<T> top_;      // output values    (T x N x H)
  Image<T> cell_;     // memory cell      (T x N x H)
  Image<T> pre_gate_; // gate values before nonlinearty   (T x N x 4*H)
  Image<T> gate_;     // gate values after nonlinearty    (T x N x 4*H)

  Image<T> c_0_; // previous cell state value         (N x H)
  Image<T> h_0_; // previous hidden activation value  (N x H)
  Image<T> c_T_; // next cell state value             (N x H)
  Image<T> h_T_; // next hidden activation value      (N x H)

// intermediate values
  Image<T> h_to_gate_;   // (N x 4*H)
//  Image<T> h_to_h_;   (N_ x H_)

  LSTMLayer(int t, int H, int I, int N):
          T_(t), H_(H), I_(I), N_(N){}

  // bottom (T x N x I)
  // top    (T x N x I)
  void forward(Func& bottom, Func & out){
    Var i, j, k;

    Func top_data("top_data");
    top_data(i, j, k) = top_(i, j, k);
    bool clip;    //TODO: figure out what is this

    Func bias_multiplier("bias_multiplier");
    bias_multiplier(i, j, k) = bias_multiplier_(i, j, k);  // (T x N x 1)

    Func weight_i("weight_i");    weight_i(i, j) = weight_i_(i, j);
    Func weight_h("weight_h");    weight_h(i, j) = weight_h_(i, j);
    Func bias("bias");            bias(i, j) = bias_(i, j);

    Func pre_gate_data("pre_gate_data");  pre_gate_data(i, j, k) = pre_gate_(i, j, k);
    Func gate_data("gate_data");          gate_data(i, j, k) = gate_(i, j, k);
    Func cell_data("cell_data");          cell_data(i, j, k) = cell_(i, j, k);
    Func h_to_gate("h_to_gate");          h_to_gate(i, j) = h_to_gate_(i, j);

    // initialize previous state
    Func c_0 ("c_0_");
    Func h_0 ("h_0_");
    if (clip) {
      c_0(i, j) = c_T_(i, j);
      h_0(i, j) = h_T_(i, j);
    }else{
      c_0(i, j) = c_0_(i, j);
      h_0(i, j) = h_0_(i, j);
    }

    // compute input to hidden forward propagation
    pre_gate_data = halstm::matrix_dot(true, false, false, true, bottom, weight_i, N_, 4*H_, I_);
    Func multiplied_bias;
    multiplied_bias = halstm::matrix_dot(true, false, false, false, bias_multiplier, bias, N_, 4*H_, 1);
    pre_gate_data = halstm::matrix_add(true, pre_gate_data, multiplied_bias, N_, 4*H_);

    for (int t = 0; t < T_; t++){
      RDom t_(t-1, t);

      Func h_t("h_t");      // h to produce
      Func c_t("c_t");      // c to produce
      Func gate_t[4];      // 4 x N x H
      Func h_to_gate_t("h_to_gate+t");

      Func pre_gate_t("pre_gate_t"); pre_gate_t(i, j) = pre_gate_data(i, j, t);    // (N * 4*H)
      Func h_t_1("h_t_1");
      Func c_t_1("c_t_1");

      if(t > 0){
        h_t_1(i, j) = top_data(i, j, t-1);
        c_t_1(i, j) = cell_data(i, j, t-1);
      }else if (t == 0){
        h_t_1(i, j) = h_0(i, j);
        c_t_1(i, j) = c_0(i, j);
      }

      //hidden-to-hidden propagation
      h_to_gate = halstm::matrix_dot(false, false, false, true, h_t_1, weight_h, N_, 4*H_, H_);

      // combine hidden input and last layer input
      pre_gate_t = halstm::matrix_add(false, h_to_gate, pre_gate_t, N_, 4*H_);

      //apply nonlinearty
      RDom gates_range(0, H_, H_, 2*H_, 2*H_, 3*H_, 3*H_, 4*H_);

      Func pre_gate_t0("pre_gate_t0");
      pre_gate_t0(i, j) = pre_gate_t(gates_range.x, j);
      gate_t[0](i, j) = (halstm::Sigmoid_(pre_gate_t0))(i, j);
      Func pre_gate_t1("pre_gate_t1");
      pre_gate_t1(i, j) = pre_gate_t(gates_range.y, j);
      gate_t[1](i, j) = (halstm::Sigmoid_(pre_gate_t1))(i, j);
      Func pre_gate_t2("pre_gate_t2");
      pre_gate_t2(i, j) = pre_gate_t(gates_range.z, j);
      gate_t[2](i, j) = (halstm::Sigmoid_(pre_gate_t2))(i, j);
      Func pre_gate_t3("pre_gate_t3");
      pre_gate_t3(i, j) = pre_gate_t(gates_range.w, j);
      gate_t[3](i, j) = (halstm::Tanh_(pre_gate_t3))(i, j);

      Func forget_gate; forget_gate(i, j) = halstm::matrix_mul(gate_t[1], c_t_1);
      Func input_gate;  input_gate(i, j) = halstm::matrix_mul(gate_t[0], gate_t[3]);
      c_t(i, j) = halstm::matrix_add(false, forget_gate, input_gate, N_, H_);
      Func filter_gate;  filter_gate(i, j) = halstm::Tanh_(c_t);
      h_t(i, j) = halstm::matrix_mul(filter_gate, gate_t[2]);   //N_, H_

      out(i, j, t) = h_t(i, j);
    }
  }
};

#endif // HALSTM_LAYERS_H
