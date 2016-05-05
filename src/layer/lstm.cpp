/**
 * @file lstm.cpp
 * @brief Impl of lstm layer
 *
 * @author misaka-10032 (longqic@andrew.cmu.edu)
 */

#include "layers.h"
#include "maths.h"

namespace halstm {

  LstmLayer::LstmLayer(int T, int N, int I, int H) :
      T_(T), N_(N), I_(I), H_(H) {
    Var x, y, z;

    // give names
    Wih_ = Func("Wih_");
    Whh_ = Func("Whh_");
    b_ = Func("b_");
    b_mul_ = Func("b_mul_");

    dWih_ = Func("dWih_");
    dWhh_ = Func("dWhh_");
    db_ = Func("db_");

    h0_ = Func("h0_");
    c0_ = Func("c0_");
    for (int t = 0; t < T; t++) {
      std::string t_str = std::to_string(t);
      h_.push_back(Func("h_["+t_str+"]"));
      c_.push_back(Func("c_["+t_str+"]"));
    }

    // initialize
    b_mul_(x, y, z) = 1.f;
    h0_(x, y) = 0.f;
    c0_(x, y) = 0.f;
  }

  /**
   * in:  (I_, N_, T_)
   * out: (H_, N_, T_)
   */
  void LstmLayer::Forward(Func &in, Func &out) {
    // TODO: schedule
    Var x("x"), y("y"), z("z");
    Func pre_gate("pre_gate");  // (4*H_, N_, T_)
    Func bias("bias");
    // (1, N_, T_) dot (4*H_, 1)  -> (4*H_, N_, T_)
    Dot_3dx2d(false, false, b_mul_, b_, x, y, z, 1, bias);
    // (I_, N_, T_) dot (4*H_, I_)  -> (4*H_, N_, T_)
    Dot_3dx2d(false, true, in, Wih_, x, y, z, I_, pre_gate);
    pre_gate(x, y, z) += bias(x, y, z);
    pre_gate.compute_root();
    out(x, y, z) = (float) 0;

    for (int t = 0; t < T_; t++) {
      // TODO: delete debug
      printf("t=%d\n", t);
      // TODO: clip if needed
      Func& h_prev = t == 0 ? h0_ : h_[t-1];
      Func& c_prev = t == 0 ? c0_ : c_[t-1];
      Func pre_gate_t("pre_gate_t");
      pre_gate_t(x, y) = pre_gate(x, y, t);  // TODO: optimize
      Func h_to_gate("h_to_gate");
      // (H_, N_) dot (4*H_, H_) -> (4H_, N_)
      Dot_2dx2d(false, true, h_prev, Whh_, x, y, H_, h_to_gate);
      if (t > 0) {
        pre_gate_t(x, y) += h_to_gate(x, y);
      }

      // go through gates
      Sigmoid_2d(RDom(0, H_, 0, N_), pre_gate_t, pre_gate_t);
      if (t == 0) {
        Set_2d(RDom(H_, H_, 0, N_), 0, pre_gate_t);
      } else {
        Sigmoid_2d(RDom(H_, H_, 0, N_), pre_gate_t, pre_gate_t);
      }
      Sigmoid_2d(RDom(2*H_, H_, 0, N_), pre_gate_t, pre_gate_t);
      Tanh_2d(RDom(3*H_, H_, 0, N_), pre_gate_t, pre_gate_t);

      // now pre_gate is gate
      c_[t](x, y) = pre_gate_t(x+H_, y) * c_prev(x, y) +
          pre_gate_t(x, y) * pre_gate_t(x+3*H_, y);
      h_[t](x, y) = pre_gate_t(x+2*H_, y) * tanh(c_[t](x, y));

      // TODO: better schedule
      c_[t].compute_root();
      h_[t].compute_root();
    }

    // TODO: delete debug
    out.trace_stores();
    // update outputs
    for (int t = 0; t < T_; t++) {
      out(x, y, t) = h_[t](x, y);
    }
  }

  void LstmLayer::Backward(Func &dout, Func &din) {
    // TODO
  }

  vector<Image<float>> LstmLayer::params() {
    // TODO
    return vector<Image<float>>();
  }

  vector<Image<float>> LstmLayer::dparams() {
    // TODO
    return vector<Image<float>>();
  }

  vector<Func> LstmLayer::f_params() {
    // TODO
    return vector<Func>();
  }

  vector<Func> LstmLayer::f_dparams() {
    // TODO
    return vector<Func>();
  }
}