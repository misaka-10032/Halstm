/**
 * @file lstm.cpp
 * @brief Impl of lstm layer
 *
 * @author misaka-10032 (longqic@andrew.cmu.edu)
 */

#include "layers.h"
#include "maths.h"
#include "CycleTimer.h"

#define debug 1

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
#ifdef debug
  double start_time = CycleTimer::currentSeconds();
#endif
    Var x("x"), y("y"), z("z");
    Func pre_gate("pre_gate");  // (4*H_, N_, T_)
    Func bias("bias");

    // (1, N_, T_) dot (4*H_, 1)  -> (4*H_, N_, T_)
    Dot_3dx2d(false, false, b_mul_, b_, x, y, z, 1, bias);
    // (I_, N_, T_) dot (4*H_, I_)  -> (4*H_, N_, T_)
    Dot_3dx2d(false, true, in, Wih_, x, y, z, I_, pre_gate);
    pre_gate(x, y, z) += bias(x, y, z);
    out(x, y, z) = (float) 0;

    // scheduling for pre-loop
    bias.compute_at(pre_gate, x);
    pre_gate.compute_root();
    pre_gate.parallel(y);

    for (int t = 0; t < T_; t++) {
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

      pre_gate_t.compute_root();
      pre_gate_t.parallel(y);

      // go through gates (H_, N_)
      Func gate[4];
      gate[0](x, y) = 1.0f / (1.0f + fast_exp(-pre_gate_t(x, y)));
      if (t == 0) {
        gate[1](x, y) = 0.0f;
      }else{
        gate[1](x, y) = 1.0f / (1.0f + fast_exp(-pre_gate_t(x+H_, y)));
      }
      gate[2](x, y) = 1.0f / (1.0f + fast_exp(-pre_gate_t(x+2*H_, y)));
      gate[3](x, y) = tanh(pre_gate_t(x+3*H_, y));


      c_[t](x, y) = gate[1](x, y) * c_prev(x, y) +
                    gate[0](x, y) * gate[3](x, y);
      h_[t](x, y) = gate[2](x, y) * tanh(c_[t](x, y));

      // Scheduling for t in T
      Var xin, xout, yin, yout;
      Var gate_tile_idx;

      gate[0].compute_at(c_[t], x);
      gate[0].parallel(y);
      gate[1].compute_at(c_[t], x);
      gate[1].parallel(y);
      gate[2].compute_at(h_[t], x);
      gate[2].parallel(y);
      gate[3].compute_at(c_[t], x);
      gate[3].parallel(y);

      //TODO: how to pipelining through T?
//      if (t < T_-1){
//        c_[t].compute_at(c_[t+1], y);
//        h_[t].compute_at(h_[t+1], y);
//      }else{
      c_[t].compute_root();
      h_[t].compute_root();
//      }

      c_[t].parallel(y);
      h_[t].parallel(y);
    }

    // TODO: delete debug
    //out.trace_stores();
    // update outputs
    for (int t = 0; t < T_; t++) {
      out(x, y, t) = h_[t](x, y);
    }
  }

  void LstmLayer::Backward(Func &dout, Func &din) {
    Var x("x"), y("y"), z("z");
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