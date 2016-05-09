/**
 * @file lstm.cpp
 * @brief Impl of lstm layer
 *
 * @author misaka-10032 (longqic@andrew.cmu.edu)
 */

#include "layers.h"
#include "maths.h"
#include "CycleTimer.h"
#include "schedule.h"

#define debug 1

namespace halstm {

  LstmLayer::LstmLayer(int T, int N, int I, int H) :
      T_(T), N_(N), I_(I), H_(H) {
    Var x, y, z;

    Wih_ = Func("Wih_");
    Whh_ = Func("Whh_");
    b_ = Func("b_");

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
    h0_(x, y) = 0.f;
    c0_(x, y) = 0.f;
  }

  /**
   * in:  (I_, N_, T_)
   * out: (H_, N_, T_)
   */
  void LstmLayer::Forward(const Func& in, Func &out) {
    Var x("x"), y("y"), z("z");
    Func pre_gate_ub("pre_gate_ub");  // (4*H_, N_, T_)

    // (I_, N_, T_) dot (4*H_, I_)  -> (4*H_, N_, T_)
    Dot_3dx2d(false, true, in, Wih_, x, y, z, I_, N_, pre_gate_ub);

    Var yz("yz");
    Func pre_gate("pre_gate");
    pre_gate(x, y, z) = pre_gate_ub(x, y, z) + b_(x, 0);
    pre_gate.fuse(y, z, yz).parallel(yz).vectorize(x, VEC_SZ);
    pre_gate.compute_root();

    for (int t = 0; t < T_; t++) {
      Func &h_prev = t == 0 ? h0_ : h_[t - 1];
      Func &c_prev = t == 0 ? c0_ : c_[t - 1];
      Func pre_gate_t("pre_gate_t");
      pre_gate_t(x, y) = pre_gate(x, y, t);
      Func h_to_gate("h_to_gate");

      // (H_, N_) dot (4*H_, H_) -> (4H_, N_)
      Dot_2dx2d(false, true, h_prev, Whh_, x, y, H_, h_to_gate);

      if (t > 0) {
        pre_gate_t(x, y) += h_to_gate(x, y);
      }

      // go through gates (H_, N_)
      Func gate[4];
      gate[0](x, y) = 1.0f / (1.0f + fast_exp(-pre_gate_t(x, y)));
      if (t == 0) {
        gate[1](x, y) = 0.0f;
      } else {
        gate[1](x, y) = 1.0f / (1.0f + fast_exp(-pre_gate_t(x + H_, y)));
      }
      gate[2](x, y) = 1.0f / (1.0f + fast_exp(-pre_gate_t(x + 2 * H_, y)));
      gate[3](x, y) = tanh(pre_gate_t(x + 3 * H_, y));

//      for (int i = 0; i < 4; i++) {
//        gate[i].parallel(y).vectorize(x, VEC_SZ).compute_root();
//      }

      gate[0].compute_at(c_[t], y);
      gate[1].compute_at(c_[t], y);
      gate[2].compute_at(h_[t], y);
      gate[3].compute_at(c_[t], y);

      c_[t](x, y) = gate[1](x, y) * c_prev(x, y) +
                    gate[0](x, y) * gate[3](x, y);
      h_[t](x, y) = gate[2](x, y) * tanh(c_[t](x, y));

      c_[t].parallel(y).vectorize(x, VEC_SZ).compute_root();
      h_[t].parallel(y).vectorize(x, VEC_SZ).compute_root();

      c_[t].compute_root();
      h_[t].compute_root();
    }

    out(x, y, z) = 0.f;
    for (int t = 0; t < T_; t++) {
      out(x, y, t) = h_[t](x, y);
    }
    out.fuse(y, z, yz).parallel(yz).vectorize(x, VEC_SZ);
  }

  void LstmLayer::Backward(const Func& out, const Func& dout,
                           const Func& in, Func& din) {
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