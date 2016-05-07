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
  void LstmLayer::Forward(const Func& in, Func &out) {
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
    Var par, xin, xout, yin, yout;
    pre_gate.compute_root();
    pre_gate.parallel(y).vectorize(x, 4);

    for (int t = 0; t < T_; t++) {
      Func &h_prev = t == 0 ? h0_ : h_[t - 1];
      Func &c_prev = t == 0 ? c0_ : c_[t - 1];
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
      } else {
        gate[1](x, y) = 1.0f / (1.0f + fast_exp(-pre_gate_t(x + H_, y)));
      }
      gate[2](x, y) = 1.0f / (1.0f + fast_exp(-pre_gate_t(x + 2 * H_, y)));
      gate[3](x, y) = tanh(pre_gate_t(x + 3 * H_, y));


      c_[t](x, y) = gate[1](x, y) * c_prev(x, y) +
                    gate[0](x, y) * gate[3](x, y);
      h_[t](x, y) = gate[2](x, y) * tanh(c_[t](x, y));


      // Scheduling for single iteration
      Var x_outer, y_outer, x_inner, y_inner;
      Var x_inner_outer, y_inner_outer, x_vectors, y_pairs;
      Var fuse_idx;
      int x_tile = 16, y_tile = 8;

      // gates' staging: inline vs. compute_root

      gate[0].compute_at(c_[t], x);
      gate[1].compute_at(c_[t], x);
      gate[2].compute_at(h_[t], x);
      gate[3].compute_at(c_[t], x);

//      gate[0].compute_root();
//      gate[1].compute_root();
//      gate[2].compute_root();
//      gate[3].compute_root();

      gate[0].vectorize(x, 4);
      gate[1].vectorize(x, 4);
      gate[2].vectorize(x, 4);
      gate[3].vectorize(x, 4);

      gate[0].parallel(y);
      gate[1].parallel(y);
      gate[2].parallel(y);
      gate[3].parallel(y);


      // gates's optimization: parallel or tile:

//      gate[0].tile(x, y, x_outer, y_outer, x_inner, y_inner, x_tile, y_tile)
//              .fuse(x_outer, y_outer, gate_fuse_idx)
//              .parallel(gate_fuse_idx);
//      gate[1].tile(x, y, x_outer, y_outer, x_inner, y_inner, x_tile, y_tile)
//              .fuse(x_outer, y_outer, gate_fuse_idx)
//              .parallel(gate_fuse_idx);
//      gate[2].tile(x, y, x_outer, y_outer, x_inner, y_inner, x_tile, y_tile)
//              .fuse(x_outer, y_outer, gate_fuse_idx)
//              .parallel(gate_fuse_idx);
//      gate[3].tile(x, y, x_outer, y_outer, x_inner, y_inner, x_tile, y_tile)
//              .fuse(x_outer, y_outer, gate_fuse_idx)
//              .parallel(gate_fuse_idx);


      //TODO: how to pipelining through T?

      c_[t].compute_root();
      h_[t].compute_root();

      c_[t].parallel(y);
      h_[t].parallel(y);

//      c_[t].tile(x, y, x_outer, y_outer, x_inner, y_inner, x_tile, y_tile)
//              .fuse(x_outer, y_outer, fuse_idx)
//              .parallel(fuse_idx);
//      h_[t].tile(x, y, x_outer, y_outer, x_inner, y_inner, x_tile, y_tile)
//              .fuse(x_outer, y_outer, fuse_idx)
//              .parallel(fuse_idx);
//      c_[t].tile(x_inner, y_inner, x_inner_outer, y_inner_outer, x_vectors, y_pairs, 4, 4)
//              .vectorize(x_vectors).unroll(y_pairs);
//      h_[t].tile(x_inner, y_inner, x_inner_outer, y_inner_outer, x_vectors, y_pairs, 4, 4)
//              .vectorize(x_vectors).unroll(y_pairs);
    }

    for (int t = 0; t < T_; t++) {
      out(x, y, t) = h_[t](x, y);
    }
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