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
#include <vector>

using namespace Halide;
using std::vector;
using std::shared_ptr;

namespace halstm {

  class Layer {
  public:
    virtual void Forward(Func &in, Func &out) = 0;
    virtual void Backward(Func &dout, Func &din) = 0;

    virtual vector<Image<float>> params() = 0;
    virtual vector<Image<float>> dparams() = 0;
    virtual vector<Func> f_dparams() = 0;
  };

  class Criterion : public Layer {
  public:
    virtual void Loss(Func &pred, Func &tgt, Func &dout, Func &loss) = 0;
    virtual vector<Image<float>> params() { return vector<Image<float>>(); }
    virtual vector<Image<float>> dparams() { return vector<Image<float>>(); }
    virtual vector<Func> f_dparams() { return vector<Func>(); }
  };

  class Softmax : public Criterion {
  public:
    Softmax() { /* TODO */ }
//    static shared_ptr<Softmax> New() {
//      return shared_ptr<Softmax>(new Softmax());
//    }
    virtual void Forward(Func &in, Func &out) = 0;
    virtual void Backward(Func &dout, Func &din) = 0;
    void Loss(Func &pred, Func &tgt, Func &dout, Func &loss) {
      // TODO
    }

  };

  class LstmLayer : public Layer {
  public:
    LstmLayer(int T, int N, int I, int H);
    static shared_ptr<LstmLayer> New(int T, int N, int I, int H) {
      return shared_ptr<LstmLayer>(new LstmLayer(T, N, I, H));
    }

    virtual void Forward(Func &in, Func &out);
    virtual void Backward(Func &dout, Func &din);

    virtual vector<Image<float>> params();
    virtual vector<Image<float>> dparams();
    virtual vector<Func> f_dparams();

  protected:
    int T_; // length of sequence
    int N_; // batch size
    int H_; // num of hidden units;
    int I_; // input dimension

    // parameters
    Image<float> weight_i_;    // (4*H x I)
    Image<float> weight_h_;    // (4*H x H)
    Image<float> bias_;        // (1 x 4*H)
    Image<float> bias_multiplier_;  // (T x N x 1)

    // dparams
    Image<float> dweight_i_;
    Image<float> dweight_h_;
    Image<float> dbias_;
    Func f_dweight_i_;
    Func f_dweight_h_;
    Func f_dbias_;

    std::vector<Image<float>> top_;
    std::vector<Image<float>> cell_;

    Image<float> c_0_; // previous cell state value         (N x H)
    Image<float> h_0_; // previous hidden activation value  (N x H)
    Image<float> c_T_; // next cell state value             (N x H)
    Image<float> h_T_; // next hidden activation value      (N x H)
  };
}
#endif // HALSTM_LAYERS_H
