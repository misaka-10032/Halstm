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
    virtual void Forward(const Func& in, Func& out) = 0;

    virtual void Backward(const Func& out, const Func& dout,
                          const Func& in, Func &din) = 0;

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

  class LstmLayer : public Layer {
  public:
    LstmLayer(int T, int N, int I, int H);
    static shared_ptr<LstmLayer> New(int T, int N, int I, int H) {
      return shared_ptr<LstmLayer>(new LstmLayer(T, N, I, H));
    }

    virtual void Forward(const Func& in, Func& out);
    virtual void Backward(const Func& out, const Func &dout,
                          const Func& in, Func &din);

    virtual vector<Image<float>> params();
    virtual vector<Image<float>> dparams();
    virtual vector<Func> f_params();  // TODO: make super virtual method
    virtual vector<Func> f_dparams();

  protected:
    int T_; // length of sequence
    int N_; // batch size
    int H_; // num of hidden units;
    int I_; // input dimension

    // (x, y, z) corresponds to Halide (reverse) order
    // parameters
    Func Wih_;    // (I_, 4*H_)
    Func Whh_;    // (H_, 4*H_)
    Func b_;      // (4*H_, 1)
    Func b_mul_;  // (1, N_, T_)

    // gradients
    Func dWih_;   // (I_, 4*H_)
    Func dWhh_;   // (H_, 4*H_)
    Func db_;     // (4*H_, 1)

    Func h0_;     // (H_, N_)
    Func c0_;     // (H_, N_)
    vector<Func> h_;  // T_ elements of shape (H_, N_)
    vector<Func> c_;  // T_ elements of shape (H_, N_)
  };

  class Softmax : public Criterion {
  public:
    Softmax() { /* TODO */ }

//    static shared_ptr<Softmax> New() {
//      return shared_ptr<Softmax>(new Softmax());
//    }
    virtual void Forward(const Func& in, Func &out) = 0;

    virtual void Backward(const Func& out, const Func &dout,
                          const Func& in, Func &din) = 0;

    void Loss(Func &pred, Func &tgt, Func &dout, Func &loss) {
      // TODO
    }

  };
}

#endif // HALSTM_LAYERS_H
