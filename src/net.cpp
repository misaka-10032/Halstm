/**
 * @file net.cpp
 * @brief TODO
 *
 * @author misaka-10032 (longqic@andrew.cmu.edu)
 */

#include <boost/range/adaptor/reversed.hpp>
#include "net.h"

namespace halstm {

  NetBuilder::NetBuilder(Image<float> &in, Image<float> &out) {
    fin_ = Func(in);
    fout_ = Func(out);
    fcurr_ = fin_;
  }

  NetBuilder& NetBuilder::Append(shared_ptr<Layer> &&layer) {
    vector<Image<float>> params = layer->params();
    vector<Image<float>> dparams = layer->dparams();
    vector<Func> f_dparams = layer->f_dparams();
    net_.params_.insert(net_.params_.end(),
                        params.begin(), params.end());
    net_.dparams_.insert(net_.dparams_.end(),
                         dparams.begin(), dparams.end());
    net_.f_dparams_.insert(net_.f_dparams_.end(),
                           f_dparams.begin(), f_dparams.end());

    layers_.push_back(layer);
    Func out;
    layer->Forward(curr_func_, out);
    fcurr_ = out;

    return *this;
  }

  NetBuilder& NetBuilder::Measure(shared_ptr<Criterion> &&criterion) {
    // in is scores
    // out is probs
    Func pred;
    Func dout;
    Func loss;
    Func din;
    criterion->Forward(fcurr_, pred);
    criterion->Loss(pred, fout_, dout, loss);
    criterion->Backward(dout, din);
    fcurr_ = din;
    criterion_ = criterion;

    // maintain funcs_ and images_
    funcs_.push_back(loss);
    images_.push_back(Image<float>(1));

    return *this;
  }

  Net& NetBuilder::Build() {
    using boost::adaptors::reverse;
    // backward
    for (auto layer : reverse(layers_)) {
      Func din;
      layer->Backward(fcurr_, din);
      fcurr_ = din;
    }

    // maintain funcs_ and images_
    for (auto dparam : reverse(net_.dparams_)) {
      net_.images_.push_back(dparam);
    }
    for (auto f_dparam: reverse(net_.f_dparams_)) {
      net_.funcs_.push_back(f_dparams);
    }

    return net_;
  }

}