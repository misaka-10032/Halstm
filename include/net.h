/**
 * @file net.h
 * @brief Prototypes for net.
 *
 * @author misaka-10032 (longqic@andrew.cmu.edu)
 */

#ifndef HALSTM_NET_H
#define HALSTM_NET_H

#include <vector>
#include <Halide.h>
#include "layers.h"

using std::shared_ptr;
using std::vector;

namespace halstm {

  class Net {
    friend class NetBuilder;

  public:
    void Forward(Image<float> &in, Image<float> &out);
    void Backward(Image<float> &diff);
    void Step(Image<float> &in, Image<float> &out);

  private:
    // params_ and dparams_ have one-to-one mapping
    vector<Image<float>> params_;
    vector<Image<float>> dparams_;
    vector<Func> f_dparams_;

    // funcs_ and images_ have one-to-one mapping
    // funcs to be realized, including f_dparams and others
    vector<Func> funcs_;
    // realized images, including dparams_ and others
    vector<Image<float>> images_;

    Pipeline pipeline_;
  };

  class NetBuilder {
  public:
    NetBuilder(Image<float> &in, Image<float> &out);
    NetBuilder& Append(shared_ptr<Layer>&& layer);
    NetBuilder& Measure(shared_ptr<Criterion>&& criterion);
    Net& Build();

  private:
    Net net_;
    vector<Func> grads_;
    Func fcurr_, fin_, fout_;
    vector<shared_ptr<Layer>> layers_;
    shared_ptr<Criterion> criterion_;
  };

}
#endif // HALSTM_NET_H
