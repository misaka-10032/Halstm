/**
 * @file test_layer.h
 * @brief Prototypes for test_layer.
 *
 * @author misaka-10032 (longqic@andrew.cmu.edu)
 */

#ifndef HALSTM_TEST_LAYER_H
#define HALSTM_TEST_LAYER_H

#include <vector>
#include <cxxtest/TestSuite.h>
#include "caffe/proto/caffe.pb.h"

// trick to use protected field
#define protected public
#include "layers.h"
#include "caffe/common_layers.hpp"
#undef protected

#include "caffe/blob.hpp"
#include "caffe/util/math_functions.hpp"
#include "utils.h"

using namespace std;
using caffe::Blob;

#define T_ 2
#define N_ 3
#define I_ 4
#define H_ 5

class TestLstmLayer : public CxxTest::TestSuite {
public:
  caffe::LstmLayer<float>* NewCaffeLstmLayer() {
    caffe::LayerParameter layer_param;
    caffe::LSTMParameter* lstm_param =
        layer_param.mutable_lstm_param();
    lstm_param->set_batch_size(N_);
    lstm_param->set_num_output(H_);
    lstm_param->mutable_weight_filler()->set_type("uniform");
    lstm_param->mutable_weight_filler()->set_min(-0.01);
    lstm_param->mutable_weight_filler()->set_max(0.01);
    lstm_param->mutable_bias_filler()->set_type("constant");
    lstm_param->mutable_bias_filler()->set_value(0);
    caffe::LstmLayer<float>* layer =
        new caffe::LstmLayer<float>(layer_param);
    return layer;
  }

  void DelCaffeLstmLayer(caffe::LstmLayer<float>* layer) {
    delete layer;
  }

  vector<Blob<float>*>* NewBlobVec(int N, int C, int H, int W) {
    vector<Blob<float>*>* blobVec = new vector<Blob<float>*>();
    Blob<float>* blob = new Blob<float>(N, C, H, W);
    caffe::FillerParameter filler_param;
    filler_param.set_min(-0.1);
    filler_param.set_max(0.1);
    caffe::UniformFiller<float> filler(filler_param);
    filler.Fill(blob);
    blobVec->push_back(blob);
    return blobVec;
  }

  void DelBlobVec(vector<Blob<float>*>* blobVec) {
    while (!blobVec->empty()) {
      Blob<float>* blob = blobVec->back();
      delete blob;
      blobVec->pop_back();
    }
    delete blobVec;
  }

  void TestForward() {
    TS_TRACE("TestForward...");
    vector<Blob<float>*>* bottom = NewBlobVec(T_*N_, 1, I_, 1);
    vector<Blob<float>*>* top = NewBlobVec(T_*N_, 1, H_, 1);
    caffe::LstmLayer<float>* caffeLstmLayer = NewCaffeLstmLayer();

    caffeLstmLayer->LayerSetUp(*bottom, *top);
    caffeLstmLayer->Forward(*bottom, *top);
    TS_TRACE("caffe layer setup!");

    Image<float> in(I_, N_, T_, "in");
    Image<float> out(H_, N_, T_, "out");
    // setup out bottom
    blobToImage(*(*bottom)[0], in);
    TS_TRACE("in setup!");

    Var x, y, z;
    Func fin("fin");
    fin(x, y, z) = in(x, y, z);

    // set up our weights
    halstm::LstmLayer lstmLayer(T_, N_, I_, H_);
    TS_TRACE("LstmLayer setup!");
    Image<float> Wih(I_, 4*H_, "Wih");
    Image<float> Whh(H_, 4*H_, "Whh");
    Image<float> b(4*H_, 1, "b");
    blobToImage(*caffeLstmLayer->blobs_[0], Wih);
    blobToImage(*caffeLstmLayer->blobs_[1], Whh);
    blobToImage(*caffeLstmLayer->blobs_[2], b);
//    fillImage(Wih, 1.f);
//    fillImage(Whh, 1.f);
//    fillImage(b, 1.f);

    lstmLayer.Wih_ = Func(Wih);
    lstmLayer.Whh_ = Func(Whh);
    lstmLayer.b_ = Func(b);
    TS_TRACE("weights setup!");

    Func fout("fout");
    lstmLayer.Forward(fin, fout);
    TS_TRACE("forward success!");
    // TODO: delete debug
    fout.compile_to_lowered_stmt("fout.html", {}, HTML);
    out = fout.realize(H_, N_, T_);
    TS_TRACE("realize success!");
    TS_ASSERT(blobEqImage(*(*top)[0], out));

    DelBlobVec(bottom);
    DelBlobVec(top);
    DelCaffeLstmLayer(caffeLstmLayer);
  }
};

#endif // HALSTM_TEST_LAYER_H
