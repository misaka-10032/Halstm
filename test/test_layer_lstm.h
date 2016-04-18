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
#include "layers.h"
#include "caffe/proto/caffe.pb.h"
#include "caffe/common_layers.hpp"
#include "caffe/blob.hpp"
#include "caffe/util/math_functions.hpp"

using namespace std;
using caffe::Blob;

#define T_ 4
#define N_ 3
#define I_ 8
#define H_ 6


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
    vector<Blob<float>*>* bottom = NewBlobVec(T_*N_, 1, I_, 1);
    vector<Blob<float>*>* top = NewBlobVec(T_*N_, 1, H_, 1);
    caffe::LstmLayer<float>* caffeLstmLayer = NewCaffeLstmLayer();
    caffeLstmLayer->LayerSetUp(*bottom, *top);
    caffeLstmLayer->Forward(*bottom, *top);
    DelBlobVec(bottom);
    DelBlobVec(top);
    DelCaffeLstmLayer(caffeLstmLayer);
  }

  void TestAddition() {
    TS_ASSERT(1 + 1 > 1);
    TS_ASSERT_EQUALS(1 + 1, 2);
  }

  void TestMultiplication() {
    TS_TRACE("Starting multiplication test");
    TS_ASSERT_EQUALS(2 * 2, 4);
    TS_TRACE("Finishing multiplication test");
  }
};

#endif // HALSTM_TEST_LAYER_H
