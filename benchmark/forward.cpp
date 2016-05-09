/**
 * @file forward.cpp
 * @brief benchmarks for forward
 *
 * @author misaka-10032 (longqic@andrew.cmu.edu)
 */

#include <vector>
#include <caffe/filler.hpp>
#include "benchmark.h"
#include "caffe/blob.hpp"
#include "caffe/common_layers.hpp"
#define protected public
#include "layers.h"
#include "caffe/common_layers.hpp"
#undef protected

using caffe::Blob;
using caffe::FillerParameter;
using caffe::UniformFiller;

const int RUNS = 8;
const int nNs = 8;
const int Ns[nNs] = {16, 32, 48, 64, 80, 96, 112, 128};
const int T = 16;
const int I = 128;
const int H = 256;

static vector<Blob<float>*>* NewBlobVec(int N, int C, int H, int W) {
  vector<Blob<float>*>* blobVec = new vector<Blob<float>*>();
  Blob<float>* blob = new Blob<float>(N, C, H, W);
  FillerParameter filler_param;
  filler_param.set_min(-.1f);
  filler_param.set_max(.1f);
  UniformFiller<float> filler(filler_param);
  filler.Fill(blob);
  blobVec->push_back(blob);
  return blobVec;
}

static void DelBlobVec(vector<Blob<float>*>* blobVec) {
  while (!blobVec->empty()) {
    Blob<float>* blob = blobVec->back();
    delete blob;
    blobVec->pop_back();
  }
  delete blobVec;
}

static void InitImage(Image<float>& image, int n) {
  for (int i = 0; i < n; i++) {
    image.data()[i] =
        static_cast<float> (rand()) / static_cast <float> (RAND_MAX);
  }
}

int main(int argc, char** argv) {
  printf("caffe lstm\n");
  for (int i = 0; i < nNs; i++) {
    int N = Ns[i];
    caffe::LayerParameter layer_param;
    caffe::LSTMParameter* lstm_param =
        layer_param.mutable_lstm_param();
    lstm_param->set_batch_size(N);
    lstm_param->set_num_output(H);
    lstm_param->mutable_weight_filler()->set_type("uniform");
    lstm_param->mutable_weight_filler()->set_min(-.1f);
    lstm_param->mutable_weight_filler()->set_max(.1f);
    lstm_param->mutable_bias_filler()->set_type("uniform");
    lstm_param->mutable_bias_filler()->set_min(-.1f);
    lstm_param->mutable_bias_filler()->set_max(-.1f);
    caffe::LstmLayer<float>* caffeLstmLayer =
        new caffe::LstmLayer<float>(layer_param);

    vector<Blob<float>*>* bottom = NewBlobVec(T*N, 1, I, 1);
    vector<Blob<float>*>* top = NewBlobVec(T*N, 1, H, 1);
    caffeLstmLayer->LayerSetUp(*bottom, *top);
    caffeLstmLayer->Reshape(*bottom, *top);

    double time = benchmark(RUNS, 1, [&]() {
      caffeLstmLayer->Forward(*bottom, *top);
    });
    printf("%e ", time);

    delete caffeLstmLayer;
  }
  printf("\n");

  printf("caffe lstm naive\n");
  for (int i = 0; i < nNs; i++) {
    int N = Ns[i];
    caffe::LayerParameter layer_param;
    caffe::LSTMParameter* lstm_param =
        layer_param.mutable_lstm_param();
    lstm_param->set_batch_size(N);
    lstm_param->set_num_output(H);
    lstm_param->mutable_weight_filler()->set_type("uniform");
    lstm_param->mutable_weight_filler()->set_min(-.1f);
    lstm_param->mutable_weight_filler()->set_max(.1f);
    lstm_param->mutable_bias_filler()->set_type("uniform");
    lstm_param->mutable_bias_filler()->set_min(-.1f);
    lstm_param->mutable_bias_filler()->set_max(-.1f);
    caffe::NaiveLstmLayer<float>* naiveLstmLayer =
        new caffe::NaiveLstmLayer<float>(layer_param);

    vector<Blob<float>*>* bottom = NewBlobVec(T*N, 1, I, 1);
    vector<Blob<float>*>* top = NewBlobVec(T*N, 1, H, 1);
    naiveLstmLayer->LayerSetUp(*bottom, *top);
    naiveLstmLayer->Reshape(*bottom, *top);

    double time = benchmark(RUNS, 1, [&]() {
      naiveLstmLayer->Forward(*bottom, *top);
    });
    printf("%e ", time);

    delete naiveLstmLayer;
  }
  printf("\n");

  printf("halstm\n");
  for (int i = 0; i < nNs; i++) {
    int N = Ns[i];
    Image<float> Wih(I, 4*H, "Wih"); InitImage(Wih, I*4*H);
    Image<float> Whh(H, 4*H, "Whh"); InitImage(Whh, H*4*H);
    Image<float> b(4*H, 1, "b");     InitImage(b, 4*H);
    halstm::LstmLayer lstmLayer(T, N, I, H);
    lstmLayer.Wih_ = Func(Wih);
    lstmLayer.Whh_ = Func(Whh);
    lstmLayer.b_ = Func(b);

    Image<float> in(I, N, T, "in"); InitImage(in, I*N*T);
    Image<float> out(H, N, T, "out");
    Func fin(in), fout("fout");
    lstmLayer.Forward(fin, fout);

    double time = benchmark(RUNS, 1, [&]() {
      fout.realize(out);
    });
    printf("%e ", time);
  }
  printf("\n");

  return 0;
}

