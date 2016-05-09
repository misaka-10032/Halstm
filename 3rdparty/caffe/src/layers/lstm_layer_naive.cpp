#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
#include "CycleTimer.h"

//#define debug

namespace caffe {

    template <typename Dtype>
    void caffe_cpu_matrix_mul(bool transA, bool transB, const int M, const int N, const int K,
                              const Dtype alpha, const Dtype* A, const Dtype* B, const Dtype beta,
                              Dtype* C) {
//  C := alpha*op( A )*op( B ) + beta*C
      int i, j, k;
      int **AT;
      int **BT;

      if (transA) {
        AT = new int *[M];
        for(i = 0; i < M; i++)
          AT[i] = new int[K];
        for (i = 0; i < K; i++)
          for (j = 0; j < M; j++)
            AT[j][i] = A[i*M+j];
      }
      if (transB) {
        BT = new int *[K];
        for(i = 0; i < K; i++)
          BT[i] = new int[N];
        for (i = 0; i < N; i++)
          for (j = 0; j < K; j++) {
            BT[j][i] = B[i * K + j];
          }
      }

      for (i = 0; i < M; i++)
        for (j = 0; j < N; j++) {
          C[i*N+j] *= beta;
          float dot = 0.f;
          for (k = 0; k < K; k++)
            dot += (transA ? AT[i][k] : A[i*K+k])
                   * (transB ? BT[k][j] : B[k*N+j]);
          C[i*N+j] += alpha * dot;
        }

      if (transA) {
        for (int i = 0; i < M; i++) {
          delete[] AT[i];
        }
        delete[] AT;
      }
      if (transB) {
        for (int i = 0; i < K; i++) {
          delete[] BT[i];
        }
        delete[] BT;
      }
    }

    template <typename Dtype>
    inline Dtype sigmoid(Dtype x) {
      return 1. / (1. + exp(-x));
    }

    template <typename Dtype>
    void NaiveLstmLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                      const vector<Blob<Dtype>*>& top) {
      clipping_threshold_ = this->layer_param_.lstm_param().clipping_threshold();
      this->N_ = this->layer_param_.lstm_param().batch_size(); // batch_size
      this->H_ = this->layer_param_.lstm_param().num_output(); // number of hidden units
      this->I_ = bottom[0]->count() / bottom[0]->num(); // input dimension

      // Check if we need to set up the weights
      if (this->blobs_.size() > 0) {
        LOG(INFO) << "Skipping parameter initialization";
      } else {
        this->blobs_.resize(3);
        shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
                this->layer_param_.lstm_param().weight_filler()));

        // input-to-hidden weights
        // Intialize the weight
        vector<int> weight_shape;
        weight_shape.push_back(4*H_);
        weight_shape.push_back(I_);
        this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
        weight_filler->Fill(this->blobs_[0].get());

        // hidden-to-hidden weights
        // Intialize the weight
        weight_shape.clear();
        weight_shape.push_back(4*H_);
        weight_shape.push_back(H_);
        this->blobs_[1].reset(new Blob<Dtype>(weight_shape));
        weight_filler->Fill(this->blobs_[1].get());

        // If necessary, intiialize and fill the bias term
        vector<int> bias_shape(1, 4*H_);
        this->blobs_[2].reset(new Blob<Dtype>(bias_shape));
        shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
                this->layer_param_.lstm_param().bias_filler()));
        bias_filler->Fill(this->blobs_[2].get());
      }  // parameter initialization
      this->param_propagate_down_.resize(this->blobs_.size(), true);

      vector<int> cell_shape;
      cell_shape.push_back(N_);
      cell_shape.push_back(H_);
      this->c_0_.Reshape(cell_shape);
      this->h_0_.Reshape(cell_shape);
      this->c_T_.Reshape(cell_shape);
      this->h_T_.Reshape(cell_shape);
      this->h_to_h_.Reshape(cell_shape);

      vector<int> gate_shape;
      gate_shape.push_back(N_);
      gate_shape.push_back(4);
      gate_shape.push_back(H_);
      this->h_to_gate_.Reshape(gate_shape);
    }

    template <typename Dtype>
    void NaiveLstmLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                   const vector<Blob<Dtype>*>& top) {
      // Figure out the dimensions
      this->T_ = bottom[0]->num() / this->N_; // length of sequence

      CHECK_EQ(bottom[0]->num() % this->N_, 0) << "Input size "
                "should be multiple of batch size";
      CHECK_EQ(bottom[0]->count() / this->T_ / this->N_, this->I_) << "Input size "
                "incompatible with inner product parameters.";
      vector<int> original_top_shape;
      original_top_shape.push_back(T_*N_);
      original_top_shape.push_back(H_);
      top[0]->Reshape(original_top_shape);

      // Gate initialization
      vector<int> gate_shape;
      gate_shape.push_back(this->T_);
      gate_shape.push_back(this->N_);
      gate_shape.push_back(4);
      gate_shape.push_back(this->H_);
      pre_gate_.Reshape(gate_shape);
      gate_.Reshape(gate_shape);

      vector<int> top_shape;
      top_shape.push_back(T_);
      top_shape.push_back(N_);
      top_shape.push_back(H_);
      cell_.Reshape(top_shape);
      top_.Reshape(top_shape);
      top_.ShareData(*top[0]);
      top_.ShareDiff(*top[0]);

      // Set up the bias multiplier
      vector<int> multiplier_shape(1, this->N_*this->T_);
      bias_multiplier_.Reshape(multiplier_shape);
      caffe_set(bias_multiplier_.count(), Dtype(1),
                bias_multiplier_.mutable_cpu_data());
    }

    template <typename Dtype>
    void NaiveLstmLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                       const vector<Blob<Dtype>*>& top) {
#ifdef debug
  double start_time = CycleTimer::currentSeconds();
#endif

      CHECK_EQ(top[0]->cpu_data(), top_.cpu_data());
      Dtype* top_data = top_.mutable_cpu_data();
      const Dtype* bottom_data = bottom[0]->cpu_data();
      const Dtype* clip = NULL;
      if (bottom.size() > 1) {
        clip = bottom[1]->cpu_data();
        CHECK_EQ(bottom[1]->num(), bottom[1]->count());
      }
      const Dtype* weight_i = this->blobs_[0]->cpu_data();
      const Dtype* weight_h = this->blobs_[1]->cpu_data();
      const Dtype* bias = this->blobs_[2]->cpu_data();

      Dtype* pre_gate_data = pre_gate_.mutable_cpu_data();
      Dtype* gate_data = gate_.mutable_cpu_data();
      Dtype* cell_data = cell_.mutable_cpu_data();
      Dtype* h_to_gate = h_to_gate_.mutable_cpu_data();

      // Initialize previous state
      if (clip) {
        caffe_copy(this->c_0_.count(), this->c_T_.cpu_data(), this->c_0_.mutable_cpu_data());
        caffe_copy(this->h_0_.count(), this->h_T_.cpu_data(), this->h_0_.mutable_cpu_data());
      }
      else {
        caffe_set(this->c_0_.count(), Dtype(0.), this->c_0_.mutable_cpu_data());
        caffe_set(this->h_0_.count(), Dtype(0.), this->h_0_.mutable_cpu_data());
      }

      // Compute input to hidden Forward propagation
//      caffe_cpu_gemm(CblasNoTrans, CblasTrans, T_*N_, 4*H_, I_, Dtype(1.),
//                     bottom_data, weight_i, Dtype(0.), pre_gate_data);
      caffe_cpu_matrix_mul(false, true, this->T_*this->N_, 4*this->H_, this->I_, Dtype(1.),
                           bottom_data, weight_i, Dtype(0.), pre_gate_data);
//      caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, T_*N_, 4*H_, 1, Dtype(1.),
//                     bias_multiplier_.cpu_data(), bias, Dtype(1.), pre_gate_data);
      caffe_cpu_matrix_mul(false, false, this->T_*this->N_, 4*this->H_, 1, Dtype(1.),
                     bias_multiplier_.cpu_data(), bias, Dtype(1.), pre_gate_data);

#ifdef debug
  double before_T_time = CycleTimer::currentSeconds();
  printf("\n[Caffe LSTM] Before Entering T Loop Time : %e\n", before_T_time - start_time);
#endif

      // Compute recurrent Forward propagation
      for (int t = 0; t < this->T_; ++t) {
        Dtype* h_t = top_data + top_.offset(t);
        Dtype* c_t = cell_data + cell_.offset(t);
        Dtype* pre_gate_t = pre_gate_data + pre_gate_.offset(t);
        Dtype* gate_t = gate_data + gate_.offset(t);
        Dtype* h_to_gate_t = h_to_gate;
        const Dtype* clip_t = clip ? clip + bottom[1]->offset(t) : NULL;
        const Dtype* h_t_1 = t > 0 ? (h_t - top_.offset(1)) : h_0_.cpu_data();
        const Dtype* c_t_1 = t > 0 ? (c_t - cell_.offset(1)) : c_0_.cpu_data();

        // Hidden-to-hidden propagation
//        caffe_cpu_gemm(CblasNoTrans, CblasTrans, N_, 4*H_, H_, Dtype(1.),
//                       h_t_1, weight_h, Dtype(0.), h_to_gate);
        caffe_cpu_matrix_mul(false, true, N_, 4*H_, H_, Dtype(1.),
                             h_t_1, weight_h, Dtype(0.), h_to_gate);

        for (int n = 0; n < N_; ++n) {
          const bool cont = clip_t ? clip_t[n] : t > 0;
          if (cont) {
            caffe_add(4*H_, pre_gate_t, h_to_gate, pre_gate_t);
          }
          for (int d = 0; d < H_; ++d) {
            // Apply nonlinearity
            gate_t[d] = sigmoid(pre_gate_t[d]);
            gate_t[H_ + d] = cont ? sigmoid(pre_gate_t[H_ + d]) : Dtype(0.);
            gate_t[2*H_ + d] = sigmoid(pre_gate_t[2*H_ + d]);
            gate_t[3*H_ + d] = tanh(pre_gate_t[3*H_ + d]);

            // Compute cell : c(t) = f(t)*c(t-1) + i(t)*g(t)
            c_t[d] = gate_t[H_ + d] * c_t_1[d] + gate_t[d] * gate_t[3*H_ + d];
            h_t[d] = gate_t[2*H_ + d] * tanh(c_t[d]);
          }

          h_t += H_;
          c_t += H_;
          c_t_1 += H_;
          pre_gate_t += 4*H_;
          gate_t += 4*H_;
          h_to_gate_t += 4*H_;
        }
#ifdef debug
    double T_time = CycleTimer::currentSeconds();
    printf("[Caffe LSTM] T = %d, Time = %e\n", t, T_time-start_time);
#endif
      }
      // Preserve cell state and output value for truncated BPTT
      caffe_copy(N_*H_, cell_data + cell_.offset(T_-1), c_T_.mutable_cpu_data());
      caffe_copy(N_*H_, top_data + top_.offset(T_-1), h_T_.mutable_cpu_data());
    }

    template <typename Dtype>
    void NaiveLstmLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                        const vector<bool>& propagate_down,
                                        const vector<Blob<Dtype>*>& bottom) {
      const Dtype* top_data = top_.cpu_data();
      const Dtype* bottom_data = bottom[0]->cpu_data();
      const Dtype* clip = NULL;
      if (bottom.size() > 1) {
        clip = bottom[1]->cpu_data();
        CHECK_EQ(bottom[1]->num(), bottom[1]->count());
      }
      const Dtype* weight_i = this->blobs_[0]->cpu_data();
      const Dtype* weight_h = this->blobs_[1]->cpu_data();
      const Dtype* gate_data = gate_.cpu_data();
      const Dtype* cell_data = cell_.cpu_data();

      Dtype* top_diff = top_.mutable_cpu_diff();
      Dtype* pre_gate_diff = pre_gate_.mutable_cpu_diff();
      Dtype* gate_diff = gate_.mutable_cpu_diff();
      Dtype* cell_diff = cell_.mutable_cpu_diff();

      caffe_copy(N_*H_, c_T_.cpu_diff(), cell_diff + cell_.offset(T_-1));

      for (int t = T_-1; t >= 0; --t) {
        Dtype* dh_t = top_diff + top_.offset(t);
        Dtype* dc_t = cell_diff + cell_.offset(t);
        Dtype* gate_diff_t = gate_diff + gate_.offset(t);
        Dtype* pre_gate_diff_t = pre_gate_diff + pre_gate_.offset(t);
        Dtype* dh_t_1 = t > 0 ? top_diff + top_.offset(t-1) : h_0_.mutable_cpu_diff();
        Dtype* dc_t_1 = t > 0 ? cell_diff + cell_.offset(t-1) : c_0_.mutable_cpu_diff();
        const Dtype* clip_t = clip ? clip + bottom[1]->offset(t) : NULL;
        const Dtype* c_t = cell_data + cell_.offset(t);
        const Dtype* c_t_1 = t > 0 ? cell_data + cell_.offset(t-1) : c_0_.cpu_data();
        const Dtype* gate_t = gate_data + gate_.offset(t);

        for (int n = 0; n < N_; ++n) {
          const bool cont = clip_t ? clip_t[n] : t > 0;
          for (int d = 0; d < H_; ++d) {
            const Dtype tanh_c = tanh(c_t[d]);
            gate_diff_t[2*H_ + d] = dh_t[d] * tanh_c;
            dc_t[d] += dh_t[d] * gate_t[2*H_ + d] * (Dtype(1.) - tanh_c * tanh_c);
            dc_t_1[d] = cont ? dc_t[d] * gate_t[H_ + d] : Dtype(0.);
            gate_diff_t[H_ + d] = cont ? dc_t[d] * c_t_1[d] : Dtype(0.);
            gate_diff_t[d] = dc_t[d] * gate_t[3*H_ + d];
            gate_diff_t[3*H_ +d] = dc_t[d] * gate_t[d];

            pre_gate_diff_t[d] = gate_diff_t[d] * gate_t[d] * (Dtype(1.) - gate_t[d]);
            pre_gate_diff_t[H_ + d] = gate_diff_t[H_ + d] * gate_t[H_ + d]
                                      * (1 - gate_t[H_ + d]);
            pre_gate_diff_t[2*H_ + d] = gate_diff_t[2*H_ + d] * gate_t[2*H_ + d]
                                        * (1 - gate_t[2*H_ + d]);
            pre_gate_diff_t[3*H_ + d] = gate_diff_t[3*H_ + d] * (Dtype(1.) -
                                                                 gate_t[3*H_ + d] * gate_t[3*H_ + d]);
          }

          // Clip deriviates before nonlinearity
          if (clipping_threshold_ > Dtype(0.)) {
            caffe_bound(4*H_, pre_gate_diff_t, -clipping_threshold_,
                        clipping_threshold_, pre_gate_diff_t);
          }

          dh_t += H_;
          c_t += H_;
          c_t_1 += H_;
          dc_t += H_;
          dc_t_1 += H_;
          gate_t += 4*H_;
          gate_diff_t += 4*H_;
          pre_gate_diff_t += 4*H_;
        }

        // Backprop output errors to the previous time step
//        caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, N_, H_, 4*H_,
//                       Dtype(1.), pre_gate_diff + pre_gate_.offset(t),
//                       weight_h, Dtype(0.), h_to_h_.mutable_cpu_data());
        caffe_cpu_matrix_mul(false, false, N_, H_, 4*H_,
                       Dtype(1.), pre_gate_diff + pre_gate_.offset(t),
                       weight_h, Dtype(0.), h_to_h_.mutable_cpu_data());
        for (int n = 0; n < N_; ++n) {
          const bool cont = clip_t ? clip_t[n] : t > 0;
          const Dtype* h_to_h = h_to_h_.cpu_data() + h_to_h_.offset(n);
          if (cont) {
            caffe_add(H_, dh_t_1, h_to_h, dh_t_1);
          }
        }
      }

      if (this->param_propagate_down_[0]) {
        // Gradient w.r.t. input-to-hidden weight
//        caffe_cpu_gemm(CblasTrans, CblasNoTrans, 4*H_, I_, T_*N_, Dtype(1.),
//                       pre_gate_diff, bottom_data, Dtype(1.), this->blobs_[0]->mutable_cpu_diff());
        caffe_cpu_matrix_mul(true, false, 4*H_, I_, T_*N_, Dtype(1.),
                             pre_gate_diff, bottom_data, Dtype(1.), this->blobs_[0]->mutable_cpu_diff());
      }

      if (this->param_propagate_down_[1]) {
        // Gradient w.r.t. hidden-to-hidden weight
//        caffe_cpu_gemm(CblasTrans, CblasNoTrans, 4*H_, H_, (T_-1)*N_, Dtype(1.),
//                       pre_gate_diff + pre_gate_.offset(1), top_data,
//                       Dtype(1.), this->blobs_[1]->mutable_cpu_diff());
        caffe_cpu_matrix_mul(true, false, 4*H_, H_, (T_-1)*N_, Dtype(1.),
                             pre_gate_diff + pre_gate_.offset(1), top_data,
                             Dtype(1.), this->blobs_[1]->mutable_cpu_diff());

        // Add Gradient from previous time-step
//        caffe_cpu_gemm(CblasTrans, CblasNoTrans, 4*H_, H_, 1, Dtype(1.),
//                       pre_gate_diff, h_0_.cpu_data(),
//                       Dtype(1.), this->blobs_[1]->mutable_cpu_diff());
        caffe_cpu_matrix_mul(true, false, 4*H_, H_, 1, Dtype(1.),
                             pre_gate_diff, h_0_.cpu_data(),
                             Dtype(1.), this->blobs_[1]->mutable_cpu_diff());
      }
      if (this->param_propagate_down_[2]) {
        // Gradient w.r.t. bias
        caffe_cpu_gemv(CblasTrans, T_*N_, 4*H_, Dtype(1.), pre_gate_diff,
                       bias_multiplier_.cpu_data(), Dtype(1.),
                       this->blobs_[2]->mutable_cpu_diff());
      }
      if (propagate_down[0]) {
        // Gradient w.r.t. bottom data
//        caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, T_*N_, I_, 4*H_, Dtype(1.),
//                       pre_gate_diff, weight_i, Dtype(0.), bottom[0]->mutable_cpu_diff());
        caffe_cpu_matrix_mul(false, false, T_*N_, I_, 4*H_, Dtype(1.),
                             pre_gate_diff, weight_i, Dtype(0.), bottom[0]->mutable_cpu_diff());
      }
    }

#ifdef CPU_ONLY
    STUB_GPU(NaiveLstmLayer);
#endif

    INSTANTIATE_CLASS(NaiveLstmLayer);
    REGISTER_LAYER_CLASS(NaiveLstm);

}  // namespace caffe
