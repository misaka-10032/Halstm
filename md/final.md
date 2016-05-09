# HaLSTM: An LSTM Implementation in Halide

## Summary

HaLSTM is an efficient Long short-term memory neuron network's implementation in Halide. Halide has some nice properties as a domian specific language that motivates us to use it for a deep learning project: it enables pipelining multiple computational stages and decoupling the algorithm and the scheduling of it. This project correctly achieves correctness and performance optimization for the evaluation of LSTM model, and the result shows that Halide could be considered as an deasible option for quick development and fine-tuning the model in machine learning area.

## Background

#### LSTM Neural Network

Long short-term memory (LSTM) is a recurrent neural network (RNN) architecture that is well-suited to learn from experience to classify, process and predict time series, especially when there are very long time lags of unknown size between important events. One of LSTM's successful application is in NLP area, where it is capable of fine-tuning the model's ability to handle long-term dependency. Compared to other famous deep learning models, LSTM has more complex topology in each neuron. This makes it more difficult to implement LSTM and exploit potential optimizations.

![](https://raw.githubusercontent.com/misaka-10032/Halstm/gh-pages/assets/lstm-mechnism.png)
> Image from [colah's blog](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)

From engineer's point of view, LSTM can be visualized like this.

![](https://raw.githubusercontent.com/misaka-10032/Halstm/gh-pages/assets/lstm-impl.png)
> Image from [Nvidia's blog](https://devblogs.nvidia.com/parallelforall/optimizing-recurrent-neural-networks-cudnn-5/)

First step is matrix multiplication $P_t=[W_f W_o W_z W_i]\,X_t+[b_f b_o b_z b_i]$ and $Q_t=[R_f R_o R_z R_i] H_{t-1}$. Second step is element-wise addition $PG_t=P_t+Q_t$. Third step is to go through gates such as $\sigma$ and $\tanh$. Finally, there would be several element-wise operations before producing $H_t$ and $C_t$ and entering next time step. Note that $P_t, t\in[0,T)$ can be computed ahead of time, because we know all the $X_t$'s in this layer.

#### Halide

[Halide](http://halide-lang.org/) is an domain-specific language embedded in C++ designed for image processing. While we find it's potential to be applied in deep learning area, since the evaluation and training of a neural networks involves lots of matrix operations. The semantics of Halide could be smoothly applied to describe these operations. Halide provides properties of constructing pipelines and decoupling algorithm and scheduling, which can make the code more concise and emperical.

<!--
Halide abstracts operations into `Var`s and `Func`s. It separates a program into _define_,  _schedule_ and _realize_ parts. This is an example of _define_ part of _blur_ operation (code from [official website](http://halide-lang.org/)).

```cpp
  Func blur_x, blur_y;
  Var x, y, xi, yi;

  // The algorithm - no storage or order
  blur_x(x, y) = (input(x-1, y) + input(x, y) + input(x+1, y))/3;
  blur_y(x, y) = (blur_x(x, y-1) + blur_x(x, y) + blur_x(x, y+1))/3;
```

This is _schedule_ part.

```
  blur_y.tile(x, y, xi, yi, 256, 32)
        .vectorize(xi, 8).parallel(y);
  blur_x.compute_at(blur_y, x).vectorize(x, 8);
```

This is _realize_ part.

```
  Image<float> image(M, N);
  blur_y.realize(image);
```
-->

## Implementation

#### LSTM-forward in Halide

Halide definition and scheduling for LSTM-forward is written in `void LstmLayer::Forward(const Func& in, Func& out)` of _lstm.cpp_. 

* `in` is a $T\times N\times I$ tensor
* `out` is a $T\times N \times H$ tensor

where

* $T$ is sequence length
* $N$ is batch size
* $I$ is dimension of input layer
* $H$ is dimension of output (hidden) layer

First, it goes through a linear transform $P=WX+b$.

```
Dot_3dx2d(false, true, in, Wih_, x, y, z, I_, N_, pre_gate_ub);
pre_gate(x, y, z) = pre_gate_ub(x, y, z) + b_(x, 0);
```

Second, it enters a loop of time steps. In each step $t$, it first has a linear transform on the previous time step $Q_t=R_t H_t$, and do an element-wise addition with it $PG_t = R_t + P_t$.

```
Dot_2dx2d(false, true, h_prev, Whh_, x, y, H_, h_to_gate);
pre_gate_t(x, y) += h_to_gate(x, y);
```

Then, it goes through a set of gates.

```
Func gate[4];
gate[0](x, y) = 1.0f / (1.0f + fast_exp(-pre_gate_t(x, y)));    
gate[1](x, y) = 1.0f / (1.0f + fast_exp(-pre_gate_t(x + H_, y)));
gate[2](x, y) = 1.0f / (1.0f + fast_exp(-pre_gate_t(x + 2 * H_, y)));
gate[3](x, y) = tanh(pre_gate_t(x + 3 * H_, y));
```

Finally, intermediate states $H_t$ and $C_t$ are produced.

```
c_[t](x, y) = gate[1](x, y) * c_prev(x, y) +
              gate[0](x, y) * gate[3](x, y);
h_[t](x, y) = gate[2](x, y) * tanh(c_[t](x, y));
```

#### Correctness Test

To ensure our implementation is correct, [Caffe-lstm](https://github.com/junhyukoh/caffe-lstm) is integrated into our project unchanged. We control input and weights the same, and compare the output. As we use different scheduling, result can be a little bit different; any difference within _1e-3_ is deemed acceptable.

#### Developing Productivity

To implement the evaluation of a deep learning model, one can use Halide to define a pipeline of operations on data. HaLstm replace layers and LSTM network unit's inner structure with Halide pipelines. Compared to common programming languages, the Halide code can be very concise.


One of Halide's nice properties is its productivity for developers. Most of the complex logic in LSTM can be expressed in Halide quite concisely and elegantly. It provides developers with opportunities to fastly iterate over their code-base and verify different ideas. Below is a table of comparison on lines of code between LSTM's caffe implementation and Halide's implementation on the forward function. 

![](https://raw.githubusercontent.com/misaka-10032/Halstm/gh-pages/assets/lines.png)

## Performance Optimization

#### Exploiting Basic Parallelism

In each units of LSTM deep nets, element-wise operations are used heavily. Each gate's output are applied to historical and state data all in this manner. This provides us potential to exploit  more parallelsim and vectorization, since the computation during this process are independent to each other without further communication.

In HaLstm, all element-wise operations are expressed in matrix scalar addition and multiplicaiton. Our policy to optimize for this kind of computation is quite straight-forward: parallelizing by matrix's row dimension, and for each row, vectorize the computation by using SSE to compute in 4-wide vectors.

![](https://raw.githubusercontent.com/misaka-10032/Halstm/gh-pages/assets/matrix-simd.png)

#### Pipelining Multiple Stages
We have made fully use of Halide's advantage on pipelining multiple computational stages, both within and across the units of the deep nets. The idea is, instead of saving each computational stage's fully intermediate result in memory, computation of one stage will be inlined or partially saved for the next stage. Thus the producer-consumer locality can be efficiently exploited.

For two adjacent computational stages A and B, there're three choices to decide how to schedule them according to the properties of the runtime :

- __Full Pipelining__
	Inlining computation of A within B. Could be applied when the program is bandwidth bound. In HaLstm, many trivial operations are fully pipelined by default to avoid extra memory access. 
- __Storing Partial Result__ 
	For certain computing dimension in B, compute A on demand. Storing partial result of previous stage may exploit producer-consumer locality better compared to applying barriers. In HaLstm, we used this method for four gates within the stage of producing one unit's final output.
- __Barriers__
	Evaluate A completely before B starts. Could be applied when stage A's full result will be needed in stage B, or fully compute stage A may yield performance benefit. In HaLstm, each layer's parameter-input computation will be conducted in this way, due to our specialized optimization for the matrix multiplication.
	
![](https://raw.githubusercontent.com/misaka-10032/Halstm/gh-pages/assets/pipelines.png)

#### Optimization on Matrix Multiplication

* __Block-wise multiplication for better locality__

![](https://raw.githubusercontent.com/misaka-10032/Halstm/gh-pages/assets/mul-block.png)
> Slide from [lecture](http://15418.courses.cs.cmu.edu/spring2016/lecture/dnneval/slide_023)

In Halide, it can be interpreted as

```
Func AA("AA"), BB("BB");
Var xi("xi"), xo("xo"), yi("yi"), yo("yo"), xy("xy");
AA(xi, yi, xo, yo) = A_(xi + xo * TILE_SZ, yi + yo * TILE_SZ);
BB(xi, yi, xo, yo) = B_(xi + xo * TILE_SZ, yi + yo * TILE_SZ);

RDom rd(0, TILE_SZ, 0, rsize / TILE_SZ);
Func CC("CC");
CC(xi, xo, yi, yo) = 0.f;
CC.fuse(yo, yi, y).parallel(y);
CC(xi, xo, yi, yo) += BB(xi, rd.x, xo, rd.y) * AA(rd.x, yi, rd.y, yo);

CC.update()
  .reorder({rd.x, xi, yi, rd.y, xo, yo})
CC.compute_root();

C(x, y) = CC(x % TILE_SZ, x / TILE_SZ, y % TILE_SZ, y / TILE_SZ);
```

* __SIMD within block__

![](https://raw.githubusercontent.com/misaka-10032/Halstm/gh-pages/assets/mul-simd.png)
> Slide from [lecture](http://15418.courses.cs.cmu.edu/spring2016/lecture/dnneval/slide_026)

One more line of scheduling is needed,

```
CC.update()
  .reorder({rd.x, xi, yi, rd.y, xo, yo})
  .parallel(yo).vectorize(xi);  // <--
```

## Experiment & Analysis

Halide generates optimized code in runtime. When being `realize` for the first time, it includes both JIT time and run time; in second time, the generated code will be cached, so then the time will be the actual run time. `benchmark(...)` is a helper function to obtain the actual run time.

All experiments are done on [MBP2015](https://support.apple.com/kb/SP719?locale=en_US) with 2.5GHz quad-core Intel Core i7 processor. Horizontally, the results are compared with [Caffe-lstm](https://github.com/junhyukoh/caffe-lstm) . Vertically, they are also compared with themselves for different scheduling strategies.

Parameter settings are

* Sequence length $T=16$
* Batch size $N=32$
* Input dimension $I=32$
* Output (hidden) dimension $H=64$

#### Optimization1: Matrix Multiplication

Without any scheduling,

<!--
| Caffe-lstm | Halstm (before) | Speedup |
|:-:|:-:|:-:|
| 0.00483937 | 0.014822 | 0.33 |
-->

<table style="display:block; margin-left:auto; margin-right:auto; text-align:center;">
<tr>
<th>Caffe-lstm</th>
<th>Halstm (before)</th>
<th>Speedup</th>
</tr>
<tr>
<td>5.47 ms</td>
<td>14.82 ms</td>
<td>0.37x</td>
</tr>
</table>

Now let's begin to close the gap! Halide provides a profiling tool. If we set environment variable `HL_JIT_TARGET=host-profile`, we can get

```
pre_gate:              201.181900ms   (31%)
h_to_gate$1:           29.072832ms    (4%)
h_to_gate$2:           28.954476ms    (4%)
...
```

Notable computation time concentrates in `pre_gate` and `h_to_gate$i` (`$i` indicates in $i$th loop). Trace back to code, `pre_gate` is a _3d-by-2d_ matrix multiplication; `h_to_gate$i` are _2d-by-2d_ matrix multiplications. All other operations are element-wise. Therefore, we begin to optimize matrix multiplication (see previous section for detail). After optimization, we get

<table style="display:block; margin-left:auto; margin-right:auto; text-align:center">
<tr>
<th>Halstm (before)</th>
<th>Halstm (after)</th>
<th>Speedup</th>
</tr>
<tr>
<td>14.82 ms</td>
<td>6.51 ms</td>
<td>2.28x</td>
</tr>
</table>

#### Optimization2: SIMD and Multi-Thread Execution

Besides matrix multiplication, There are several expensive element-wise operation such as sigmoid function $\sigma$ and $\tanh$. Element-wise parallelism can be exploit through SIMD and Multi-Thread Execution. For example,

```
c_[t].parallel(y).vectorize(x, VEC_SZ).compute_root();
h_[t].parallel(y).vectorize(x, VEC_SZ).compute_root();
```

<!--
To measure the effect of parallelizing element-wise operations in our LSTM model, we conduct a group of experiment with the setting of with and without this optimization (parallelize on matrix's rows and vectorize within each row). The experiment is done on a single layer with 16 units. Batch size is 32, with input dimension of 128 and hidden state dimension of 128. The results are showed in the below table.
-->

We measured performance before and after the optimization.

<table style="display:block; margin-left:auto; margin-right:auto; text-align:center">
<tr>
<th>Before</th>
<th>After</th>
<th>Speedup</th>
</tr>
<tr>
<td>6.51 ms</td>
<td>2.87 ms</td>
<td>2.27x</td>
</tr>
</table>

<!--
| Without Parallelism | With Parallelism | Speedup |
|:-:|:-:|:-:|
| 85.92ms | 61.42ms | 1.39x |
-->

#### Optimization3: Pipelining 

Originally we put barriers for each stage in the series of element-wise operations. However, they are not necessary. If computed inline, several memory operations can be saved. Further, better producer-consumer locality could be exploited if we store partial results. We were not sure which one would be the best, so we carried out three experiments. Barrier can be implemented through `compute_root()`; fully inlining can be implemented through `compute_inline()`; partial-result can be implemented through `compute_at()`.


<table style="display:block; margin-left:auto; margin-right:auto; text-align:center">
<tr>
<th>Barrier</th>
<th>Inline</th>
<th>Partial</th>
</tr>
<tr>
<td>2.87 ms</td>
<td>2.74 ms</td>
<td>2.46 ms</td>
</tr>
</table>


<!--
To measure the effect of pipelining multiple computational stages, we test our model's performance with two settings: with and without pipeling optimizaitons. The experiment is done on a single layer with 16 units. Batch size is 32, with input dimension of 128 and hidden state dimension of 128. The results are showed in the below table. 


| Without Pipilining | With Pipelining | Speedup |
|:-:|:-:|:-:|
| 85.92ms | 61.42ms | 1.39x |


By observing the experiement result, we can argue that pipelining multiple stages achieves moderate speed-up, due to better producer-consumer locality.
-->

We observe that partial-result achieves best performance.

#### Put all together

<table style="display:block; margin-left:auto; margin-right:auto; text-align:center">
<tr>
<th>Caffe-lstm</th>
<th>Halstm (after)</th>
<th>Speedup</th>
</tr>
<tr>
<td>5.47 ms</td>
<td>2.46 ms</td>
<td>2.22x</td>
</tr>
</table>

We also changed several parameter settings

* $T=16$
* $N=16, 32, 48, 64, 80, 96, 112, 128$
* $I=128$
* $H=256$

![](https://raw.githubusercontent.com/misaka-10032/Halstm/gh-pages/assets/caffe-vs-halstm.png)

The actual meaning of having a larger $N$ is to make the network have a higher throuput. As is indicated in the figure, our approach achieves about 2x speedup with CPU scheduling.


## References

[1]Jonathan Ragan-Kelley, Connelly Barnes, Andrew Adams, Sylvain Paris, Frdo Durand, and Saman Amarasinghe. 2013. Halide: a language and compiler for optimizing parallelism, locality, and recomputation in image processing pipelines. In Proceedings of the 34th ACM SIGPLAN conference on Programming language design and implementation (PLDI ’13). ACM, New York, NY, USA, 519-530.

[2]S. Hochreiter, J. Schmidhuber, Long short-term memory, Neural Computation, 9 (1997), pp. 1735–1780

[3]Ian Goodfellow, Yoshua Bengio, and Aaron Courville, Sequence Modeling: Recurrent and Recursive Nets, Deep Learning, Book in preparation for MIT Press, http://www.deeplearningbook.org

[4]Jia, Yangqing and Shelhamer, Evan and Donahue, Jeff and Karayev, Sergey and Long, Jonathan and Girshick, Ross and Guadarrama, Sergio and Darrell, Trevor. Caffe: Convolutional Architecture for Fast Feature Embedding. 2014. arXiv:1408.5093

## Collaborative Work Division

Equal amount of work is done by us.

## Epilog

As we know, Caffe-lstm (and all other ML frameworks) uses linear algebra library for matrix operations (e.g. vecLib on Mac), which is highly optimized under the hood. Curious though, what if we don't use these libraries?

![](https://raw.githubusercontent.com/misaka-10032/Halstm/gh-pages/assets/naive.png)

Oh Jesus...
