# Halstm -- when LSTM meets Halide

## Summary

Halstm is a high-performance implementation of LSTM based on Halide. It's portable to multiple platforms and architectures thanks to the powerful backend of Halide.

## Background

[Halide](http://halide-lang.org/) is a language for high-performance image processing; [LSTM](https://en.wikipedia.org/wiki/Long_short-term_memory) is one of the core layers in [RNN](https://en.wikipedia.org/wiki/Recurrent_neural_network). Basically, the forward and backward operators in neural networks are based on matrix operations. Therefore, Halide comes to play because whatever it's able to do on the images can also be done on the matrices. Halide saves extra efforts in writing boilerplate for optimization, and allows programmers to focus on the substantial scheduling work. Using Halide along may not results in significant faster code than the high-performance math libraries such as MLK. However, by fine-tuning a pipeline which minimizes work and has nice cache locality, it's believed that we can achieve comparable performance. After all, one of the most attractive feature of Halide is its compatibility across multiple platforms and architectures.

## Challenges

Challenge comes in two folds: on one hand, we haven't implemented LSTM before. We have used well implemented LSTM in [Lasagne](https://lasagne.readthedocs.org/en/latest/) and [Theano](http://deeplearning.net/software/theano/), but have not digged down into implementation details. The topology structure of LSTM makes it not trivial to conduct derivatives in the training process. A deep understanding of Recurrent Neural Network(RNN) and LSTM is required to engineering it from scratch. On the other hand, it may take some time for us to digest the abstraction and implementation of Halide. Halide provides excellent abstraction for developers to write tight code, but it still requires understanding on the scheduling policies and tricks to exploit program's parallelsim and locality properties, which demands research efforts and experiences.

## Goals

##### Plan to achieve

* Implement LSTM with Halide for CPU
* Achieve comparable benchmark with Theano and TensorFlow

##### Hope to achieve

* Optimize scheduling and beat Theano
* Implement sheduling for GPU 

## Resource

Ravi has implemented [Halide-NN](https://github.com/ravi-teja-mullapudi/Halide-NN), which implements basic layers in a CNN with CPU scheduling. Last year, Jeffrey and Jeff have implemented [Espresso](https://github.com/jczhang/espresso), which is CNN with GPU scheduling. We may reference these two projects in architecting the neural network framework. However, no one has implemented LSTM yet. For this part, we will read the [related papers](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf) and [tutorials](http://colah.github.io/posts/2015-08-Understanding-LSTMs/), and current implementations in [Theano](http://deeplearning.net/tutorial/lstm.html) and [TensorFlow](https://www.tensorflow.org/versions/r0.7/tutorials/recurrent/index.html).

## Schedule

##### Week1: Backgrounds (2016 April 1 - 2016 April 3)

* Go through Halide's tutorial, documents and source code to learn the image programming language.
* Get a high level understanding about RNN and LSTM, and their application on NLP tasks.

##### Week2: Theoretical Research (2016 April 4 - 2016 April 10)

* Conduct research on theory of LSTM. Understand the forward and backward propagation's mathmatical form and derivative method.
* Study current implementations of LSTM (e.g. TensorFlow).
* Explore options of Halide's implementation on deep learning models.

##### Week3: Coding Bootcamp (2016 April 11 - 2016 April 17)

* Start to implement the LSTM model in Halide.
* Prepare checkpoint report.

##### Week4: Robust Implementation (2016 April 18 - 2016 April 24)

* Engineering work on project debugging and optimization.

##### Week5: Scheduling Optimization (2016 April 25 - 2016 May 1)

* Start experiments on collected datasets.
* Explore scheduling policies in Halide to speed up the model's training and evaluation.

##### Week6: Experiments and Evaluation (2016 May 2 - 2016 May 8)

* Select one or more applications for evaluation and comparision with baselines.
* Analyze potential performance improvements or pitfalls.
* Final report and presentation preparation. 