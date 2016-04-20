# Halstm -- Checkpoint

## Work So Far

* Studied LSTM and Halide
* Designed code structure and test cases 
* Integrated [caffe-lstm](https://github.com/junhyukoh/caffe-lstm) into framework
* Implementation of basic components in LSTM: activation function, matrix multiplication, elementwise addition, etc.

### Code Structure

* `NetBuilder`
  * `Append(...)`: append layers.
  * `Criteria(...)`: define the criteria layer, e.g. softmax.
  * `Build()`: chain up layers, and build an entire pipeline.
* `Net`: a materialized network.
  * `Forward(...)`: do an actual forward.
  * `Backward(...)`: do an actual backward.
* `Layer`: a layer that defines pipeline operations only; not yet materialized.
  * `Forward(Func& in, Func& out)`: defines forward operations
  * `Backward(Func& out, Func& in)`: defines backward operations

### Test Case

[caffe-lstm](https://github.com/junhyukoh/caffe-lstm) has a well implemented LSTM layer. Our correctness goal is to match this result, given input, output, and weight the same. Now the LSTM part and its dependency have been extracted into our project; it can successfully compile and run.


## Problems We've Met

* Halide is a programming language designed for image processing. Using it to conduct linear algebra operations yields implementation difficulties to some degree. Especially, the order a Halide function defines for execution order is object to the intuition of matrix index's visiting. Trying to overcome this and accelerate the progress of implementation.

* Halide is released not for a very long time. Resource is limited when debugging the Halide program.

## Calibrated Goals and Deliverables

We may not finally be able to build an LSTM app, but still we want to ensure that we've implemented the LSTM layer correct. Possibly, we also want to see how fast it could be, and how we can improve it.

* As correctness check, we will match our result with that of [caffe-lstm](https://github.com/junhyukoh/caffe-lstm), given some randomly generated input/output/weight.
* As performance measure, we will stack several LSTM layers together, and see if the pipelined version can possibly be faster.

Possibly, we also want to chain up the forward and backward process and actually train on some dataset. 

## Refined Schedule

* Week4 (April 18 - 24, 2016)
  * Implement LSTM Forward; ensure correctness.

* Week5 (April 25 - May 1, 2016)
  * Implement LSTM Backward; ensure correctness.
  * Run performance test; adjust scheduling. 

* Week6 (2016 May 2 - 2016 May 8)
  * Adjust scheduling.
  * Train on real dataset (if time permits, because extra work is needed to reformat data and implement an IO module). 
