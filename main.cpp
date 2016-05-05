#include <iostream>
#include <Halide.h>
#include "net.h"

using namespace std;
using namespace halstm;
using Halide::Image;

int main() {
  const int T = 10;
  const int N = 8;
  const int I = 6;
  const int H = 4;
  Image<float> data;
  Image<float> labels;
  Net net = NetBuilder(data, labels)
      .Append(LstmLayer::New(T, N, I, H))
      .Measure(Softmax::New())
      .Build();
  cout << "Hello, World!" << endl;
  return 0;
}
