/**
 * @file utils.c
 * @brief TODO
 *
 * @author misaka-10032 (longqic@andrew.cmu.edu)
 */

#include "utils.h"

#define EPSILON 1e-5

void BlobToImage(Blob<float> &blob, Image<float> &image) {
  int size = blob.count();
  memcpy(image.data(), blob.cpu_data(), size * sizeof(float));
}

bool BlobEqImage(const Blob<float> &blob, const Image<float> &image) {
  int size = blob.count();
  for (int i = 0; i < size; i++) {
    if (fabs(blob.cpu_data()[i]-image.data()[i]) > EPSILON) {
      int rest = i;
      int w = rest % blob.width(); rest /= blob.width();
      int h = rest % blob.height(); rest /= blob.height();
      int c = rest % blob.channels(); rest /= blob.channels();
      int n = rest;
      printf("mismatch at %d (%d, %d, %d, %d): %.10f vs %.10f\n",
             i, n, c, h, w,
             blob.cpu_data()[i], image.data()[i]);
      return false;
    }
  }
  return true;
}

void fillImage(Image<float>& image, float v) {
  int D = image.dimensions();
  int size = 1;
  for (int d = 0; d < D; d++) {
    size *= image.extent(d);
  }
  for (int i = 0; i < size; i++) {
    image.data()[i] = v;
  }
}
