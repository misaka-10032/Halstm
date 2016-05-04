/**
 * @file utils.c
 * @brief TODO
 *
 * @author misaka-10032 (longqic@andrew.cmu.edu)
 */

#include "utils.h"

void blobToImage(Blob<float>& blob, Image<float>& image) {
  int size = blob.num() * blob.channels() * blob.height() * blob.width();
  memcpy(image.data(), blob.cpu_data(), size * sizeof(float));
}

bool blobEqImage(const Blob<float>& blob, const Image<float>& image) {
  int size = blob.num() * blob.channels() * blob.height() * blob.width();
  for (int i = 0; i < size; i++) {
    if (blob.cpu_data()[i] != image.data()[i]) {
      printf("mismatch at %d: %f vs %f\n", i,
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
