/**
 * @file utils.c
 * @brief TODO
 *
 * @author misaka-10032 (longqic@andrew.cmu.edu)
 */

#include "utils.h"

void blobToImage(Blob<float>& blob, Image<float>& image) {
  // go reverse order
  // blob  n, c, h, w
  // image
  cout << "blobToImage" << endl;
  int size = blob.num() * blob.channels() * blob.height() * blob.width();
  memcpy(image.data(), blob.cpu_data(), size * sizeof(float));
//  int idxBlob = 0;
//  for (int n = 0; n < blob.num(); n++) {
//    for (int c = 0; c < blob.channels(); c++) {
//      for (int h = 0; h < blob.height(); h++) {
//        for (int w = 0; w < blob.width(); w++) {
//          int idxImg = w;
//          idxImg = (idxImg * blob.height()) + h;
//          idxImg = (idxImg * blob.channels()) + c;
//          idxImg = (idxImg * blob.num()) + n;
//          printf("(%d, %d, %d, %d) -> %d\n", n, c, h, w, idxImg);
//          image.data()[idxImg] = blob.cpu_data()[idxBlob++];
//        }
//      }
//    }
//  }
}

bool blobEqImage(const Blob<float>& blob, const Image<float>& image) {
  // go reverse order
  int idxBlob = 0;
  for (int n = 0; n < blob.num(); n++) {
    for (int c = 0; c < blob.channels(); c++) {
      for (int h = 0; h < blob.height(); h++) {
        for (int w = 0; w < blob.width(); w++) {
          int idxImg = w;
          idxImg = (idxImg * blob.height()) + h;
          idxImg = (idxImg * blob.channels()) + c;
          idxImg = (idxImg * blob.num()) + n;
          if (image.data()[idxImg] != blob.cpu_data()[idxBlob++]) {
            cout << "mismatch at (n, c, h, w)=(" <<
            n << ", " << c << ", " << h << ", " << w <<
            ")" << endl;
            return false;
          }
        }
      }
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
