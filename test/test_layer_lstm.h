/**
 * @file test_layer.h
 * @brief Prototypes for test_layer.
 *
 * @author misaka-10032 (longqic@andrew.cmu.edu)
 */

#ifndef HALSTM_TEST_LAYER_H
#define HALSTM_TEST_LAYER_H

#include <cxxtest/TestSuite.h>
#include "layers.h"

class TestLstmLayer : public CxxTest::TestSuite {
public:
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
