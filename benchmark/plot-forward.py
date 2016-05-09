#!/usr/bin/env python
# encoding: utf-8
"""
Created by misaka-10032 (longqic@andrew.cmu.edu).
All rights reserved.

TODO: purpose
"""
__author__ = 'misaka-10032'

import argparse
import matplotlib.pyplot as plt


def main(args):
    n = [16, 32, 48, 64, 80, 96, 112, 128]
    caffe = [8.917000e-03, 1.625000e-02, 2.415800e-02, 3.066200e-02,
            3.731400e-02, 4.385000e-02, 5.031400e-02, 5.701900e-02]
    naive = [7.159800e-01, 1.401799e+00, 2.248344e+00, 2.728365e+00,
             3.411234e+00, 4.075604e+00, 4.773326e+00, 5.317739e+00]
    halstm = [9.257000e-03, 1.105700e-02, 1.230900e-02, 1.431400e-02,
              1.711800e-02, 2.034800e-02, 2.260600e-02, 3.106500e-02]

    line_caffe, = plt.plot(n, caffe, 'r-')
    line_halstm, = plt.plot(n, halstm, 'b-')
    # line_naive, = plt.plot(n, naive, 'g-')
    plt.legend([line_caffe, line_halstm],
               ['Caffe-lstm', 'Halstm'])
    # plt.legend([line_naive, line_caffe, line_halstm],
    #            ['Naive-lstm', 'Caffe-lstm', 'Halstm'])
    plt.xlabel('N')
    plt.ylabel('Time (s)')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('arg', help='...')
    # parser.add_argument('-o', '--optional', action='store_true', help='...')
    # parser.add_argument('-i', '--input', type=int, help='...')
    main(parser.parse_args())
