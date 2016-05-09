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
import numpy as np


def main(args):
    ax = plt.subplot(111)
    width = .35
    lines = [57, 83]
    ind = np.arange(len(lines))
    ax.bar(ind+width, lines, width, color='b')
    ax.set_xticks(ind+1.5*width)
    ax.set_xticklabels(('Halstm', 'Caffe-lstm'))
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('arg', help='...')
    # parser.add_argument('-o', '--optional', action='store_true', help='...')
    # parser.add_argument('-i', '--input', type=int, help='...')
    main(parser.parse_args())
