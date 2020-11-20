#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Cross entropy functions
# (c) Zexun Chen 2020-11-20
# sxtpy2010@gmail.com

import numpy as np
from util.cross_entropy import LZ_cross_entropy
from util.entropy import LZ_entropy


def cumulative_LZ_CE(W1_list, W2, PTs_list,
                     e=2):
    """ Cumulative LZ cross entropy
    :param W1_list: nested list filled in many alters' sequences
    :param W2: ego's sequence
    :param PTs_list: nested list filled in many alters' PTs
    :param e: minimum length of sequence
    :return: cumulative LZ cross entropy
    """

    alters_len = [len(x) for x in W1_list]
    ego_len = len(W2)

    wb = []
    alters_L = []
    for W1, PTs in zip(W1_list, PTs_list):
        alters_L.append(LZ_cross_entropy(W1, W2, PTs, lambdas=True, e=e))
        # count how many element > 1, which is wb
        wb.append(sum(1 for x in alters_L[-1] if x > 1))

    alters_Lmax = np.amax(alters_L, axis=0)
    sum_L = sum(alters_Lmax)
    ave_length = np.average(wb, weights=alters_len)

    return (1.0 * ego_len / sum_L) * np.log2(ave_length)


def recursive_cumulative_LZ_CE():
    """ Recursive computation of Cumulative LZ cross entropy"""

    return
