#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Cross entropy functions
# (c) Zexun Chen 2020-11-20
# sxtpy2010@gmail.com

import numpy as np
from util.cross_entropy import LZ_cross_entropy
from util.entropy import LZ_entropy


def cumulative_LZ_CE(W1_list, W2, PTs_list, individual=False, ego_include=False,
                     e=2):
    """ Cumulative LZ cross entropy
    :param W1_list: nested list filled in many alters' sequences
    :param W2: ego's sequence
    :param PTs_list: nested list filled in many alters' PTs
    :param individual: list all cumulative cross entropy add alters one by one
    :param ego_include: whether the cumulative_LZ_CE includes ego
    :param e: minimum length of sequence
    :return: cumulative LZ cross entropy

    Bagrow, James P., Xipei Liu, and Lewis Mitchell. "Information flow reveals
    prediction limits in online social activity." Nature human behaviour 3.2
    (2019): 122-128.
    """
    ego_len = len(W2)

    if ego_len <= e:
        raise ValueError('The length of ego sequence is too short')

    if (any(isinstance(i, list) for i in W1_list)) & (any(isinstance(i, list) for i in PTs_list)):
        wb = []
        alters_L = []
        alters_len = []

        if individual:
            # allocate a empty list to save all CCE
            CCE = []
            if ego_include:
                for W1, PTs in zip(W1_list, PTs_list):
                    alters_len.append(len(W1))
                    alters_L.append(LZ_cross_entropy(W1, W2, PTs, lambdas=True, e=e))
                    # count how many element > 1, which is wb
                    wb.append(sum(1 for x in alters_L[-1] if x > 1))

                    # add ego_len to alters_len and alters_L
                    ego_L = LZ_entropy(W2, lambdas=True, e=e)
                    wb_ego = wb + [sum(1 for x in ego_L if x > 1)]
                    alters_Lmax = np.amax(alters_L + [ego_L], axis=0)
                    sum_L = sum(alters_Lmax)
                    ave_length = np.average(alters_len + [ego_len], weights=wb_ego)
                    # append all CCE one by one
                    CCE.append((1.0 * ego_len / sum_L) * np.log2(ave_length))
            else:
                for W1, PTs in zip(W1_list, PTs_list):
                    alters_len.append(len(W1))
                    alters_L.append(LZ_cross_entropy(W1, W2, PTs, lambdas=True, e=e))
                    # count how many element > 1, which is wb
                    wb.append(sum(1 for x in alters_L[-1] if x > 1))

                    if sum(wb) != 0:
                        alters_Lmax = np.amax(alters_L, axis=0)
                        sum_L = sum(alters_Lmax)
                        ave_length = np.average(alters_len, weights=wb)
                        # append all CCE one by one
                        CCE.append((1.0 * ego_len / sum_L) * np.log2(ave_length))
                    else:
                        CCE.append(np.nan)
            return CCE
        else:
            for W1, PTs in zip(W1_list, PTs_list):
                alters_len.append(len(W1))
                alters_L.append(LZ_cross_entropy(W1, W2, PTs, lambdas=True, e=e))
                # count how many element > 1, which is wb
                wb.append(sum(1 for x in alters_L[-1] if x > 1))

            if wb != 0:
                alters_Lmax = np.amax(alters_L, axis=0)
                sum_L = sum(alters_Lmax)
                ave_length = np.average(alters_len, weights=wb)
                # compute cumulative cross entropy
                CCE = (1.0 * ego_len / sum_L) * np.log2(ave_length)
                return CCE
            else:
                return np.nan
    else:
        return LZ_cross_entropy(W1_list, W2, PTs_list, e=e)
