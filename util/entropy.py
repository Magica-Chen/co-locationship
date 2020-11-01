#!/usr/bin/env python
# -*- coding: utf-8 -*-

# entropy_functions.py
# (c) Zexun Chen 2020-11-01
# sxtpy2010@gmail.com

import numpy as np
import collections


def shannon_entropy(seq):
    """Plain old Shannon entropy (in bits)."""
    C, n = collections.Counter(seq), float(len(seq))
    return -sum(c / n * np.log2(c / n) for c in list(C.values()))


# Since the LZ-entropy estimation only converge when the length is large,
# so we add one more arg for LZ-entropy function
def LZ_entropy(seq, lambdas=False, e=2):
    """Estimate the entropy rate of the symbols encoded in `seq`, a list of
    strings.

    Kontoyiannis, I., Algoet, P. H., Suhov, Y. M., & Wyner, A. J. (1998).
    Nonparametric entropy estimation for stationary processes and random
    fields, with applications to English text. IEEE Transactions on Information
    Theory, 44(3), 1319-1327.

    Bagrow, James P., Xipei Liu, and Lewis Mitchell. "Information flow reveals
    prediction limits in online social activity." Nature human behaviour 3.2
    (2019): 122-128.
    """
    N = len(seq)

    if N < e:
        return np.nan
    else:
        L = []
        for i, w in enumerate(seq):
            seen = True
            prevSeq = " %s " % " ".join(seq[0:i])
            c = i
            while seen and c < N:
                c += 1
                seen = (" %s " % " ".join(seq[i:c])) in prevSeq
            l = c - i
            L.append(l)

        if lambdas:
            return L
        return (1.0 * N / sum(L)) * np.log2(N)
