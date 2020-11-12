#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Get predictability function
# (c) Zexun Chen 2020-11-01
# sxtpy2010@gmail.com

import numpy as np
import mpmath


def getPredictability(N, S, e=100):
    if (N >= e) & np.isfinite(S) & (S < np.log2(N + 1e-10)):
        f = lambda x: (((1 - x) / (N - 1)) ** (1 - x)) * x ** x - 2 ** (-S)
        root = mpmath.findroot(f, 1)
        return float(root.real)
    else:
        return np.nan
