#!/usr/bin/env python
# -*- coding: utf-8 -*-
# List some util functions
# (c) Zexun Chen, 2020-11-12
# sxtpy2010@gmail.com

import functools
import operator


def fast_indices(lst, element):
    # fast approach to find all indices of the given element
    result = []
    offset = -1
    while True:
        try:
            offset = lst.index(element, offset + 1)
        except ValueError:
            return result
        result.append(offset)


def tuple_concat(x):
    # concat a tuple of list to a list
    return functools.reduce(operator.iconcat, x, [])

