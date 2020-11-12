#!/usr/bin/env python
# -*- coding: utf-8 -*-
# List some util functions
# (c) Zexun Chen, 2020-11-12
# sxtpy2010@gmail.com


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
