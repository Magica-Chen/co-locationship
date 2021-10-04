#!/usr/bin/env python
# -*- coding: utf-8 -*-
# List some util functions
# (c) Zexun Chen, 2020-11-12
# sxtpy2010@gmail.com

import functools
import operator
import pickle
import seaborn as sns


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


def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def read_object(filename):
    with open(filename, 'rb') as obj:  # Overwrites any existing file.
        return pickle.load(obj)


def ci_transfer(df, on, target):
    lower_col = 'lower_' + target
    upper_col = 'upper_' + target
    mean_col = 'mean_' + target

    f = df.groupby(on).count().reset_index()[on]

    f[lower_col], f[upper_col] = zip(*df.groupby(on)[target].apply(lambda x:
                                                                   sns.utils.ci(sns.algorithms.bootstrap(x.dropna()),
                                                                                which=95)))
    f[mean_col] = df.groupby(on)[target].mean().reset_index()[target]
    return f

