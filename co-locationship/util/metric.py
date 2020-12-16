#!/usr/bin/env python
# -*- coding: utf-8 -*-
# List some metric functions
# (c) Zexun Chen, 2020-11-12
# sxtpy2010@gmail.com


def jaccard_similarity(list1, list2):
    s1 = set(list1)
    s2 = set(list2)
    return len(s1.intersection(s2)) / len(s1.union(s2))

