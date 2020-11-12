#!/usr/bin/env python
# -*- coding: utf-8 -*-
# List some metric functions
# (c) Zexun Chen, 2020-11-12
# sxtpy2010@gmail.com

from collections import Counter


def jaccard_similarity(list1, list2):
    s1 = set(list1)
    s2 = set(list2)
    return len(s1.intersection(s2)) / len(s1.union(s2))


def colocation_rate(ego, alters, placeidT):
    ego_seq = placeidT[ego]['placeid'].astype(str).values.tolist()
    ego_Counter = Counter(ego_seq)

    if type(alters) is list:
        common_elements = Counter()
        for alter in alters:
            alter_Counter = Counter(placeidT[alter]['placeid'].astype(str).values.tolist())
            temp_common_elements = ego_Counter & alter_Counter
            common_elements += temp_common_elements
    else:
        alter_Counter = Counter(placeidT[alters]['placeid'].astype(str).values.tolist())
        common_elements = ego_Counter & alter_Counter

    return [len(common_elements) / len(ego_Counter),
            sum(common_elements.values()) / len(ego_seq)]
