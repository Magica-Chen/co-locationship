#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Class for co-locationship, meetup_strategy
# (c) Zexun Chen, 2020-12-09
# sxtpy2010@gmail.com

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from scipy.stats import kendalltau
from scipy.stats import ttest_ind


def spearman_kendall_test(df, item):
    category = sorted(list(set(df['category'].tolist())))
    dataset = sorted(list(set(df['dataset'].tolist())))

    test_result = []
    for ds in dataset:
        for cat in category:
            count_sm, count_kd = 0, 0

            df_temp = df[(df['dataset'] == ds) & (df['category'] == cat)]
            ur_ds = df_temp['userid'].unique().tolist()
            for user in ur_ds:
                rank = df_temp[df_temp['userid'] == user]['Rank'].tolist()

                item_specify = df_temp[df_temp['userid'] == user][item].tolist()

                coef_sm, p_sm = spearmanr(rank, item_specify)
                coef_kd, p_kd = kendalltau(rank, item_specify)

                if (coef_sm > 0) & (p_sm < 0.05):
                    count_sm += 1

                if (coef_kd > 0) & (p_kd < 0.05):
                    count_kd += 1

            test_result.append([ds, cat,
                                count_sm, count_sm / len(ur_ds),
                                count_kd, count_kd / len(ur_ds),
                                len(ur_ds)]
                               )

    stats_test = pd.DataFrame(test_result, columns=['dataset',
                                                    'category',
                                                    'SpN', 'SpP', 'Kn', 'Kp',
                                                    'total']
                              ).sort_values(['dataset', 'category'])

    return stats_test


def two_side_t_test(df, item):
    category = sorted(list(set(df['category'].tolist())))
    dataset = sorted(list(set(df['dataset'].tolist())))
    n_cat = len(category)

    stats_list = []
    for ds in dataset:
        df_temp = df[df['dataset'] == ds]
        ur_ds = df_temp['userid'].unique().tolist()
        n_users = len(ur_ds)

        result = []
        for cat1 in category:
            for cat2 in category:
                count = 0
                for user in ur_ds:
                    df_cat1 = df_temp[(df_temp['category'] == cat1) & (df_temp['userid'] == user)][item]
                    df_cat2 = df_temp[(df_temp['category'] == cat2) & (df_temp['userid'] == user)][item]
                    stats, p = ttest_ind(df_cat1, df_cat2)

                    if p > 0.05:
                        count += 1
                result.append(count / n_users)
        result = np.array(result).reshape(n_cat, n_cat)
        result = pd.DataFrame(result, columns=category, index=category)
        stats_list.append(result)

    return stats_list
