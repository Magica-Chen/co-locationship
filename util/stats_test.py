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
from scipy.stats import ttest_rel


def spearman_kendall_test(df, item, ranking='Rank', alpha=0.05, increasing=True):
    """
    Do spearman's and kendall's test for the increasing or decreasing trend.
    :param df: dataframe, it should include both column 'item' and column 'ranking'
    :param item: string, column of target's label
    :param ranking:string, column of rank's label
    :param alpha: significant level
    :param increasing: bool, test for increasing trend or decreasing trend
    :return: dataframe filled in all test results
    """
    category = sorted(list(set(df['category'].tolist())))
    dataset = sorted(list(set(df['dataset'].tolist())))

    test_result = []
    for ds in dataset:
        for cat in category:
            count_sm, count_kd = 0, 0

            df_temp = df[(df['dataset'] == ds) & (df['category'] == cat)]
            ur_ds = df_temp['userid'].unique().tolist()
            for user in ur_ds:
                rank = df_temp[df_temp['userid'] == user][ranking].tolist()

                item_specify = df_temp[df_temp['userid'] == user][item].tolist()

                coef_sm, p_sm = spearmanr(rank, item_specify)
                coef_kd, p_kd = kendalltau(rank, item_specify)
                if increasing:
                    if (coef_sm > 0) & (p_sm < alpha):
                        count_sm += 1

                    if (coef_kd > 0) & (p_kd < alpha):
                        count_kd += 1
                else:
                    if (coef_sm < 0) & (p_sm < alpha):
                        count_sm += 1

                    if (coef_kd < 0) & (p_kd < alpha):
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


def two_side_t_test(df, item, alpha=0.01, method='paired', difference=False):
    """
    Do two-side t test, including t-test and paired sample t-test.
    :param df: dataframe, it should include the column 'item'
    :param item: string, column of target's label
    :param alpha: significant level
    :param method: string, using 'paired' or not
    :param difference: bool, test for difference or for same
    :return: a nested list filled with dataframe of test results, and a list of datasets' names
    """
    category = sorted(list(set(df['category'].tolist())))
    dataset = sorted(list(set(df['dataset'].tolist())))
    n_cat = len(category)

    if method is 'paired':
        func = ttest_rel
    else:
        func = ttest_ind

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
                    stats, p = func(df_cat1, df_cat2)

                    if difference:
                        if (p < alpha) | (np.isnan(p)):
                            count += 1
                    else:
                        if (p > alpha) | (np.isnan(p)):
                            count += 1
                result.append(count / n_users)
        result = np.array(result).reshape(n_cat, n_cat)
        result = pd.DataFrame(result, columns=category, index=category)
        stats_list.append(result)

    return stats_list, dataset
