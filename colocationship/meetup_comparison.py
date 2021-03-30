#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Class for co-locationship, meetup_strategy
# (c) Zexun Chen, 2020-12-09
# sxtpy2010@gmail.com

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from . import util

RANK_COLUMN = 'Rank'
CATEGORY_COLUMN = 'category'
DATASET_COLUMN = 'dataset'
USERID_X_COLUMN = 'userid_x'
USERID_Y_COLUMN = 'userid_y'


class ComparisonNetwork(object):
    """
    Create a class to compare some co-locationship network or social relationship network
    """

    def __init__(self, network_class, network_name, threshold=10, **kwargs):
        """
        :param network_class: class, or list of classes
        :param network_name: string, or list of strings for the corresponding classes
        :param threshold: threshold for the maximum x-axis to show
        :param **kwargs: 'common_users' if the common egos are specified.
        """
        if type(network_class) is list:
            df_network_list = []
            userlist_list = []
            for network, name in zip(network_class, network_name):
                interim = network.network_details
                interim[CATEGORY_COLUMN] = name
                userlist = network.final_userlist
                df_network_list.append(interim)
                userlist_list.append(userlist)
            df_network_all = pd.concat(df_network_list)
            # get common users for all classes
            common_users = sorted(list(set(userlist_list[0]).intersection(*userlist_list)))
            df_network_all = df_network_all[df_network_all[USERID_X_COLUMN].isin(common_users)]
            self.userlist = common_users
            self.category = network_name

        else:
            df_network_all = network_class.network_details
            df_network_all[CATEGORY_COLUMN] = network_name
            self.userlist = network_class.final_userlist
            self.category = [network_name]

        df_network_all = df_network_all[df_network_all[RANK_COLUMN] <= threshold]
        df_network_all = df_network_all.assign(Pi_alters_ratio=df_network_all['Pi_alters'] / df_network_all['Pi'],
                                               Pi_ego_alters_ratio=df_network_all['Pi_ego_alters'] / df_network_all[
                                                   'Pi']
                                               )
        self.data = df_network_all
        self.statistics = []

        if 'common_users' in kwargs:
            common_users = sorted(list(set(kwargs['common_users']) & set(self.userlist)))
            self.data = self.data[self.data[USERID_X_COLUMN].isin(common_users)]
            self.userlist = common_users

    def __call__(self):
        """
        Self call function just shows how many shared users in the comparison
        :return: None
        """
        print("There are " + str(len(self.userlist)) + " common users.")

    def plot_errorbar(self, target='alters',
                      mode='talk', style="whitegrid", l=10, w=6, ci=95):
        """
        Errorbar plot: Rank vs target
        :param target: string, it should be 'alters', 'ego+alters' ,'ODLR, or 'CODLR'
        :param mode: seaborn setting
        :param style: seaborn setting
        :param l: length
        :param w: wide
        :param ci: seaborn setting, confidence interval
        :return: fig
        """

        if target == 'ODLR':
            y_axis = 'ODLR'
            y_label = '$\eta_{ego}(alter)$'
        elif target == 'CODLR':
            y_axis = 'CODLR'
            y_label = '$\eta_{ego}(alters)$'
        elif target == 'RCCP alters':
            y_axis = 'Pi_alters_ratio'
            y_label = '$\Pi_{alters}/ \Pi_{ego}$'
        elif target == 'RCCP ego+alters':
            y_axis = 'Pi_ego_alters_ratio'
            y_label = '$\Pi_{ego+alters}/ \Pi_{ego}$'
        elif target == 'CCP alters':
            y_axis = 'Pi_alters'
            y_label = '$\Pi_{alters}$'
        elif target == "CCP ego+alters":
            y_axis = 'Pi_ego_alters'
            y_label = '$\Pi_{ego+alters}$'
        elif target == "CCE alters":
            y_axis = 'CCE_alters'
            y_label = '$\hat{S}_{ego|alters}$'
        else:
            raise ValueError("Please type correct target!")

        sns.set_context(mode)
        sns.set_style(style)
        fig, ax = plt.subplots(figsize=(l, w))
        sns.pointplot(x=RANK_COLUMN, y=y_axis, data=self.data,
                      hue=CATEGORY_COLUMN, ci=ci, join=False, ax=ax)
        ax.set_ylabel(y_label)
        ax.set_xlabel("Included Number of Alters")
        ax.legend_.set_title(None)

        if len(self.statistics) == 0:
            self.statistics = util.utils.ci_transfer(df=self.data,
                                                     on=['Rank', 'category'],
                                                     target=y_axis)
        else:
            mean_col = 'mean_' + y_axis
            if mean_col not in self.statistics.columns:
                statistics = util.utils.ci_transfer(df=self.data,
                                                    on=['Rank', 'category'],
                                                    target=y_axis)
                self.statistics = self.statistics.merge(statistics,
                                                        how='left',
                                                        on=['Rank', 'category'])

        return fig

    def plot_histogram(self, mode='talk', style="whitegrid", l=15, w=6, n_bins=50):
        """
        Cross entropy and cross predictability histogram plot
        :param mode: seaborn setting
        :param style: seaborn setting
        :param l: length
        :param w: wide
        :param n_bins: number of bins for histogram plot
        :return:
        """
        category = self.data[CATEGORY_COLUMN].unique().tolist()
        sns.set_context(mode)
        sns.set_style(style)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(l, w))
        for cat in category:
            sns.distplot(self.data[self.data[CATEGORY_COLUMN] == cat]['CE_alter'],
                         label=cat,
                         bins=n_bins,
                         ax=ax1)
        ax1.set(xlabel='Cross-entropy (bits)', ylabel='Density')
        ax1.legend(loc='upper left')

        for cat in category:
            sns.distplot(self.data[self.data[CATEGORY_COLUMN] == cat]['Pi_alter'],
                         label=cat,
                         bins=n_bins,
                         ax=ax2)
        ax2.set(xlabel='Predictability', ylabel='Density')
        ax2.legend(loc='upper left')

        return fig

    def plot_similarity(self, local=False,
                        mode='talk', style="whitegrid", l=10, w=6):
        """
        Plot matrix of Jaccard similarity between the comparison
        :param local: bool, local Jaccard similarity or Global similarity
        :param mode: seaborn setting
        :param style: seaborn setting
        :param l: length
        :param w: wide
        :return: None
        """
        n_cat = len(self.category)
        if n_cat > 1:
            sns.set_context(mode)
            sns.set_style(style)
            fig, ax = plt.subplots(figsize=(l, w))

            if local:
                # Compute local Jaccard similarity
                js_list = []
                for i in range(n_cat):
                    cat1 = self.category[i]
                    for j in range(i + 1, n_cat):
                        cat2 = self.category[j]
                        item_name = cat1 + ' vs ' + cat2
                        for user in self.userlist:
                            alter_cat1 = \
                                self.data[(self.data[CATEGORY_COLUMN] == cat1) & (self.data[USERID_X_COLUMN] == user)][
                                    USERID_Y_COLUMN].tolist()
                            alter_cat2 = \
                                self.data[(self.data[CATEGORY_COLUMN] == cat2) & (self.data[USERID_X_COLUMN] == user)][
                                    USERID_Y_COLUMN].tolist()
                            js = util.jaccard_similarity(alter_cat1, alter_cat2)
                            js_list.append([item_name, js])

                result = pd.DataFrame(js_list, columns=['item_name', 'js_value'])

                sns.boxplot(data=result, x='item_name', y='js_value', ax=ax)
                ax.set_xlabel('')
                ax.set_ylabel('Local Jaccard Similarity')
                if len(result['item_name'].unique()) > 1:
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

            else:
                # Compute global Jaccard similarity
                js_list = []
                for cat1 in self.category:
                    for cat2 in self.category:
                        alter_cat1 = self.data[self.data[CATEGORY_COLUMN] == cat1][USERID_Y_COLUMN].tolist()
                        alter_cat2 = self.data[self.data[CATEGORY_COLUMN] == cat2][USERID_Y_COLUMN].tolist()
                        js = util.jaccard_similarity(alter_cat1, alter_cat2)
                        js_list.append(js)
                result = np.array(js_list).reshape(n_cat, n_cat)
                result = pd.DataFrame(result, columns=self.category, index=self.category)

                sns.heatmap(result, cmap="YlGnBu", ax=ax,
                            linewidths=.5, annot=True)
                # ax.xaxis.tick_top() # x axis on top
                ax.xaxis.set_label_position('bottom')
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
                ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha='right')

            return fig
        else:
            return 'Must have two networks!'

    def stats_test_monotonicity(self, target='alters', alpha=0.05, increasing=True):
        """
        Spearman's and Kendall's rank test for monotonicity
        :param target: string, it should be 'alters', 'ego+alters' ,'ODLR, or 'CODLR'
        :param alpha: float, significant level
        :param increasing: bool, test for increasing trend or decreasing trend
        :return: dataframe, filled in stats tests results
        """
        df = self.data
        df[DATASET_COLUMN] = 'dataset'
        if target is 'ODLR':
            item = 'ODLR'
        elif target is 'CODLR':
            item = 'CODLR'
        elif target is 'alters':
            item = 'Pi_alters_ratio'
        else:
            item = 'Pi_ego_alters_ratio'

        stats_test = util.spearman_kendall_test(df,
                                                item=item,
                                                rank_in=RANK_COLUMN,
                                                category_in=CATEGORY_COLUMN,
                                                dataset_in=DATASET_COLUMN,
                                                userid_in=USERID_X_COLUMN,
                                                alpha=alpha,
                                                increasing=increasing)
        return stats_test

    def stats_test_consistency(self, target, alpha=0.01,
                               mode='talk', l=5.5, w=4.5):
        """
        Do two-side t test, including t-test and paired sample t-test.
        :param df: dataframe, it should include the column 'item'
        :param target: string, it should be 'alters', 'ego+alters', 'ODLR, or 'CODLR'
        :param alpha: significant level
        :param mode: seaborn setting
        :param l: length
        :param w: wide
        :return: matrix plot filled in test results
        """
        df = self.data
        df[DATASET_COLUMN] = 'dataset'
        if target is 'ODLR':
            item = 'ODLR'
        elif target is 'CODLR':
            item = 'CODLR'
        elif target is 'alters':
            item = 'Pi_alters_ratio'
        else:
            item = 'Pi_ego_alters_ratio'

        stats_list, dataset = util.two_side_t_test(df,
                                                   item=item,
                                                   alpha=alpha,
                                                   method='paired',
                                                   category_in=CATEGORY_COLUMN,
                                                   dataset_in=DATASET_COLUMN,
                                                   userid_in=USERID_X_COLUMN)
        n_subplots = len(dataset)
        sns.set_context(mode)

        fig, ax = plt.subplots(figsize=(l, w))
        sns.heatmap(stats_list[0], ax=ax,
                    linewidths=.5, annot=True,
                    fmt=".2%", cbar=False, cmap="YlGnBu"
                    )
        ax.xaxis.set_label_position('bottom')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha='right')

        fig.tight_layout(rect=[0, 0, .9, 1])

        return fig
