#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Class for co-locationship, meetup_strategy
# (c) Zexun Chen, 2020-12-09
# sxtpy2010@gmail.com

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import util

RANK_COLUMN = 'Rank'
CATEGORY_COLUMN = 'category'
DATASET_COLUMN = 'dataset'
USERID_X_COLUMN = 'userid_x'
USERID_Y_COLUMN = 'userid_y'


class ComparisonNetwork(object):
    """
    Create a class to compare some co-locationship network or social relationship network
    """

    def __init__(self, network_class, network_name, threshold=10):
        """
        :param network_class: class, or list of classes
        :param network_name: string, or list of strings for the corresponding classes
        :param threshold: threshold for the maximum x-axis to show
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

    def __call__(self):
        """
        Self call function just shows how many shared users in the comparison
        :return: None
        """
        print("There are " + str(len(self.userlist)) + " common users.")

    def plot_errorbar(self, target='CCP',
                      mode='talk', style="whitegrid", l=10, w=6, ci=95):
        """
        Errorbar plot: Rank vs target
        :param target: string, it should be 'CCP', 'ODLR, or 'CODLR'
        :param mode: seaborn setting
        :param style: seaborn setting
        :param l: length
        :param w: wide
        :param ci: seaborn setting, confidence interval
        :return: fig
        """

        if target is 'ODLR':
            y_axis = 'ODLR'
            y_label = '$\eta_{ego}(alter)$'
        elif target is 'CODLR':
            y_axis = 'CODLR'
            y_label = '$\eta_{ego}(alters)$'
        else:
            y_axis = 'Pi_alters_ratio'
            y_label = '$\Pi_{alters}/ \Pi_{ego}$'

        sns.set_context(mode)
        sns.set_style(style)
        fig, ax = plt.subplots(figsize=(l, w))
        sns.pointplot(x=RANK_COLUMN, y=y_axis, data=self.data,
                      hue=CATEGORY_COLUMN, ci=ci, join=False, ax=ax)
        ax.set_ylabel(y_label)
        ax.set_xlabel("Included Number of Alters")
        ax.legend_.set_title(None)
        return fig

    def plot_CE(self, mode='talk', style="whitegrid", l=10, w=6, n_bins=50):
        """
        Cross entropy histogram plot
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
        fig, ax = plt.subplots(figsize=(l, w))
        for cat in category:
            sns.distplot(self.data[self.data[CATEGORY_COLUMN] == cat]['CE_alter'], label=cat, bins=n_bins)
        ax.set(xlabel='Cross-entropy (bits)', ylabel='Density')
        ax.legend(loc='upper left')
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

            return fig
        else:
            return 'Must have two networks!'

    def stats_test_monotonicity(self, target='CCP', alpha=0.05, increasing=True):
        """
        Spearman's and Kendall's rank test for monotonicity
        :param target: string, it should be 'CCP', 'ODLR, or 'CODLR'
        :param alpha: float, significant level
        :param increasing: bool, test for increasing trend or decreasing trend
        :return: dataframe, filled in stats tests results
        """
        stats_test = util.spearman_kendall_test(self.data,
                                                item=target,
                                                rank_in=RANK_COLUMN,
                                                category_in=CATEGORY_COLUMN,
                                                dataset_in=DATASET_COLUMN,
                                                userid_in=USERID_X_COLUMN,
                                                alpha=alpha,
                                                increasing=increasing)
        return stats_test

    def stats_test_consistency(self, target, alpha=0.01,
                               mode='talk', l=5.4, w=1.8):
        """
        Do two-side t test, including t-test and paired sample t-test.
        :param df: dataframe, it should include the column 'item'
        :param target: string, it should be 'CCP', 'ODLR, or 'CODLR'
        :param alpha: significant level
        :param mode: seaborn setting
        :param l: length
        :param w: wide
        :return: matrix plot filled in test results
        """
        stats_list, dataset = util.two_side_t_test(self.data,
                                                   item=target,
                                                   alpha=alpha,
                                                   method='paired',
                                                   category_in=CATEGORY_COLUMN,
                                                   dataset_in=DATASET_COLUMN,
                                                   userid_in=USERID_X_COLUMN)
        n_subplots = len(self.category)
        sns.set_context(mode)
        fig, axn = plt.subplots(1, n_subplots, figsize=(l, w), sharey=True)

        for i, ax in enumerate(axn.flat):
            sns.heatmap(stats_list[i], ax=ax,
                        linewidths=.5, annot=True,
                        fmt=".2%", cbar=False, cmap="YlGnBu"
                        )
            ax.xaxis.set_label_position('bottom')
            ax.set_title(dataset[i], pad=15)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

        fig.tight_layout(rect=[0, 0, .9, 1])

        return fig
