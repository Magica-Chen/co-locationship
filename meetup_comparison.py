#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Class for co-locationship, meetup_strategy
# (c) Zexun Chen, 2020-12-09
# sxtpy2010@gmail.com
import pandas as pd
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import util
import matplotlib.ticker as ticker


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
                interim['category'] = name
                userlist = network.final_userlist
                df_network_list.append(interim)
                userlist_list.append(userlist)
            df_network_all = pd.concat(df_network_list)
            # get common users for all classes
            common_users = sorted(list(set(userlist_list[0]).intersection(*userlist_list)))
            df_network_all = df_network_all[df_network_all['userid_x'].isin(common_users)]
            self.userlist = common_users
            self.category = network_name

        else:
            df_network_all = network_class.network_details
            df_network_all['category'] = network_name
            self.userlist = network_class.final_userlist
            self.category = [network_name]

        df_network_all = df_network_all[df_network_all['Rank'] <= threshold]
        df_network_all = df_network_all.assign(Pi_alters_ratio=df_network_all['Pi_alters'] / df_network_all['Pi'],
                                               Pi_ego_alters_ratio=df_network_all['Pi_ego_alters'] / df_network_all[
                                                   'Pi']
                                               )
        self.data = df_network_all

    def print_info(self):
        print("There are " + str(len(self.userlist)) + " common users.")

    def plot_errorbar(self, target='CCP',
                      mode='talk', style="whitegrid", l=10, w=6, ci=95):
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
        sns.pointplot(x="Rank", y=y_axis, data=self.data,
                      hue='category', ci=ci, join=False, ax=ax)
        ax.set_ylabel(y_label)
        ax.set_xlabel("Included Number of Alters")
        ax.legend_.set_title(None)

    def plot_CE(self, mode='talk', style="whitegrid", l=10, w=6, n_bins=50):
        category = self.data['category'].unique().tolist()
        sns.set_context(mode)
        sns.set_style(style)
        fig, ax = plt.subplots(figsize=(l, w))
        for cat in category:
            sns.distplot(self.data[self.data['category'] == cat]['CE_alter'], label=cat, bins=n_bins)
        ax.set(xlabel='Cross-entropy (bits)', ylabel='Density')
        ax.legend(loc='upper left')

    def plot_similarity(self, local=False,
                        mode='talk', style="whitegrid", l=10, w=6):
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
                    for j in range(i+1, n_cat):
                        cat2 = self.category[j]
                        item_name = cat1 + ' vs ' + cat2
                        for user in self.userlist:
                            alter_cat1 = self.data[(self.data['category'] == cat1) & (self.data['userid_x'] == user)][
                                'userid_y'].tolist()
                            alter_cat2 = self.data[(self.data['category'] == cat2) & (self.data['userid_x'] == user)][
                                'userid_y'].tolist()
                            js = util.metric.jaccard_similarity(alter_cat1, alter_cat2)
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
                        alter_cat1 = self.data[self.data['category'] == cat1]['userid_y'].tolist()
                        alter_cat2 = self.data[self.data['category'] == cat2]['userid_y'].tolist()
                        js = util.metric.jaccard_similarity(alter_cat1, alter_cat2)
                        js_list.append(js)
                result = np.array(js_list).reshape(n_cat, n_cat)
                result = pd.DataFrame(result, columns=self.category, index=self.category)

                sns.heatmap(result, cmap="YlGnBu", ax=ax,
                            linewidths=.5, annot=True)
                # ax.xaxis.tick_top() # x axis on top
                ax.xaxis.set_label_position('bottom')
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

        else:
            return 'Must have two networks!'
