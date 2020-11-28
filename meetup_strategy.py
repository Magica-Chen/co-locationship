#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Class for co-locationship, meetup_strategy
# (c) Zexun Chen, 2020-11-12
# sxtpy2010@gmail.com

import numpy as np
import pandas as pd
import random
from datetime import timedelta
import util
import time

# from itertools import combinations
# from itertools import chain
# from collections import Counter

SEED = 2020  # set random seed for our random function
EPSILON = 2  # set the basic number to compute variables


class Co_Locationship(object):
    """
    Create a class to investigate co-locationship
    """

    def __init__(self, path, mins_records=150, **kwargs):
        """
        :param path: path of source file
        :param mins_records: the required min number of records for each user
        :param kwargs: resolution and other kargs
        """
        # rdata means raw dataset and pdata means processed dataset
        # since we only needs userid, placieid and datetime in our computation,
        # so these attributes are required.
        self.rdata = pd.read_csv(path)
        self.pdata = util.pre_processing(df_raw=self.rdata,
                                         min_records=mins_records,
                                         **kwargs)
        # all the following computations are based on processed data
        self.userlist = sorted(list(set(self.pdata['userid'].tolist())))
        self.freq = None
        if 'placeidT' in kwargs:
            self.placeidT = kwargs['placeidT']
        else:
            self.placeidT = None

        if 'network' in kwargs:
            self.network = kwargs['network']
        else:
            self.network = None

        if 'network_details' in kwargs:
            self.network_details = kwargs['network_details']
        else:
            self.network_details = None

    def __call__(self, **kwargs):
        """ Call itself, Extract the time-ordered placeid sequence
        """
        if self.placeidT is None:
            self.placeidT = {user: self.pdata[self.pdata['userid'] == user
                                              ].set_index('datetime').sort_index()[['placeid']]
                             for user in self.userlist}
        return self.placeidT

    def _find_co_locator(self, ego):
        """ Find all the co_locators for ego
        :param ego: string, ego's userid
        :return: dataframe, filled with co-locators information
        """

        df_ego = self.pdata[self.pdata['userid'] == ego][['userid',
                                                          'placeid',
                                                          'datetimeR']]

        df_alters = self.pdata[self.pdata['userid'] != ego][['userid',
                                                             'placeid',
                                                             'datetimeR']]
        # filter df_alters and improve the speed of merge
        df_alters = df_alters[df_alters['datetimeR'].isin(df_ego['datetimeR'])]
        df_alters = df_alters[df_alters['placeid'].isin(df_ego['placeid'])]

        """ Here meetup means two users appear in the same placeid at the same time, so we merge two 
        dataframes, keep on placeid and datatime, if they meet, it will be complete row record, 
        otherwise, the record should have NaN. Therefore, we remove all the records with NaN and we
        can have all the meetup information.
        """
        meetup = df_ego.merge(df_alters,
                              how='left',
                              on=['placeid',
                                  'datetimeR']).dropna()[['userid_x',
                                                          'placeid',
                                                          'datetimeR',
                                                          'userid_y']] \
            .drop_duplicates().groupby(['userid_x',
                                        'userid_y']).size() \
            .reset_index(name='count')
        return meetup

    def build_network(self, freq='H'):
        """ Build network by concating the meetups for all users
        :param freq: when comparing timestamp, which scale we use
        :return: merged dataframe with all the co_locators information
        """
        self.freq = freq
        self.pdata['datetimeR'] = pd.to_datetime(self.pdata['datetime']).dt.floor(freq)

        meetup_list = [self._find_co_locator(user) for user in self.userlist]
        user_meetup = pd.concat(meetup_list, sort=False)
        # 'meetup' as column name mean how many times ego and alter meetup.
        self.network = user_meetup.rename(columns={'count': 'meetup'})
        return self.network

    def _extract_info(self, user):
        """ Protect method: extract temporal-spatial information for each user
        Arg:
            user: string, a userid
        Return:
            user_time: datetime, user's timestamps
            N_uniq_placeid: int, the number user's unique visited placeids
            N_placeid: int, the number of user's visited placeids
            user_placeid: list, time-ordered visited placeid in a list
        """
        if self.placeidT is None:
            raise ValueError('Please generate placeidT first')

        user_temporal_placeid = self.placeidT[user]
        user_time = pd.to_datetime(user_temporal_placeid.index).tolist()
        user_placeid = user_temporal_placeid['placeid'].astype(str).values.tolist()
        N_uniq_placeid = len(set(user_placeid))
        N_placeid = len(user_placeid)

        return user_time, N_uniq_placeid, N_placeid, user_placeid

    def _calculate_pair(self, ego, alter):
        """
        :param ego: ego
        :param alter:  alter
        :return: N_previous, non_meetup, CE_alter, Pi_alter
        """
        ego_time, length_ego_uni, length_ego, ego_placeid = self._extract_info(ego)
        alter_time, length_alter_uni, length_alter, alter_placeid = self._extract_info(alter)

        """ Sort ego_time + alter_time and obtain the relative position PTs """
        total_time = sorted(ego_time + alter_time)
        PTs = [(total_time.index(x) - ego_time.index(x)) for x in ego_time]

        """ The largest number of PTs is the number of how many points of alter checked
        in before the last check-in of ego, which denoted as 'N_previous' 
        """
        N_previous = max(PTs)

        """ Use LZ_cross_entropy function to compute L """
        L, CE_alter = util.LZ_cross_entropy(alter_placeid, ego_placeid, PTs, both=True,
                                            lambdas=True, e=EPSILON)

        """ 'non_meetup' counts how many overlapped locations that are not co-locations """
        non_meetup = sum(L) - length_ego

        """  predictability """
        Pi_alter = util.getPredictability(length_ego_uni, CE_alter, e=EPSILON)

        return N_previous, non_meetup, CE_alter, Pi_alter

    def calculate_info(self):
        if self.network is None:
            raise ValueError('Please build network first')
        else:
            interim = self.network.copy()
            interim['N_previous'], interim['non_meetup'], interim['CE_alter'], interim['Pi_alter'] = zip(
                *[self._calculate_pair(x, y) for x, y in zip(interim['userid_x'], interim['userid_y'])]
            )
            self.network_details = interim

            return self.network_details

    def _quality_control(self):
        """
        Perform quality control
        """
        if self.network_details is None:
            raise ValueError('Please build network details first')
        else:
            # Remove ego-alter pairs which are not qualified
            qualified_network = self.network_details[(self.network_details['meetup'] > 1) &
                                                     (self.network_details['non_meetup'] > 0)]

            self.network_details = qualified_network

    def _contribution_control(self):
        """
        Perform contribution control
        """
        if self.network_details is None:
            raise ValueError('Please build network details first')
        else:
            # Remove alters which do not have contribution to ego
            # Alter performs better than random algorithm if and only if Pi > 0
            contributed_network = self.network_details[self.network_details['Pi_alter'] > 0]
            self.network_details = contributed_network

    def network_control(self, quality=True, contribution=True, num_alters=10,
                        **kwargs):
        """
        Apply some criteria to polish the network
        :param quality: bool, whether perform quality control
        :param contribution: bool, whetehr perform contribution control
        :param num_alters: number, the minimum number of alters for a valid ego
        :param **kwargs, Sorting task, you need to use keywords, 'by', 'ascending'
        Sort alters by the criteria, [method1, .... methodN], for all of the methods,
        we can choose one of them, 'meetup', 'N_previous', 'CE_alter', 'Pi_alter', 'non-meetup'
        :param **kwargs, Set a threshold for N_previous, keyword, 'N_previous'
        :return: polished network applying control criteria or sorting criteria
        """
        if self.network_details is None:
            raise ValueError('Please build network details first')
        else:
            if quality:
                self._quality_control()

            if contribution:
                self._contribution_control()

            # filter the network with minimum N_previous
            if 'N_previous' in kwargs:
                self.network_details = self.network_details[self.network_details['N_previous'] >= kwargs['N_previous']]

            # should apply fitler N_previous first if used, and then count the number of alters
            if num_alters:
                alters_count = self.network_details.groupby('userid_x')['userid_y'].count().reset_index(name='count')
                valid_egos = alters_count[alters_count['count'] >= num_alters]['userid_x'].tolist()
                self.network_details = self.network_details[self.network_details['userid_x'].isin(valid_egos)]

            # sort alters by given criteria first
            if ('by' in kwargs) & ('ascending' in kwargs):
                self.network_details = self.network_details.sort_values(by=kwargs['by'],
                                                                        ascending=kwargs['ascending'])

            return self.network_details

    def calculate_network(self, verbose=False, filesave=False):
        """
        Calculate the details of the given network, especially cumulative cross entropy
        :param verbose: bool, whether to show ego step by step
        :param filesave: bool, whether to save the final network with details
        :return: processed network with detailed information
        """
        if self.network_details is None:
            raise ValueError('Please build network details first')
        else:
            egolist = sorted(list(set(self.network_details['userid_x'].tolist())))

            CCE_alters, Pi_alters, CCE_ego_alters, Pi_ego_alters, LZ_entropy, Pi = zip(*[self._get_CCE_Pi(ego, verbose)
                                                                                         for ego in egolist])

            self.network_details = self.network_details.assign(CCE_alters=CCE_alters,
                                                               Pi_alters=Pi_alters,
                                                               CCE_ego_alters=CCE_ego_alters,
                                                               Pi_ego_alters=Pi_ego_alters,
                                                               LZ_entropy=LZ_entropy,
                                                               Pi=Pi)
            if filesave:
                name = self.freq + 'network_details.csv'
                self.network_details.to_csv(name)

            return self.network_details

    def _get_CCE_Pi(self, ego, verbose=False):
        """
        Get the CCE and Pi for ego
        :param ego: ego name
        :param verbose: bool, whether to show the ego in computation
        :return: CCEs and Pis
        """
        ego_time, length_ego_uni, length_ego, ego_placeid = self._extract_info(ego)
        alters = self.network_details[self.network_details['userid_x'] == ego]['userid_y'].tolist()
        alters_placeid_list, PTs_list = zip(*[self._get_placeid_PT(ego_time, alter) for alter in alters])
        alters_placeid_list, PTs_list = list(alters_placeid_list), list(PTs_list)
        
        CCE_alters = util.cumulative_LZ_CE(W1_list=alters_placeid_list,
                                           W2=ego_placeid,
                                           PTs_list=PTs_list,
                                           individual=True,
                                           e=EPSILON)
        Pi_alters = [util.getPredictability(N=length_ego_uni,
                                            S=x,
                                            e=EPSILON) for x in CCE_alters]
        # alters + ego
        alters_placeid_list.append(ego_placeid)
        PTs_list.append(list(range(len(ego_time))))
        CCE_ego_alters = util.cumulative_LZ_CE(W1_list=alters_placeid_list,
                                               W2=ego_placeid,
                                               PTs_list=PTs_list,
                                               individual=True,
                                               e=EPSILON)
        Pi_ego_alters = [util.getPredictability(N=length_ego_uni,
                                                S=x,
                                                e=EPSILON) for x in CCE_ego_alters]

        N_alters = len(alters)
        entropy = util.LZ_entropy(ego_placeid, e=EPSILON)
        Pi = util.getPredictability(length_ego_uni, entropy, e=EPSILON)

        if verbose:
            print(ego)

        return CCE_alters, Pi_alters, CCE_ego_alters, Pi_ego_alters, [entropy] * N_alters, [Pi] * N_alters

    def _get_placeid_PT(self, ego_time, alter):
        """
        Get placeid and PT for alter
        :param ego_time: ego time
        :param alter: alter
        :return: alter placeid sequence and PTs of alter based on ego
        """
        alter_time, _, _, alter_placeid = self._extract_info(alter)
        total_time = sorted(ego_time + alter_time)
        PTs = [(total_time.index(x) - ego_time.index(x)) for x in ego_time]
        return alter_placeid, PTs


class Social_Relationship(Co_Locationship):
    """
    Create a Social relationship network
    """

    def __init__(self, path, path_network, mins_records=150, **kwargs):
        super(Social_Relationship, self).__init__(path, mins_records, **kwargs)

        df_friend = pd.read_csv(path_network)
        self.network = df_friend[
            (df_friend[df_friend.columns[0]].isin(self.userlist)) & (
                df_friend[df_friend.columns[1]].isin(self.userlist))]
        self.network.columns = ['userid_x', 'userid_y']
