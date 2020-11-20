#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Class for co-locationship, meetup_strategy
# (c) Zexun Chen, 2020-11-12
# sxtpy2010@gmail.com

import pandas as pd
import numpy as np
import random
from datetime import timedelta
import util

# from itertools import combinations
# from itertools import chain
# from collections import Counter

SEED = 2020  # set random seed for our random function
EPSILON = 2  # set the basic number to compute variables


class Co_Locationship(object):
    """
    Create a class to investigate co-locationship
    """

    def __init__(self, path, mins_records=150, freq='H', **kwargs):
        """
        :param path: path of source file
        :param mins_records: the required min number of records for each user
        :param freq: when comparing timestamp, which scale we use
        :param kwargs: resolution and other kargs
        """
        # rdata means raw dataset and pdata means processed dataset
        # since we only needs userid, placieid and datetime in our computation,
        # so these attributes are required.
        self.rdata = pd.read_csv(path)
        self.pdata = util.pre_processing(df_raw=self.rdata,
                                         min_records=mins_records,
                                         **kwargs)
        self.pdata['datetimeR'] = pd.to_datetime(self.pdata['datetime']).dt.floor(freq)
        # all the following computations are based on processed data
        self.userlist = sorted(list(set(self.pdata['userid'].tolist())))
        self.freq = freq
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
                                                             'datetimeH']]
        # filter df_alters and improve the speed of merge
        df_alters = df_alters[df_alters['datetimeH'].isin(df_ego['datetimeH'])]
        df_alters = df_alters[df_alters['placeid'].isin(df_ego['placeid'])]

        """ Here meetup means two users appear in the same placeid at the same time, so we merge two 
        dataframes, keep on placeid and datatime, if they meet, it will be complete row record, 
        otherwise, the record should have NaN. Therefore, we remove all the records with NaN and we
        can have all the meetup information.
        """
        meetup = df_ego.merge(df_alters,
                              how='left',
                              on=['placeid',
                                  'datetimeH']).dropna()[['userid_x',
                                                          'placeid',
                                                          'datetimeH',
                                                          'userid_y']] \
            .drop_duplicates().groupby(['userid_x',
                                        'userid_y']).size() \
            .reset_index(name='count')
        return meetup

    def build_network(self):
        """ Build network by concating the meetups for all users
        :return: merged dataframe with all the co_locators information
        """
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
        L, CE_alter = util.LZ_cross_entropy(alter_placeid, ego_placeid, PTs,
                                            lambdas=True, e=EPSILON)

        """ 'non_meetup' counts how many overlapped locations that are not co-locations """
        non_meetup = sum(L) - length_ego

        """  predictability """
        Pi_alter = util.getPredictability(length_ego_uni, CE_alter, e=EPSILON)

        return N_previous, non_meetup, CE_alter, Pi_alter

    def calculate_details(self):
        if self.network is None:
            raise ValueError('Please build network first')
        else:
            N_previous, non_meetup, CE_alter, Pi_alter = zip(*self.network.apply(lambda row:
                                                                                 self._calculate_pair(row.userid_x,
                                                                                                      row.userid_y))
                                                             )
            self.network_details = self.network.assign(N_previous=N_previous,
                                                       non_meetup=non_meetup,
                                                       CE_alter=CE_alter,
                                                       Pi_alter=Pi_alter)
            return self.network_details

    def quality_control(self):
        """
        Perform quality control
        :return: qualified_network
        """
        if self.network_details is None:
            raise ValueError('Please build network details first')
        else:
            # Remove ego-alter pairs which are not qualified
            qualified_network = self.network_details[(self.network_details['meetup'] > 1) &
                                                     (self.network_details['non_meetup'] > 0)]

            self.network_details = qualified_network

            return qualified_network

    def contribution_control(self):
        """
        Perform contribution control
        :return: contributed_network
        """
        if self.network_details is None:
            raise ValueError('Please build network details first')
        else:
            # Remove alters which do not have contribution to ego
            # Alter performs better than random algorithm if and only if Pi > 0
            contributed_network = self.network_details[self.network_details['Pi_alter'] > 0]
            self.network_details = contributed_network

            return contributed_network


