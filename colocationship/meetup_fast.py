#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Class for co-locationship and social relationship strategy
# (c) Zexun Chen, 2021-12-13
# sxtpy2010@gmail.com

import pandas as pd
from . import util
import random as rd
from collections import Counter

EPSILON = 2  # set the basic number to compute variables


class Fast_network(object):
    """
    Create a class to compute the details of network fast
        Only can be used if you have already had the valid network
        and you would like to compute the statistics of network
    """

    def __init__(self, network, placeidT, traj_shuffle=False, knockoff=False):
        """
        :param network: the network is given
        :param placeidT: time-ordered placeid sequence of all users
        :param traj_shuffle: whether we need to shuffle the trajectories
        :param knockoff: whether we need to knock off the percentage of the whole trajectories
        """
        self.network = network
        self.placeidT = placeidT
        self.egolist = sorted(list(set(network['userid_x'].tolist())))
        self.shuffle = traj_shuffle
        self.knockoff = knockoff
        self.network_details = None

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

    def calculate_network(self, verbose=False, filesave=False, **kwargs):
        """
        Calculate the details of the given network, especially cumulative cross entropy
        :param verbose: bool, whether to show ego step by step
        :param filesave: bool, whether to save the final network with details
        :return: processed network with detailed information
        """

        CCE_alters, Pi_alters, CCE_ego_alter, Pi_ego_alter, CCE_ego_alters, Pi_ego_alters, LZ_entropy, Pi, \
        rank, ODLR, CODLR = zip(
            *[self._get_ego_info(ego, verbose)
              for ego in self.egolist]
        )
        # need to concat a tuple of list to a list
        self.network_details = self.network.assign(Rank=util.tuple_concat(rank),
                                                   CCE_alters=util.tuple_concat(CCE_alters),
                                                   Pi_alters=util.tuple_concat(Pi_alters),
                                                   CCE_ego_alter=util.tuple_concat(CCE_ego_alter),
                                                   Pi_ego_alter=util.tuple_concat(Pi_ego_alter),
                                                   CCE_ego_alters=util.tuple_concat(CCE_ego_alters),
                                                   Pi_ego_alters=util.tuple_concat(Pi_ego_alters),
                                                   LZ_entropy=util.tuple_concat(LZ_entropy),
                                                   Pi=util.tuple_concat(Pi),
                                                   ODLR=util.tuple_concat(ODLR),
                                                   CODLR=util.tuple_concat(CODLR)
                                                   )
        if filesave:
            self.result_save(**kwargs)

        return self.network_details

    def result_save(self, **kwargs):
        if 'name' in kwargs:
            name = kwargs['name'] + '_CLN_network_details_' + str(self.freq) + '.csv'
        else:
            name = 'CLN_network_details_' + str(self.freq) + '.csv'
        self.network_details.to_csv(name, index=False)

    def _get_ego_info(self, ego, verbose=False):
        """
        Get ego info, including, CE, CP, CE+ego, CP+ego, CCE, CCP, CCE+ego, CCP+egp, Rank, ODLR, CODLR
        :param ego: ego name
        :param verbose: bool, whether to show the ego in computation
        :return: CE, CP, CE+ego, CP+ego, CCE, CCP, CCE+ego, CCP+egp, Rank, ODLR, CODLR
        """
        ego_time, length_ego_uni, length_ego, ego_placeid = self._extract_info(ego)
        alters = self.network[self.network['userid_x'] == ego]['userid_y'].tolist()
        alters_placeid_tuple, PTs_tuple = zip(*[self._get_placeid_PT(ego_time, alter) for alter in alters])
        alters_placeid_list, PTs_list = list(alters_placeid_tuple), list(PTs_tuple)

        # compute ODLR and CODLR
        ego_Counter = Counter(ego_placeid)
        cumulative_common_elements = Counter()
        ODLR = []
        CODLR = []
        for alter_placeid in alters_placeid_list:
            alter_Counter = Counter(alter_placeid)
            common_elements = ego_Counter & alter_Counter
            ODLR.append(len(common_elements) / length_ego_uni)
            cumulative_common_elements += common_elements
            CODLR.append(len(cumulative_common_elements) / length_ego_uni)

        # alters
        CCE_alters = util.cumulative_LZ_CE(W1_list=alters_placeid_list,
                                           W2=ego_placeid,
                                           PTs_list=PTs_list,
                                           individual=True,
                                           e=EPSILON)
        Pi_alters = [util.getPredictability(N=length_ego_uni,
                                            S=x,
                                            e=EPSILON) for x in CCE_alters]

        # alter + ego (single alter + ego)
        CCE_ego_alter = [util.cumulative_LZ_CE(W1_list=[alter_placeid, ego_placeid],
                                               W2=ego_placeid,
                                               PTs_list=[PT, list(range(len(ego_time)))],
                                               individual=False,
                                               e=EPSILON) for alter_placeid, PT in zip(alters_placeid_list,
                                                                                       PTs_list)]
        Pi_ego_alter = [util.getPredictability(N=length_ego_uni,
                                               S=x,
                                               e=EPSILON) for x in CCE_ego_alter]

        # alters + ego
        CCE_ego_alters = util.cumulative_LZ_CE(W1_list=alters_placeid_list,
                                               W2=ego_placeid,
                                               PTs_list=PTs_list,
                                               individual=True,
                                               ego_include=True,
                                               e=EPSILON)
        Pi_ego_alters = [util.getPredictability(N=length_ego_uni,
                                                S=x,
                                                e=EPSILON) for x in CCE_ego_alters]

        N_alters = len(alters)
        entropy = util.LZ_entropy(ego_placeid, e=EPSILON)
        Pi = util.getPredictability(length_ego_uni, entropy, e=EPSILON)

        if verbose:
            print(ego)
        # last return value is rank
        return CCE_alters, Pi_alters, CCE_ego_alter, Pi_ego_alter, CCE_ego_alters, Pi_ego_alters, \
               [entropy] * N_alters, [Pi] * N_alters, list(range(1, N_alters + 1)), ODLR, CODLR

    def _get_placeid_PT(self, ego_time, alter):
        """
        Get placeid and PT for alter
        :param ego_time: ego time
        :param alter: alter
        :return: alter placeid sequence and PTs of alter based on ego
        """
        alter_time, _, _, alter_placeid = self._extract_info(alter)

        if self.knockoff:
            n = len(alter_placeid)
            m = int(n * self.knockoff) + 1
            # generate a random indices
            indices = rd.sample(range(n), k=n-m)
            # based on indices, choose the corresponding placeid and time
            alter_placeid = [alter_placeid[i] for i in indices]
            alter_time = [alter_time[i] for i in indices]

        total_time = sorted(ego_time + alter_time)
        PTs = [(total_time.index(x) - ego_time.index(x)) for x in ego_time]

        if self.shuffle:
            rd.shuffle(alter_placeid)

        return alter_placeid, PTs
