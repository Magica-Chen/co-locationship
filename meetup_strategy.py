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
# from collections import OrderedDict

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
        # all the following computations are based on processed data
        self.userlist = sorted(list(set(self.pdata['userid'].tolist())))
        self.freq = freq
        self.placeidT = None

    def __call__(self, **kwargs):
        """ Call itself, Extract the time-ordered placeid sequence
        """
        if 'placeidT' in kwargs:
            self.placeidT = kwargs['placeidT']
        else:
            self.placeidT = {user: self.pdata[self.pdata['userid'] == user
                                              ].set_index('datetime').sort_index()[['placeid']]
                             for user in self.userlist}

