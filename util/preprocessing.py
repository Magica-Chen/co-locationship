#!/usr/bin/env python
# -*- coding: utf-8 -*-

# entropy_functions.py
# (c) Zexun Chen, 2020-11-12
# sxtpy2010@gmail.com

import pandas as pd


def geo2id(df, lat='lat', lon='lon', **kwargs):
    """ convert lat and lon to a geo-id
    :param df: dataframe or series with geo-coordinate information
    :param lon: longitude
    :param lat: latitude
    :return: a new dataframe with geo-id with name "placeid"
    """
    df_geo = df[['userid', 'datetime', lat, lon]]
    if 'resolution' in kwargs:
        # if resolution is given, use `round` function to get the specific resolution
        df_geo = df_geo.round({lat: kwargs['resolution'], lon: kwargs['resolution']})
    df_id = df_geo.groupby([lat, lon]).size().reset_index(name='count')[[lat, lon]]
    # Just give an id for each (lat, lon) pair
    df_id['placeid'] = df_id.index
    return df_geo.merge(df_id, how='left', on=[lat, lon])


def pre_processing(df_raw, min_records=150, filesave=False, **kwargs):
    """ pre-processing the given dataset
    :param df_raw: dataframe, raw dataset
    :param min_records: the min requirement of users' records, remove all invalid users' information.
    :param filesave: whether save the pre-processed results
    :return: pre-processed dataframe
    """

    if 'resolution' in kwargs:
        df_raw = geo2id(df_raw, **kwargs)[['userid',
                                           'placeid',
                                           'datetime']]

    # for weeplace dataset, '-' also means missing placeid
    # we can also list other special symbol, which means missing values
    if 'missing' in kwargs:
        symbol = kwargs['missing']
        if type(kwargs['missing']) is str:
            df_raw = df_raw[df_raw['placeid'] != symbol]
        if type(kwargs['missing']) is list:
            df_raw = df_raw[~df_raw['placeid'].isin(symbol)]

    df_no_missing = df_raw.dropna(subset=['userid',
                                          'placeid',
                                          'datetime'])[['userid',
                                                        'placeid',
                                                        'datetime'
                                                        ]]

    df_no_missing['placeid'] = df_no_missing['placeid'].astype(str)

    df = df_no_missing.groupby('userid')['datetime'].count().reset_index(name='count')
    user = list(set(df[df['count'] >= min_records]['userid'].tolist()))

    df_processed = df_no_missing[df_no_missing['userid'].isin(user)]

    if filesave:
        name = 'data/weeplace_checkins_' + str(min_records) + 'processed.csv'
        df_processed.to_csv(name, index=False)

    return df_processed
