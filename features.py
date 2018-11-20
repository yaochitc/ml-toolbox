import pandas as pd
import numpy as np
from datetime import timedelta


def add_time_features(dfs, date_col):
    for df in dfs:
        df['year'] = df[date_col].apply(lambda x: x.year)
        df['month'] = df[date_col].apply(lambda x: x.month)
        df['day'] = df[date_col].apply(lambda x: x.day)
        df['weekday'] = df[date_col].apply(lambda x: x.weekday())


def add_inverval_aggregated(dfs, agg_cols, agg_funcs=['mean'], group_cols=None,
                            pre_agg_cols=None, pre_agg=None, intervals=['year', 'month', 'day', 'weekday']):
    if not group_cols is None:
        group_str = '_'.join(group_cols) + '_'
    else:
        group_cols = []
        group_str = ''

    featured_dfs = []
    for df in dfs:
        for interval in intervals:
            groupby_cols = group_cols + [interval]
            if not pre_agg_cols is None and len(pre_agg_cols) > 0:
                agg_df = df.groupby(groupby_cols + pre_agg_cols).agg(pre_agg).reset_index()
            else:
                agg_df = df
            agg_df = agg_df.groupby(groupby_cols).agg({agg_col: agg_funcs for agg_col in agg_cols})
            agg_df.columns = [f"{group_str}{x[1]}_{x[0]}_per_{interval}" for x in agg_df.columns.ravel()]
            agg_df = agg_df.reset_index()
            featured_dfs.append(pd.merge(df, agg_df, on=groupby_cols, how='left'))
    return featured_dfs


def get_timespan(df, dt, minus, periods, freq='D'):
    return df[pd.date_range(dt - timedelta(days=minus), periods=periods, freq=freq)]

def get_day_aggregated(df, daypoint):
    X = {}
    for i in [3, 7, 14, 30, 60, 140]:
        tmp = get_timespan(df, daypoint, i, i)
        X['diff_%s_mean' % i] = tmp.diff(axis=1).mean(axis=1).values
        X['mean_%s_decay' % i] = (tmp * np.power(0.9, np.arange(i)[::-1])).sum(axis=1).values
        X['mean_%s' % i] = tmp.mean(axis=1).values
        X['median_%s' % i] = tmp.median(axis=1).values
        X['min_%s' % i] = tmp.min(axis=1).values
        X['max_%s' % i] = tmp.max(axis=1).values
        X['std_%s' % i] = tmp.std(axis=1).values
    return pd.DataFrame(X)