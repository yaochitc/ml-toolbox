import numpy as np
import matplotlib.pyplot as plt
from features import add_time_features


def time_trend(df, date_col, agg_col, agg):
    df.loc[:, [date_col, agg_col]]
    df = add_time_features(df)

    year_agg = df.groupby('year')[agg_col].agg([agg])
    month_agg = df.groupby('month')[agg_col].agg([agg])
    day_agg = df.groupby('day')[agg_col].agg([agg])
    weekday_agg = df.groupby('weekday')[agg_col].agg([agg])

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(20,7))
    ax1.scatter(year_agg.index.values, year_agg[agg])
    ax1.locator_params(nbins=2)
    ax1.ticklabel_format(axis='y', style='plain')
    ax1.set_xlabel('Year', fontsize=12)

    ax2.scatter(month_agg.index.values, month_agg[agg])
    ax2.locator_params(nbins=12)
    ax2.ticklabel_format(axis='y', style='plain')
    ax2.set_xlabel('Month', fontsize=12)

    ax3.scatter(day_agg.index.values, day_agg[agg])
    ax3.locator_params(nbins=10)
    ax3.ticklabel_format(axis='y', style='plain')
    ax3.set_xlabel('Day', fontsize=12)

    ax4.scatter(weekday_agg.index.values, weekday_agg[agg])
    ax4.locator_params(nbins=7)
    ax4.ticklabel_format(axis='y', style='plain')
    ax4.set_xlabel('Weekday', fontsize=12)

def feature_importance(feature_importance, train, size=20):
    feature_importance = feature_importance()[:size]
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5

    plt.figure(figsize=(12, 6))
    plt.barh(pos, feature_importance[sorted_idx], align='center')
    plt.yticks(pos, train.columns[sorted_idx])
    plt.xlabel('Relative Importance')
    plt.title('Variable Importance')