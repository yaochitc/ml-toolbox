from functools import reduce
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def drop_const(dfs):
    def intersect(l1, l2):
        return [value for value in l1 if value in l2]

    def find_const_cols():
        for df in dfs:
            df_nunique = df.nunique()
            unique_cols = df_nunique[df_nunique <= 1].index.values
            df_na = df.isna().sum()
            nona_cols = df_na[df_na == 0].index.values
            yield intersect(unique_cols, nona_cols)

    const_cols = reduce(intersect, find_const_cols())

    for df in dfs:
        df.drop(columns=const_cols, inplace=True)

def set_type(dfs, cols, type):
    for col in cols:
        for df in dfs:
            df[col] = df[col].astype(type)

def label_encode(dfs, cols):
    for col in cols:
        encoder = LabelEncoder()
        for df in dfs:
            df[col] = encoder.fit_transform(df[col].values.astype('str'))

def onehot_encode(train_df, test_df, cols):
    train_df = pd.get_dummies(train_df, columns=cols)
    test_df = pd.get_dummies(test_df, columns=cols)

    return train_df.align(test_df, join='outer', axis=1)

