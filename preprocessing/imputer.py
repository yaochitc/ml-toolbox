class GroupImputer(object):
    def __init__(self, group, value, strategy="max"):
        self.group = group
        self.value = value
        self.strategy = strategy

    def fit(self, df):
        return df.groupby(self.group)[self.value].transform(lambda x: x.fillna(x.agg(self.strategy)))
