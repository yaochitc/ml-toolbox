import os
import pandas as pd

PREFIX_TRAIN = 'train'
PREFIX_VALID = 'val'
PREFIX_TEST = 'test'

def _save_csv(X, y, prefix, basepath):
    X_path = os.path.join(basepath, f'{prefix}_X.csv')
    y_path = os.path.join(basepath, f'{prefix}_y.csv')
    X.to_csv(X_path, index=False)
    if not y is None:
        pd.DataFrame(y).to_csv(y_path, index=False, header=None)

def _load_csv(prefix, basepath):
    X_path = os.path.join(basepath, f'{prefix}_X.csv')
    y_path = os.path.join(basepath, f'{prefix}_y.csv')
    X = pd.read_csv(X_path)
    if os.path.exists(y_path):
        y = pd.read_csv(y_path, header=None)[0].values
        return X, y
    else:
        return X

def save_train(X, y, basepath='data'):
    _save_csv(X, y, PREFIX_TRAIN, basepath)

def save_valid(X, y, basepath='data'):
    _save_csv(X, y, PREFIX_VALID, basepath)

def save_test(X, basepath='data'):
    _save_csv(X, None, PREFIX_TEST, basepath)

def load_train(basepath='data'):
    return _load_csv(PREFIX_TRAIN, basepath)

def load_valid(basepath='data'):
    return _load_csv(PREFIX_VALID, basepath)

def load_test(basepath='data'):
    return _load_csv(PREFIX_TEST, basepath)