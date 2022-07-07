import numpy as np
import pandas as pd

from pandas_datareader import data as pdr
import yfinance as yf


def get_data_from_yahoo_financeAPI():
    df = pdr.get_data_yahoo('SPY', '2012-01-01', '2017-01-01')
    df = df.dropna()

    tmp_df = df[['Open', 'High', 'Low', 'Close']].copy()
    tmp_df['Open-Close'] = tmp_df['Open'] - tmp_df['Close']
    tmp_df['High-Low'] = tmp_df['High'] - tmp_df['Low']
    tmp_df = tmp_df.dropna()
    X = tmp_df[['Open-Close', 'High-Low']]
    Y = np.where(tmp_df['Close'].shift(-1) > tmp_df['Open'].shift(-1), 1, 0)

    split_percentage = 0.7
    split = int(split_percentage * len(tmp_df))
    X_train = X[:split]
    Y_train = Y[:split]

    X_test = X[split:]
    Y_test = Y[split:]

    return X_train, X_test, Y_train, Y_test, X, split, tmp_df
