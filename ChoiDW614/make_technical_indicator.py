import warnings
warnings.filterwarnings('ignore')
import glob
import os
import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from xgboost import plot_importance
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score
from sklearn import svm
import seaborn as sns
sns.set()


def moving_average(df, n):
    MA = pd.Series(df['CLOSE_SPY'].rolling(n, min_periods=n).mean(), name='MA_' + str(n))
    df = df.join(MA)
    return df


def volume_moving_average(df, n):
    df = pd.read_csv('./bin/data/ETFs_main.csv')
    VMA = pd.Series(df['VOLUME'].rolling(n, min_periods=n).mean(), name='VMA_' + str(n))
    df = df.join(VMA)
    return df


def relative_strength_index(df, n):
    """Calculate Relative Strength Index(RSI) for given data.

    :param df: pandas.DataFrame
    :param n:
    :return: pandas.DataFrame
    """
    i = 0
    UpI = [0]
    DoI = [0]
    while i + 1 <= df.index[-1]:
        UpMove = df.loc[i + 1, 'HIGH'] - df.loc[i, 'HIGH']
        DoMove = df.loc[i, 'LOW'] - df.loc[i + 1, 'LOW']
        if UpMove > DoMove and UpMove > 0:
            UpD = UpMove
        else:
            UpD = 0
        UpI.append(UpD)
        if DoMove > UpMove and DoMove > 0:
            DoD = DoMove
        else:
            DoD = 0
        DoI.append(DoD)
        i = i + 1
    UpI = pd.Series(UpI)
    DoI = pd.Series(DoI)
    PosDI = pd.Series(UpI.ewm(span=n, min_periods=n).mean())
    NegDI = pd.Series(DoI.ewm(span=n, min_periods=n).mean())

    RSI = pd.Series(PosDI / (PosDI + NegDI), name='RSI_'+ str(n))
    df = df.join(RSI)
    return df


def make_technical_indicator():
    df = pd.read_csv('./bin/data/ETFs_main.csv')
    df = moving_average(df, 45)
    df = volume_moving_average(df, 45)
    df = relative_strength_index(df, 14)
    df = df.set_index('Dates')
    df = df.dropna()

    df['target'] = df['CLOSE_SPY'].pct_change()
    df['target'] = np.where(df['target'] > 0, 1, 0)    # record whether it's up or down from the previous day
    df['target'].value_counts()
    df['target'] = df['target'].shift(-1)

    df = df.dropna()
    df['target'] = df['target'].astype(np.int64)
    y_var = df['target']
    x_var = df.drop(['target', 'OPEN', 'HIGH', 'LOW', 'VOLUME', 'CLOSE_SPY'], axis=1)

    up = df[df['target'] == 1].target.count()
    total = df.target.count()
    # print('up/down ratio: {0:.2f}'.format((up/total)))
    return df
