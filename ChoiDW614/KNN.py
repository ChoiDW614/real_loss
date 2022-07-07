# Data Manipulation
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from pandas_datareader import data as pdr
from src import yfinance_to_load_data
import yfinance as yf


def training():
    train_acc = []
    test_acc = []

    X_train, X_test, y_train, y_test, X, split, tmp_df = yfinance_to_load_data.get_data_from_yahoo_financeAPI()

    for n in range (1, 15):
        clf = KNeighborsClassifier(n_jobs=-1, n_neighbors=n)
        clf.fit(X_train, y_train)
        prediction = clf.predict(X_test)
        train_acc.append(clf.score(X_train, y_train))
        test_acc.append((prediction == y_test).mean())

    plt.figure(figsize=(12, 9))
    plt.plot(range(1, 15), train_acc, label='TRAIN set')
    plt.plot(range(1, 15), test_acc, label='TEST set')
    plt.xlabel("n_neighbors")
    plt.ylabel("accuracy")
    plt.xticks(np.arange(0, 16, step=1))
    plt.legend()
    plt.show()

    knn = KNeighborsClassifier(n_neighbors=7)
    knn.fit(X_train, y_train)

    accuracy_train = accuracy_score(y_train, knn.predict(X_train))
    accuracy_test = accuracy_score(y_test, knn.predict(X_test))

    print('train precision : %.2f' % accuracy_train)
    print('test precision : %.2f' % accuracy_test)

    tmp_df['Predicted_Signal'] = knn.predict(X)

    tmp_df['SPY_ret'] = np.log(tmp_df['Close'] / tmp_df['Close'].shift(1))
    cum_spy_ret = tmp_df[split:]['SPY_ret'].cumsum() * 100

    tmp_df['st_ret'] = tmp_df['SPY_ret'] * tmp_df['Predicted_Signal'].shift(1)
    cum_st_ret = tmp_df[split:]['st_ret'].cumsum() * 100

    plt.figure(figsize=(10, 5))
    plt.plot(cum_spy_ret, color='r', label='spy set')
    plt.plot(cum_st_ret, color='g', label='st set')
    plt.legend()
    plt.show()

    std = cum_st_ret.std()
    sharpe = (cum_st_ret - cum_spy_ret) / std
    sharpe = sharpe.mean()
    print('Sharpe ratio :  % .2f' % sharpe)
