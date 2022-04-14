import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def read_data():
    # load data
    df = pd.read_csv('./bin/data/AMZN.csv', index_col='Date', parse_dates=['Date'])
    # print(df.head())
    # print(df[df.isin([np.nan, np.inf, -np.inf]).any(1)])

    price_df = df.loc[:, ['Adj Close']].copy()
    # price_df.plot(figsize=(16, 9))
    # plt.show()
    #
    # from_date = '1997-01-03'
    # to_date = '2003-01-03'
    # price_df.loc[from_data:to_data].plot(figsize=(16, 9))
    # plt.show()

    # check that the data have any nan
    price_df['daily_rtn'] = price_df['Adj Close'].pct_change()
    # print(price_df.head(10))
    price_df['st_rtn'] = (1+price_df['daily_rtn']).cumprod()
    # print(price_df.head(10))
    # price_df['st_rtn'].plot(figsize=(16, 9))
    # plt.show()

    # data slicing
    base_date = '2011-01-03'
    tmp_df = price_df.loc[base_date:, ['st_rtn']] / price_df.loc[base_date, ['st_rtn']]
    last_date = tmp_df.index[-1]
    # print('누적 수익 : ', tmp_df.loc[last_date, 'st_rtn'])
    # tmp_df.plot(figsize=(16, 9))
    # plt.show()

    # 연평균 복리 수익률 (CAGR)
    CAGR = price_df.loc['2022-04-13', 'st_rtn'] ** (252./len(price_df.index)) - 1

    # 최대 낙폭 (MDD)
    historical_max = price_df['Adj Close'].cummax()
    daily_drawdown = price_df['Adj Close'] / historical_max - 1.0
    historical_dd = daily_drawdown.cummin()
    MDD = historical_dd.min()
    # historical_dd.plot(figsize=(16, 9))
    # plt.show()

    # 변동성 (Vol)
    VOL = np.std(price_df['daily_rtn']) * np.sqrt(252.)
    # print(VOL)

    # 샤프 지수 (ex-post Sharpe ratio)
    Sharpe = np.mean(price_df['daily_rtn']) / np.std(price_df['daily_rtn']) * np.sqrt(252.)
    # print(Sharpe)

    # Print all data
    print('CAGR : ', round(CAGR*100, 2), '%')
    print('Sharpe : ', round(Sharpe, 2))
    print('VOL : ', round(VOL*100, 2), '%')
    print('MDD : ', round(-1*MDD*100, 2), '%')
