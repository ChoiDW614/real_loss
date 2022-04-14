# test.py
import datetime
import numpy as np
import pandas as pd
import FinanceDataReader as fdr
import matplotlib.pyplot as plt

# mod datetime      4/7
# learn datetime in datetime


# def testfun():
#     format = '%Y-%m-%d %H:%M:%S'
#     datetime_str = '2018-05-13 12:34:56'
#     datetime_dt = datetime.datetime.strptime(datetime_str, format)
#     print(type(datetime_dt))
#     print(datetime_dt)
#
#     datetime_str2 = datetime_dt.strftime('%Y-%m-%d %H:%M:%S')
#     print(type(datetime_str2))
#     print(datetime_str2)
#
#     datetime_str3 = datetime_dt.strftime('%Y-%m-%d %H')
#     print(type(datetime_str3))
#     print(datetime_str3)
#
#     datetime_str4 = datetime_dt.strftime('%Y-%m')
#     print(type(datetime_str4))
#     print(datetime_str4)


# mod numpy
# learn datetime64 in numpy


# def testfun():
#     print(type(np.datetime64('2019-01-01')))
#     print(np.datetime64('2019-01-01'))
#     print()
#
#     print(np.datetime64(1000, 'ns'))
#     print(np.datetime64(10000, 'D'))
#     print(np.datetime64(1000000000, 's'))
#     print()
#
#     print(type(np.array(['2007-07-13', '2006-01-13', '2010-08-13'], dtype='datetime64')))
#     print(np.array(['2007-07-13', '2006-01-13', '2010-08-13'], dtype='datetime64'))
#     print()
#
#     print(type(np.arange('2005-02', '2005-03', dtype='datetime64[D]')))
#     print(np.arange('2005-02', '2005-03', dtype='datetime64[D]'))
#     print()
#
#     print(type(np.arange('2005-02', '2006-03', dtype='datetime64[M]')))
#     print(np.arange('2005-02', '2006-03', dtype='datetime64[M]'))
#     print()
#
#     print(np.datetime64('2009-01-01') - np.datetime64('2008-01-01'))
#     print(np.datetime64('2009') - np.datetime64('2008-01'))
#     print(np.datetime64('2009-01-01') - np.datetime64('2008-01'))
#     print()


# mod pandas
# learn pandas.read in pandas


# def testfun():
#     df = pd.read_csv('./bin/data/SPY.csv')
#     print(type(df.head()))
#     print(df.head())
#
#     aapl_df = pd.read_csv('./bin/data/SPY.csv', index_col='Date', parse_dates=['Date'])
#     print(type(aapl_df.head()))
#     print(aapl_df.head())
#     print(type(aapl_df.index))
#     print(type(aapl_df.index[0]))


# mod pandas
# learn to handle missing and outliers

# def testfun():
#     s1 = pd.Series([1, np.nan, 3, 4, 5])
#     s2 = pd.Series([1, 2, np.nan, 4, 5])
#     s3 = pd.Series([1, 2, 3, np.nan, 5])
#     df = pd.DataFrame({'S1': s1,
#                        'S2': s2,
#                        'S3': s3})
#     print(df.head())
#     print()
#
#     print(df['S1'].isna())
#     print()
#
#     print(df.isna())
#     print()
#
#     print(df.isna().sum())
#     print()
#
#     print(df.isin([np.nan]))
#     print()
#
#     print(df.isin([np.nan]).sum())
#     print()
#
#     print(df.isnull())  # same as isna
#     print()
#
#     print(df.isnull().sum())
#     print()
#
#     df = df.fillna(method='pad')
#     print(df.head())
#
#     df = df.fillna(method='bfill')
#     print(df.head())
#
#     df = df.dropna()
#     df.dropna(axis='rows')
#     df.dropna(axis=0)
#     print(df.head())
#
#     df = df.dropna(axis='columns')
#     df = df.dropna(axis=1)
#     print(df.head())
#
#     aapl_df = pd.read_csv('./bin/data/AAPL.csv')
#     # boolean type indexing
#     print(aapl_df[aapl_df.isin([np.nan, np.inf, -np.inf]).any(axis=1)])


############################################################################################

# mod pandas    4/10
# indexing and slicing

# def testfun():
#     pd.set_option('display.max_columns', 9)
#     aapl_df = pd.read_csv('./bin/data/AAPL.csv', index_col='Date', parse_dates=['Date'])
#     print(aapl_df['Open'].head())
#     print(aapl_df[['Open', 'High', 'Low', 'Close']].head())
#     print(aapl_df[5:8])
#     print(aapl_df['1980-12-12':'1980-12-19'])
#
#     df = pd.read_csv('./bin/data/AAPL.csv')
#     print(df.head())
#     print(type(df.index))
#     print(type(df.index[0]))
#
#     df = pd.read_csv('./bin/data/AAPL.csv', index_col='Date', parse_dates=['Date'])
#     print(df.head())
#     print(type(df.index))
#     print(type(df.index[0]))
#     print(df.loc['1980-12-12'])
#     print(type(df.loc['1980-12-12']))
#     print(df.iloc[0])
#     print(df.loc['2018-10-10':'2018-10-20', 'Open':'Close'])
#     print(df.iloc[8000:8010, [0, 1, 2, 3]])
#     print(df.loc['2016-11'].head())
#     print(df.loc['2016-Nov-1':'2016-Nov-10'].head())
#     print(df.loc['November 1, 2016':'November 10, 2016'].head())
#
#     aapl_df = pd.read_csv('./bin/data/AAPL.csv', index_col='Date', parse_dates=['Date'])
#     aapl_df['Close_lag1'] = aapl_df['Close'].shift()
#     print(aapl_df.head())
#     aapl_df['pct_change'] = aapl_df['Close'].pct_change()
#     print(aapl_df.head())
#     aapl_df['Close_diff'] = aapl_df['Close'].diff()
#     print(aapl_df.head())
#     aapl_df['MA'] = aapl_df['Close'].rolling(window=5).mean()
#     print(type(aapl_df['Close'].rolling(window=5)))
#     print(aapl_df.head(10))
#     index = pd.date_range(start='2019-01-01', end='2019-10-01', freq='B')
#     series = pd.Series(range(len(index)), index=index)
#     print(series)
#     print(series.resample(rule='M').sum())
#     print(series.resample(rule='M').last())
#     print(series.resample(rule='MS').first())


# mod FinanceDataReader
# learn to use FinanceDataReader


# def testfun():
#     pd.set_option('display.max_columns', 11)
#     df_krx = fdr.StockListing("KRX")
#     print(len(df_krx))
#     print(df_krx.head())
#
#     df_spx = fdr.StockListing("SP500")
#     print(df_spx.head())
#     print(len(df_spx))
#     print(df_spx.head())
#
#     df = fdr.DataReader('001250', '2018')
#     print(df.head(10))
#     df['Close'].plot()
#     plt.show()
#
#     df = fdr.DataReader('AAPL', '2010')
#     print(df.head(10))
#     df['Close'].plot()
#     plt.show()
#
#     exchange_map = {
#         'KRX': 'Seoul', '한국 거래소': 'Seoul',
#         'NASDAQ': 'NASDAQ', '나스닥': 'NASDAQ',
#         'NYSE': 'NYSE', '뉴욕증권거래소': 'NYSE',
#         'AMEX': 'AMEX', '미국증권거래소': 'AMEX',
#         'Shanghai': 'Shanghai', '상해': 'Shanghai', '상하이': 'Shanghai',
#         'Shenzhen': 'Shenzhen', '심천': 'Shenzhen',
#         'Hong Kong': 'Hong Kong', '홍콩': 'Hong Kong',
#         'Tokyo': 'Tokyo', '도쿄': 'Tokyo',
#     }
#
#     jp_df1 = fdr.DataReader(symbol='7751', start='2019-01-01', exchange='TSE')
#     jp_df2 = fdr.DataReader(symbol='7751', start='2019-01-01', exchange='도쿄')
#     jp_df2['Close'].plot()
#     plt.show()
#
#     ch_df1 = fdr.DataReader(symbol='601186', start='2019-01-01', exchange='SSE')
#     ch_df2 = fdr.DataReader(symbol='601186', start='2019-01-01', exchange='상해')
#     ch_df1['Close'].plot()
#     plt.show()
