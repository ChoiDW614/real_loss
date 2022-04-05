# test.py
import datetime
import numpy as np
import pandas as pd

# mod datetime
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

def testfun():
    # s1 = pd.Series([1, np.nan, 3, 4, 5])
    # s2 = pd.Series([1, 2, np.nan, 4, 5])
    # s3 = pd.Series([1, 2, 3, np.nan, 5])
    # df = pd.DataFrame({'S1': s1,
    #                    'S2': s2,
    #                    'S3': s3})
    # print(df.head())
    # print()
    #
    # print(df['S1'].isna())
    # print()
    #
    # print(df.isna())
    # print()
    #
    # print(df.isna().sum())
    # print()
    #
    # print(df.isin([np.nan]))
    # print()
    #
    # print(df.isin([np.nan]).sum())
    # print()
    #
    # print(df.isnull())  # same as isna
    # print()
    #
    # print(df.isnull().sum())
    # print()
    #
    # df = df.fillna(method='pad')
    # print(df.head())
    #
    # df = df.fillna(method='bfill')
    # print(df.head())
    #
    # df = df.dropna()
    # df.dropna(axis='rows')
    # df.dropna(axis=0)
    # print(df.head())
    #
    # df = df.dropna(axis='columns')
    # df = df.dropna(axis=1)
    # print(df.head())

    aapl_df = pd.read_csv('./bin/data/AAPL.csv')
    # boolean type indexing
    print(aapl_df[aapl_df.isin([np.nan, np.inf, -np.inf]).any(axis=1)])
