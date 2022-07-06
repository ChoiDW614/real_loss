import pandas as pd
import numpy as np
import datetime


# position record
def create_trade_book(df):
    book = df.copy()
    book.set_index(['Date'], inplace=True)
    book['trade'] = ''
    return book


def tradings(month_last_df, book, ticker):
    for x in month_last_df.index:
        signal = ''
        momentum_index = month_last_df.loc[x, 'BF_1M_Adj Close'] / month_last_df.loc[x, 'BF_12M_Adj Close'] - 1
        flag = True if ((momentum_index > 0.0) and (momentum_index != np.inf) and (momentum_index != -np.inf))\
            else False  # and True
        if flag:
            signal = 'buy ' + ticker
        print('날짜 : ', x, '모멘텀 인덱스 : ', momentum_index, 'flag : ', flag, 'signal : ', signal)
        book.loc[x:, 'trade'] = signal


# strategy return
def returns(book, ticker):
    rtn = 1.0
    book['return'] = 1
    buy = 0.0
    sell = 0.0
    for i in book.index:
        if book.loc[i, 'trade'] == 'buy ' + ticker and book.shift(1).loc[i, 'trade'] == '':
            # Enter long
            buy = book.loc[i, 'Adj Close']
            print('진입일 : ', i, 'long 진입가격 : ', buy)
        elif book.loc[i, 'trade'] == 'buy ' + ticker and book.shift(1).loc[i, 'trade'] == 'buy '+ ticker:
            # Possession
            current = book.loc[i, 'Adj Close']
            rtn = (current - buy) / buy + 1
            book.loc[i, 'return'] = rtn
        elif book.loc[i, 'trade'] == '' and book.shift(1).loc[i, 'trade'] == 'buy ' + ticker:
            # sell
            sell = book.loc[i, 'Adj Close']
            rtn = (sell - buy) / buy + 1
            book.loc[i, 'return'] = rtn
            print('청산일 : ', i, 'long 진입가격 : ', buy, ' | long 청산가격 : ', sell,
                  ' | return : ', round(rtn, 4))

        if book.loc[i, 'trade'] == '':
            buy = 0.0
            sell = 0.0
            current = 0.0

    acc_rtn = 1.0
    for i in book.index:
        if book.loc[i, 'trade'] == '' and book.shift(1).loc[i, 'trade'] == 'buy ' + ticker:
            rtn = book.loc[i, 'return']
            acc_rtn = acc_rtn * rtn
            book.loc[i:, 'acc return'] = acc_rtn

    print('Accumulated return :', round(acc_rtn, 4))
    return round(acc_rtn, 4)


def run_dual_momentum():
    read_df = pd.read_csv('./bin/data/AAPL.csv')
    price_df = read_df.loc[:, ['Date', 'Adj Close']].copy()
    # print(price_df.head())
    price_df['STD_YM'] = price_df['Date'].map(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d').strftime('%Y-%m'))
    # print(price_df.head())

    month_list = price_df['STD_YM'].unique()
    month_last_df = pd.DataFrame()
    for m in month_list:
        month_last_df = month_last_df.append(price_df.loc[price_df[price_df['STD_YM'] == m].index[-1], :])

    month_last_df.set_index(['Date'], inplace=True)
    # print(month_last_df.head())

    month_last_df['BF_1M_Adj Close'] = month_last_df.shift(1)['Adj Close']
    month_last_df['BF_12M_Adj Close'] = month_last_df.shift(12)['Adj Close']
    month_last_df.fillna(0, inplace=True)
    # print(month_last_df.head(15))

    book = create_trade_book(price_df)
    # print(book.head())
    tradings(month_last_df, book, 'AAPL')
    returns(book, 'AAPL')
