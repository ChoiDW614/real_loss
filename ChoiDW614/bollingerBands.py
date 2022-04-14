import pandas as pd
import matplotlib.pylab as plt


def read_data(file_root):
    # read data
    pd.set_option('display.max_columns', 6)
    df = pd.read_csv(file_root)
    # print(df.describe())

    # extract date and Adj Close from df
    price_df = df.loc[:, ['Date', 'Adj Close']].copy()  # preventing modification of source data
    # print(price_df.head())

    price_df.set_index(['Date'], inplace=True)
    # print(price_df.head())

    # price_df['center'] = price_df['Adj Close'].rolling(window=20).mean()
    # price_df['ub'] = price_df['center'] + 2 * price_df['Adj Close'].rolling(window=20).std()
    # price_df['lb'] = price_df['center'] - 2 * price_df['Adj Close'].rolling(window=20).std()
    # print(price_df.iloc[18:25])
    return price_df


def bollinger_band(price_df, n, sigma):
    bb = price_df.copy()
    bb['center'] = price_df['Adj Close'].rolling(n).mean()
    bb['ub'] = bb['center'] + sigma * price_df['Adj Close'].rolling(n).std()
    bb['lb'] = bb['center'] - sigma * price_df['Adj Close'].rolling(n).std()
    return bb


def create_trade_book(sample):
    book = sample[['Adj Close']].copy()
    book['trade'] = ''
    return book


def tradings(sample, book):
    for i in sample.index:
        if sample.loc[i, 'Adj Close'] > sample.loc[i, 'ub']:
            book.loc[i, 'trade'] = ''
        elif sample.loc[i, 'lb'] > sample.loc[i, 'Adj Close']:
            if book.shift(1).loc[i, 'trade'] == 'buy':
                book.loc[i, 'trade'] = 'buy'
            else:
                book.loc[i, 'trade'] = 'buy'
        elif sample.loc[i, 'ub'] >= sample.loc[i, 'Adj Close'] >= sample.loc[i, 'lb']:
            if book.shift(1).loc[i, 'trade'] == 'buy':
                book.loc[i, 'trade'] = 'buy'
            else:
                book.loc[i, 'trade'] = ''
    return book


def returns(book):
    rtn = 1.0
    book['return'] = 1
    buy = 0.0
    sell = 0.0
    for i in book.index:
        if book.loc[i, 'trade'] == 'buy' and book.shift(1).loc[i, 'trade'] == '':
            buy = book.loc[i, 'Adj Close']
            print('진입일 :', i, 'long 진입가격 : ', buy)
        elif book.loc[i, 'trade'] == '' and book.shift(1).loc[i, 'trade'] == 'buy':
            sell = book.loc[i, 'Adj Close']
            rtn = (sell - buy) / buy + 1    # cal of profit and loss
            book.loc[i, 'return'] = rtn
            print('청산일 : ', i, 'long 진입가격 : ', buy, ' | long 청산가격 : ',
                  sell, ' | return: ', round(rtn, 4), '\n')

        if book.loc[i, 'trade'] == '':
            buy = 0.0
            sell = 0.0

    acc_rtn = 1.0
    for i in book.index:
        rtn = book.loc[i, 'return']
        acc_rtn = acc_rtn * rtn
        book.loc[i, 'acc return'] = acc_rtn

    print('Accumulated return: ', round(acc_rtn, 4))
    return round(acc_rtn, 4)


def bollinger_bands_test():
    price_df = read_data('./bin/data/AAPL.csv')
    n = 20
    sigma = 2
    bollinger = bollinger_band(price_df, n, sigma)

    base_date = '2018-01-01'
    last_date = '2022-04-13'
    sample = bollinger.loc[base_date:last_date]
    # print(sample.head())

    book = create_trade_book(sample)
    book = tradings(sample, book)
    # print(book.tail(10))

    returns(book)
    book['acc return'].plot()
    plt.show()
