# -*- coding: euc-kr -*-

import warnings
import os
import glob
import pandas as pd
import numpy as np
import datetime

warnings.filterwarnings(action="ignore")

def data_preprocessing(sample, ticker, base_date):
    sample["CODE"] = ticker
    sample = sample[sample["Date"] >= base_date][["Date", "CODE", "Adj Close"]].copy()

    sample.reset_index(inplace=True, drop=True)
    sample["STD_YM"] = sample["Date"].map(lambda x : datetime.datetime.strptime(x, "%Y-%m-%d").strftime("%Y-%m"))
    sample["1M_RET"] = 0.0
    ym_keys = list(sample["STD_YM"].unique())
    return sample, ym_keys

def create_trade_book(sample, sample_codes):
    book = pd.DataFrame()
    book = sample[sample_codes].copy()
    book["STD_YM"] = book.index.map(lambda x : datetime.datetime.strptime(x, "%Y-%m-%d").strftime("%Y-%m"))
    for c in sample_codes:
        book["p " + c] = ""
        book["r " + c] = ""
    return book

def tradings(book, s_codes):
    std_ym = ""
    buy_phase = False
    for s in s_codes:
        print(s)
        for i in book.index:
            if book.loc[i, "p " + s] == "" and book.shift(1).loc[i, "p " + s] == "ready " + s:
                std_ym = book.loc[i, "STD_YM"]
                buy_phase = True

            if book.loc[i, "p " + s] == "" and book.loc[i, "STD_YM"] == std_ym and buy_phase == True:
                book.loc[i, "p " + s] = "buy " + s

            if book.loc[i, "p " + s] == "":
                std_ym = None
                buy_phase = False
    return book

def multi_returns(book, s_codes):
    rtn = 1.0
    buy_dict = {}
    num = len(s_codes)
    sell_dict = {}

    for i in book.index:
        for s in s_codes:
            if book.loc[i, "p " + s] == "buy " + s and \
                book.shift(1).loc[i, "p " + s] == "ready " + s and \
                book.shift(2).loc[i, "p " + s] == "":
                buy_dict[s] = book.loc[i, s]

            elif book.loc[i, "p " + s] == "" and book.shift(1).loc[i, "p " + s] == "buy " + s:
                sell_dict[s] = book.loc[i, s]
                rtn = (sell_dict[s] / buy_dict[s]) - 1
                book.loc[i, "r " + s] = rtn
                print("���� û���� : ", i, " ���� �ڵ� : ", s, "long ���԰��� : ", buy_dict[s], " | long û�갡�� : ", sell_dict[s], " | return:", round(rtn * 100, 2), "%")

            if book.loc[i, "p " + s] == "":
                buy_dict[s] = 0.0
                sell_dict[s] = 0.0


    acc_rtn = 1.0

    for i in book.index:
        rtn = 0.0
        count = 0
        for s in s_codes:
            if book.loc[i, "p " + s] == "" and book.shift(1).loc[i, "p " + s] == "buy " + s:
                count += 1
                rtn += book.loc[i, "r " + s]

        if (rtn != 0.0) & (count != 0) :
            acc_rtn *= (rtn / count) + 1
            print("���� û���� : ", i, "û�� ����� : ", count, "û�� ���ͷ� : ", round((rtn / count), 4), "���� ���ͷ� : ", round(acc_rtn, 4))

        book.loc[i, "acc_rtn"] = acc_rtn

    print ("���� ���ͷ� :", round(acc_rtn, 4))

files = glob.glob("C:/Users/djw04/Desktop/PythonQuant/*.csv")

month_last_df = pd.DataFrame(columns=["Date", "CODE", "1M_RET"])
stock_df = pd.DataFrame(columns = ["Date", "CODE", "Adj Close"])

for file in files:
    if os.path.isdir(file):
        print("%s <DIR> " %file)

    else:
        folder, name = os.path.split(file)
        head, tail = os.path.splitext(name)
        print(file)
        read_df = pd.read_csv(file)

        price_df, ym_keys = data_preprocessing(read_df, head, base_date="2010-01-02")
        stock_df = stock_df.append(price_df.loc[:, ["Date", "CODE", "Adj Close"]], sort=False)
        
        for ym in ym_keys:
            m_ret = price_df.loc[price_df[price_df["STD_YM"] == ym].index[-1], "Adj Close"] / price_df.loc[price_df[price_df["STD_YM"] == ym].index[0], "Adj Close"]
            #m_ret�� �� ������ �� �� ���� / �� �� ������ �ǹ�.
            price_df.loc[price_df["STD_YM"] == ym, ["1M_RET"]] = m_ret
            #�� ������ �ش� ���� 1M_RET�� m_ret���� ����
            month_last_df = month_last_df.append(price_df.loc[price_df[price_df["STD_YM"] == ym].index[-1], ["Date", "CODE", "1M_RET"]])
            #���� ���� ������ date, code, 1m_ret������ month_last_df�� �߰���

month_ret_df = month_last_df.pivot("Date", "CODE", "1M_RET").copy()
#dataframe�� �籸���ϴµ� row���� date�� ����, column���� code�� ����, Scalar���� 1M_RET���� �����Ѵ�.
month_ret_df = month_ret_df.rank(axis=1, ascending=False, method="max", pct=True)
#���� row�� ���ؼ� ���� ��, ū ������ 1 -> 2 -> 3...���� �ο�, �� ���� �����ڰ� ���� ��� �� ū������ ����, ���� ���� ������ 1���Ͽ� �ۼ�Ʈ�� ǥ��.

month_ret_df = month_ret_df.where(month_ret_df < 0.4, np.nan)
#month_ret_df�� �� ���� 0.4���� ������� ���ΰ�, ũ�ų� ������ np.nan������ �ٲ�. (���� 40���θ� ����)
month_ret_df.fillna(0, inplace=True)
month_ret_df[month_ret_df != 0] = 1
stock_codes = list(stock_df["CODE"].unique())

sig_dict = dict()
for date in month_ret_df.index:
    ticker_list = list(month_ret_df.loc[date, month_ret_df.loc[date, :] >= 1.0].index) # Series�� index �̹Ƿ� 1 �̻��� CODE���� list�� ���ҷ� ����.
    sig_dict[date] = ticker_list
    #sig_dict = {"��¥" : ["��¥", "��¥", ... , "��¥" �ε� ������ ������ 1 �̻��� element�� ���� �����ϴ�.]}

stock_c_matrix = stock_df.pivot("Date", "CODE", "Adj Close").copy()
book = create_trade_book(stock_c_matrix, list(stock_df["CODE"].unique()))

for date, values in sig_dict.items():
    for stock in values:
        book.loc[date, "p " + stock] = "ready " + stock

#3�ܰ� : ��ȣ������� Ʈ���̵� + �����Ŵ�
book = tradings(book, stock_codes)
print(book.loc["2012-01-27":"2012-03-01", ["AAPL", "p AAPL", "r AAPL"]])

#4�ܰ� : ���ͷ� ���

multi_returns(book, stock_codes)