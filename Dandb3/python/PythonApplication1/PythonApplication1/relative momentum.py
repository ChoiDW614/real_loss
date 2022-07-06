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
                print("개별 청산일 : ", i, " 종목 코드 : ", s, "long 진입가격 : ", buy_dict[s], " | long 청산가격 : ", sell_dict[s], " | return:", round(rtn * 100, 2), "%")

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
            print("누적 청산일 : ", i, "청산 종목수 : ", count, "청산 수익률 : ", round((rtn / count), 4), "누적 수익률 : ", round(acc_rtn, 4))

        book.loc[i, "acc_rtn"] = acc_rtn

    print ("누적 수익률 :", round(acc_rtn, 4))

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
            #m_ret은 한 종목의 월 말 종가 / 월 초 종가를 의미.
            price_df.loc[price_df["STD_YM"] == ym, ["1M_RET"]] = m_ret
            #그 종목의 해당 달의 1M_RET에 m_ret값을 대입
            month_last_df = month_last_df.append(price_df.loc[price_df[price_df["STD_YM"] == ym].index[-1], ["Date", "CODE", "1M_RET"]])
            #월말 제일 마지막 date, code, 1m_ret값들을 month_last_df에 추가함

month_ret_df = month_last_df.pivot("Date", "CODE", "1M_RET").copy()
#dataframe을 재구성하는데 row값을 date에 따라, column값을 code에 따라, Scalar값은 1M_RET으로 설정한다.
month_ret_df = month_ret_df.rank(axis=1, ascending=False, method="max", pct=True)
#같은 row에 대해서 값을 비교, 큰 수부터 1 -> 2 -> 3...으로 부여, 등 수는 동점자가 있을 경우 더 큰값으로 취함, 제일 높은 점수를 1로하여 퍼센트로 표현.

month_ret_df = month_ret_df.where(month_ret_df < 0.4, np.nan)
#month_ret_df의 각 항이 0.4보다 작을경우 냅두고, 크거나 같으면 np.nan값으로 바꿈. (상위 40프로만 선택)
month_ret_df.fillna(0, inplace=True)
month_ret_df[month_ret_df != 0] = 1
stock_codes = list(stock_df["CODE"].unique())

sig_dict = dict()
for date in month_ret_df.index:
    ticker_list = list(month_ret_df.loc[date, month_ret_df.loc[date, :] >= 1.0].index) # Series의 index 이므로 1 이상인 CODE들이 list의 원소로 들어간다.
    sig_dict[date] = ticker_list
    #sig_dict = {"날짜" : ["날짜", "날짜", ... , "날짜" 인데 원소의 개수가 1 이상인 element의 수와 동일하다.]}

stock_c_matrix = stock_df.pivot("Date", "CODE", "Adj Close").copy()
book = create_trade_book(stock_c_matrix, list(stock_df["CODE"].unique()))

for date, values in sig_dict.items():
    for stock in values:
        book.loc[date, "p " + stock] = "ready " + stock

#3단계 : 신호목록으로 트레이딩 + 포지셔닝
book = tradings(book, stock_codes)
print(book.loc["2012-01-27":"2012-03-01", ["AAPL", "p AAPL", "r AAPL"]])

#4단계 : 수익률 계산

multi_returns(book, stock_codes)