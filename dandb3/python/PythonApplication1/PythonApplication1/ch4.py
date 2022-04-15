# -*- coding: euc-kr -*-

import numpy as np
import pandas as pd

df = pd.read_csv("C:/Users/djw04/Desktop/PythonQuant/SPY.csv")
#print(df.describe())
price_df = df.loc[:, ["Date", "Adj Close"]].copy()
#print(price_df.head())

price_df.set_index(["Date"], inplace=True)
#print(price_df.head())

#price_df["center"] = price_df["Adj Close"].rolling(window = 20).mean()
#print(price_df.iloc[18:25])

#price_df["ub"] = price_df["center"] + 2 * price_df["Adj Close"].rolling(window = 20).std()
#price_df["lb"] = price_df["center"] - 2 * price_df["Adj Close"].rolling(window = 20).std()

#print(price_df.iloc[18:25])

n = 20
sigma = 2

def bollinger_band(price_df, n, sigma):
    bb = price_df.copy()
    bb["center"] = bb["Adj Close"].rolling(n).mean()
    bb["ub"] = bb["center"] + sigma * bb["Adj Close"].rolling(n).std()
    bb["lb"] = bb["center"] - sigma * bb["Adj Close"].rolling(n).std()

    return bb

bollinger = bollinger_band(price_df, n, sigma)

base_date = "2009-01-02"
sample = bollinger.loc[base_date:]
#print(sample.head())

def create_trade_book(sample):
    book = sample[["Adj Close"]].copy()
    book["trade"] = ""
    return (book)

def tradings(sample, book):
    for i in sample.index:
        if sample.loc[i, "Adj Close"] > sample.loc[i, "ub"]:
            book.loc[i, "trade"] = ""
        elif sample.loc[i, "lb"] > sample.loc[i, "Adj Close"]:
            if book.shift(1).loc[i, "trade"] == "buy":
                book.loc[i, "trade"] = "buy"
            else:
                book.loc[i, "trade"] = "buy"
        elif sample.loc[i, "ub"] >= sample.loc[i, "Adj Close"] and sample.loc[i, "Adj Close"] >= sample.loc[i, "lb"]:
            if book.shift(1).loc[i, "trade"] == "buy":
                book.loc[i, "trade"] = "buy"
            else:
                book.loc[i, "trade"] = ""
    return (book)

book = tradings(sample, create_trade_book(sample))
print(book.tail(10))

def returns(book):
    rtn = 1.0
    book["return"] = 1
    buy = 0.0
    sell = 0.0
    for i in book.index:
        if book.loc[i, "trade"] == "buy" and book.shift(1).loc[i, "trade"] == "":
            buy = book.loc[i, "Adj Close"]
            print("진입일 : ", i, "long 진입가격 : ", buy)
        elif book.loc[i, "trade"] == "" and book.shift(1).loc[i, "trade"] == "buy":
            sell = book.loc[i, "Adj Close"]
            rtn = (sell - buy) / buy + 1
            book.loc[i, "return"] = rtn
            print("청산일 : ", i, "long 진입가격 : ", buy, " | long 청산가격 : ", sell, " | return:", round(rtn, 4))

        if book.loc[i, "trade"] == "":
            buy = 0.0
            sell = 0.0

    acc_rtn = 1.0
    for i in book.index:
        rtn = book.loc[i, "return"]
        acc_rtn = acc_rtn * rtn
        book.loc[i, "acc return"] = acc_rtn

    print ("Accumulated return :", round(acc_rtn, 4))
    return (round(acc_rtn, 4))

print(returns(book))

import matplotlib.pyplot as plt
book["acc return"].plot(figsize=(12,9))
plt.show()