# -*- coding: euc-kr -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import FinanceDataReader as fdr

#df_krx = fdr.StockListing("KRX")
#print(len(df_krx))
#print(df_krx.head())

#df_spx = fdr.StockListing("S&P500")
#print(df_spx.head())
#print(len(df_spx))

#df = fdr.DataReader("001250", "2018")
#print(df.head(10))
#df["Close"].plot()
#plt.show()

#df = fdr.DataReader("AAPL", "2017")
#print(df.head(10))
#df["Close"].plot()
#plt.show()

exchange_map = {
    "KRX":"Seoul", "�ѱ� �ŷ���":"Seoul",
    "NASDAQ":"NASDAQ", "������":"NASDAQ",
    "NYSE":"NYSE", "�������ǰŷ���":"NYSE",
    "AMEX":"AMEX", "�̱����ǰŷ���":"AMEX",
    "SSE":"Shanghai", "����":"Shanghai", "������":"Shanghai",
    "SZSE":"Shenzhen", "��õ":"Shenzhen",
    "HKEX":"Hong Kong", "ȫ��":"Hong Kong",
    "TSE":"Tokyo", "����":"Tokyo",
}

#jp_df1 = fdr.DataReader(symbol="7751", start="2019-01-01", exchange="TSE")
jp_df2 = fdr.DataReader(symbol="7751", exchange="����")
#print(jp_df2)

#df = pd.read_csv("C:/Users/djw04/Desktop/PythonQuant/AMZN.csv", index_col="Date", parse_dates=["Date"])

#jp_df2.loc["1997-05-16", "Close"] = np.nan
#print(df["Close"])
#print(jp_df2[jp_df2.isin([np.nan, np.inf, -np.inf]).any(1)])

#price_df.plot()
#plt.show()

#print(price_df)

from_date = "1997-01-03"
to_date = "2003-01-03"
base_date = "2011-01-03"
price_df = jp_df2.loc[:, ["Close"]].copy()
price_df["daily_rtn"] = price_df["Close"].pct_change()
price_df["st_rtn"] = (1 + price_df["daily_rtn"]).cumprod()
print(price_df.loc[base_date:, ["st_rtn"]])
#tmp_df = price_df.loc[base_date:, ["st_rtn"]] / price_df.loc[base_date, ["st_rtn"]]
#print(tmp_df)