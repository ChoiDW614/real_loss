# -*- coding: euc-kr -*-
 
#import pandas as pd
#import numpy as np
#import matplotlib.pyplot as plt

#df = pd.read_csv("C:/Users/djw04/Desktop/PythonQuant/AMZN.csv", index_col="Date", parse_dates=["Date"])
##print(df.head())

##print(df[df.isin([np.nan, np.inf, -np.inf]).any(1)])

#price_df = df.loc[:, ["Adj Close"]].copy()
##price_df.plot(figsize=(16, 9))
##plt.show()

##from_date = "1997-01-03"
##to_date = "2003-01-03"
##price_df.loc[from_date:to_date].plot(figsize=(16,9))
##plt.show()

#price_df["daily_rtn"] = price_df["Adj Close"].pct_change()
#price_df["st_rtn"] = (1+price_df["daily_rtn"]).cumprod()
##price_df.plot()
##plt.show()
##print(price_df.loc["2019-06-11":"2019-06-24", :])

##base_date = "2011-01-03"
##tmp_df = price_df.loc[base_date:, ["st_rtn"]] / price_df.loc[base_date, ["st_rtn"]]
##last_date = tmp_df.index[-1]
##print("누적 수익 : ", tmp_df.loc[last_date, "st_rtn"])
##tmp_df.plot(figsize=(16,9))
##plt.show()

#CAGR = price_df.loc["2022-04-13", "st_rtn"] ** (252./len(price_df.index)) - 1
#print(CAGR)

#historical_max = price_df["Adj Close"].cummax()
#daily_drawdown = price_df["Adj Close"] / historical_max - 1.0
#historical_dd = daily_drawdown.cummin()
#historical_dd.plot()
#plt.show()

#VOL = np.std(price_df["daily_rtn"]) * np.sqrt(252.)
#Sharpe = np.mean(price_df["daily_rtn"]) / np.std(price_df["daily_rtn"]) * np.sqrt(252.)
#MDD = historical_dd.min()

#print("CAGR : ", round(CAGR*100, 2), "%")
#print("Sharpe : ", round(Sharpe, 2))
#print("VOL : ", round(VOL*100, 2), "%")
#print("MDD : ", round(-1*MDD*100, 2), "%")