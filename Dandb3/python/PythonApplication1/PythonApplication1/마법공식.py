# -*- coding: euc-kr -*-

#import csv
#line_list = []
#with open("C:/Users/djw04/Desktop/PythonQuant/PER_ROA.csv") as csv_file:
#    csv_reader = csv.reader(csv_file, delimiter=',')
#    for row in csv_reader:
#        if '' in row:
#            pass
#        else:
#            line_list.append(row)

#    df = pd.DataFrame(data=line_list[1:], columns=line_list[0])
#    print(df.head())

import FinanceDataReader as fdr
import pandas as pd
import numpy as np

krx_df = fdr.StockListing("KRX")

df = pd.read_csv("C:/Users/djw04/Desktop/PythonQuant/PER_ROA.csv", engine="python")
df.rename(columns={"ROE" : "ROA"}, inplace=True)
df = df.loc[:, "�����":"ROA"]
df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]


def sort_value(s_value, asc = True, standard = 0):
    """
    description
        Ư�� ��ǩ���� �����Ѵ�.
    parameters
        s_value : pandas Series
            ������ �����͸� �޴´�.
    asc : bool
        True : ��������
        False : ��������

    standard : int
        ���ǿ� �´� ���� True�� ��ü�ϱ� ���� ���ذ�

    returns
        s_value_mask_rank : pandas Series
            ���ĵ� ����
    """

    s_value_mask = s_value.mask(s_value < standard, np.nan)
    s_value_mask_rank = s_value_mask.rank(ascending=asc, na_option="bottom")  #��������(���� ������ 1 > 2 > 3 > ... �̴�), nan���� ���� ���� ����(���� big)���� �Ҵ��.

    return s_value_mask_rank

per = pd.to_numeric(df["PER"])
roa = pd.to_numeric(df["ROA"])   #������ �о�� �� ROA�� �� �о�ͼ� ��� ROE�� �о��

per_rank = sort_value(per, asc=True, standard=0)
roa_rank = sort_value(roa, asc=False, standard=0)

#print(per_rank.head())
#print(roa_rank.head())

result_rank = per_rank + roa_rank
result_rank = sort_value(result_rank, asc=True)
result_rank = result_rank.where(result_rank <= 10, 0)
result_rank = result_rank.mask(result_rank > 0, 1)

#print(result_rank.head(15))
#print(result_rank.sum())

mf_df = df.loc[result_rank > 0, ["�����", "�ð��Ѿ�"]].copy()
mf_stock_list = df.loc[result_rank > 0, "�����"].values
#print(mf_df)
#print(mf_stock_list)

mf_df["���� �ڵ�"] = ""
for stock in mf_stock_list:
    mf_df.loc[mf_df["�����"] == stock, "���� �ڵ�"] = krx_df[krx_df["Name"] == stock]["Symbol"].values

#print(mf_df)

mf_df["2019_���ͷ�"] = ""
cnt = 0
for x in mf_df["���� �ڵ�"].values:
    cnt += 1
    if cnt == 3:
        mf_df.loc[mf_df["���� �ڵ�"] == x, "2019_���ͷ�"] = np.nan
        continue
    #print(x, ", ", mf_df.loc[mf_df["���� �ڵ�"] == x, "�����"].values[0])
    df = fdr.DataReader(x, "2019-01-01", "2019-12-31")
    #print(df)
    cum_ret = df.loc[df.index[-1], "Close"] / df.loc[df.index[0], "Close"] - 1

    mf_df.loc[mf_df["���� �ڵ�"] == x, "2019_���ͷ�"] = cum_ret
    df = None

#print(mf_df)

for ind, val in enumerate(mf_df["���� �ڵ�"].values):
    code_name = mf_df.loc[mf_df["���� �ڵ�"] == val, "�����"].values[0]
    print(val, code_name)
    df = fdr.DataReader(val, "2019-01-01", "2019-12-31")
    if ind == 0:
        mf_df_rtn = pd.DataFrame(index=df.index)

    df["daily_rtn"] = df["Close"].pct_change(periods=1)
    df["cum_rtn"] = (1 + df["daily_rtn"]).cumprod()
    tmp = df.loc[:, ["cum_rtn"]].rename(columns={"cum_rtn":code_name})

    mf_df_rtn = mf_df_rtn.join(tmp, how="left")
    df = None

print(mf_df_rtn.tail())