# -*- coding: euc-kr -*-
import pandas as pd
import pandas_datareader as dr

from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
#matplotlib inline
import seaborn as sns

sp500_url = "https://en.wikipedia.org/wiki/List_of_S&P_500_companies"
data_table = pd.read_html(sp500_url)
tickers = data_table[0]["Symbol"].tolist()

tickers = tickers[0:60]

#sp500_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
#data_table_security = pd.read_html(sp500_url)
security = data_table[0]["Security"].tolist()
security = security[0:60]

#sp500_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
#data_table_security = pd.read_html(sp500_url)
sector = data_table[0]["GICS Sector"].tolist()
sector = sector[0:60]

#print(len(tickers))
#print(len(security))
#print(len(sector))

prices_list = []

for ticker in tickers:
    try:
        prices = dr.DataReader(ticker, "yahoo", "01/01/2017")["Adj Close"]
        prices = pd.DataFrame(prices)
        prices.columns = [ticker]
        prices_list.append(prices)
    except:
        pass
    prices_df = pd.concat(prices_list, axis=1)

prices_df.sort_index(inplace=True)

#print(prices_df.head())

df = prices_df.pct_change().iloc[1:].T #transpose
#print(df.head())

companies = list(df.index)
movements = df.values

normalize = Normalizer()
array_norm = normalize.fit_transform(df)
df_norm = pd.DataFrame(array_norm, columns=df.columns)
final_df = df_norm.set_index(df.index)
#print(final_df.head(10))

#������ �����Ͱ� �ִ��� Ȯ�� �� ���
col_mask = df.isnull().any(axis=0)
row_mask = df.isnull().any(axis=1)
print(df.loc[row_mask, col_mask])

#Ŭ�����͸� ����

num_of_clusters = range(2, 12)
error = []

for num_clusters in num_of_clusters:
    clusters = KMeans(num_clusters)
    clusters.fit(final_df)
    error.append(clusters.inertia_/100)

table = pd.DataFrame({"Cluster_Numbers":num_of_clusters, "Error_Term":error})
#print(table)

#plt.figure(figsize=(15, 10))
#plt.plot(table["Cluster_Numbers"], table["Error_Term"], marker = "D", color = "red")
#plt.xlabel("Number of Clusters")
#plt.ylabel("SSE")
#plt.show()

clusters = KMeans(7)
clusters.fit(final_df)
#print(clusters.labels_)

labels = clusters.predict(movements) # label�� K���� �׷� �� ȸ�縶�� ��� �׷쿡 ���ϴ����� ��Ÿ�� array��.
#print(labels)

clustered_result = pd.DataFrame({"labels": labels, "tickers" : companies, "full-name":security, "sector":sector})
clustered_result.sort_values("labels")

final_df["Cluster"] = clusters.labels_

plt.figure(figsize=(12, 6))
sns.countplot(x = "Cluster", data = final_df, palette="magma")
plt.title("Cluster_count")
plt.show()
plt.savefig("cluster_count.png", dpi=300)