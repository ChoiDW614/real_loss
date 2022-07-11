# -*- coding: euc-kr -*-

import pandas as pd
import pandas_datareader as dr

from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
#matplotlib inline
import seaborn as sns

#클러스터링 시작

num_of_clusters = range(2, 12)
error = []

for num_clusters in num_of_clusters:
    clusters = KMeans(num_clusters)
    clusters.fit(final_df)
    error.append(clusters.inertia_/100)

table = pd.DataFrame({"Cluster_Numbers":num_of_clusters, "Error_Term":error})
print(table)

