#Data Manipulation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from pandas_datareader import data as pdr
#import fix_yahoo_finance as yf


df = pdr.get_data_yahoo("SPY", "2012-01-01", "2017-01-01")
df = df.dropna()
#print(df.head())

tmp_df = df[["Open", "High", "Low", "Close"]].copy()
#print(tmp_df.head())

tmp_df["Open-Close"] = tmp_df["Open"] - tmp_df["Close"]
tmp_df["High-Low"] = tmp_df["High"] - tmp_df["Low"]
tmp_df = tmp_df.dropna()

X = tmp_df[["Open-Close", "High-Low"]]
Y = np.where(tmp_df["Close"].shift(-1) > tmp_df["Open"].shift(-1), 1, 0)
#print(X.head())

split_percentage = 0.7
split = int(split_percentage * len(tmp_df))
X_train = X[:split]
Y_train = Y[:split]

X_test = X[split:]
Y_test = Y[split:]

train_acc = []
test_acc = []

#for n in range(1, 15):
#	clf = KNeighborsClassifier(n_jobs=-1, n_neighbors=n)
#	clf.fit(X_train, Y_train)
#	prediction = clf.predict(X_test)
#	train_acc.append(clf.score(X_train, Y_train))
#	test_acc.append((prediction == Y_test).mean())

#plt.figure(figsize = (12, 9))
#plt.plot(range(1, 15), train_acc, label = "TRAIN set")
#plt.plot(range(1, 15), test_acc, label = "TEST set")
#plt.xlabel("n_neighbors")
#plt.ylabel("accuracy")
#plt.xticks(np.arange(0, 16, step=1))
#plt.legend()
#plt.show()

knn = KNeighborsClassifier(n_neighbors = 7)
knn.fit(X_train, Y_train)

accuracy_train = accuracy_score(Y_train, knn.predict(X_train))
accuracy_test = accuracy_score(Y_test, knn.predict(X_test))

print("Train Accuracy : %.2f" % accuracy_train)
print("Test Accuracy : %.2f" % accuracy_test)

#log -> addition is required, not multiplication
tmp_df["Predicted_Signal"] = knn.predict(X)
tmp_df["SPY_ret"] = np.log(tmp_df["Close"] / tmp_df["Close"].shift(1))
cum_spy_ret = tmp_df[split:]["SPY_ret"].cumsum() * 100

tmp_df["st_ret"] = tmp_df["SPY_ret"] * tmp_df["Predicted_Signal"].shift(1)
cum_st_ret = tmp_df[split:]["st_ret"].cumsum() * 100

plt.figure(figsize=(10, 5))
plt.plot(cum_spy_ret, color="r", label = "spy ret")
plt.plot(cum_st_ret, color="g", label = "st ret")
plt.legend()
plt.show()

std = cum_st_ret.std()
sharpe = (cum_st_ret - cum_spy_ret) / std
sharpe = sharpe.mean()
print("Sharpe ratio : %.2f" % sharpe)