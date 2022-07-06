import warnings
warnings.filterwarnings('ignore')
import glob
import os
import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from xgboost import plot_importance
from sklearn.metrics import f1_score, recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from sklearn import svm
from make_technical_indicator import make_technical_indicator
import seaborn as sns
sns.set()


def get_train_test_set():
    df = make_technical_indicator()
    y_var = df['target']
    x_var = df.drop(['target', 'OPEN', 'HIGH', 'LOW', 'VOLUME', 'CLOSE_SPY'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(x_var, y_var, test_size=0.3, shuffle=False, random_state=3)

    # train_count = y_train.count()
    # test_count = y_test.count()

    # print('train set label ratio')
    # print(y_train.value_counts() / train_count)
    # print('test set label ratio')
    # print(y_test.value_counts() / test_count)
    return X_train, X_test, y_train, y_test


def get_confusion_matrix(y_test, pred):
    confusion = confusion_matrix(y_test, pred)
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    roc_score = roc_auc_score(y_test, pred)
    print('confusion matrix')
    print('accuracy:{0:.4f}, precision:{1:.4f}, recall:{2:.4f}, F1:{3:.4f}, ROC AUC score:{4:.4f}'
          .format(accuracy, precision, recall, f1, roc_score))


def training():
    X_train, X_test, y_train, y_test = get_train_test_set()
    xgb_dis = XGBClassifier(n_estimators=400, learning_rate=0.1, max_depth=3)
    xgb_dis.fit(X_train, y_train)
    xgb_pred = xgb_dis.predict(X_test)
    print(xgb_dis.score(X_train, y_train))
    get_confusion_matrix(y_test, xgb_pred)
