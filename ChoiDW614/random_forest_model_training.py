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
from XGBoost_model_training import get_confusion_matrix
import seaborn as sns
sns.set()


def training():
    df = make_technical_indicator()
    y_var = df['target']
    x_var = df.drop(['target', 'OPEN', 'HIGH', 'LOW', 'VOLUME', 'CLOSE_SPY'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(x_var, y_var, test_size=0.3, shuffle=False, random_state=3)

    n_estimators = range(10, 200, 10)
    params = {
        'bootstrap': [True],
        'n_estimators': [10],   # n_estimators
        'max_depth': [8],     # [4, 6, 8, 10, 12]
        'min_samples_leaf': [2],   # [2, 3, 4, 5]
        'min_samples_split': [6],      # [2, 4, 6, 8, 10]
        'max_features': [4]
    }
    my_cv = TimeSeriesSplit(n_splits=5).split(X_train)

    clf = GridSearchCV(RandomForestClassifier(), params, cv=my_cv, n_jobs=-1)
    clf.fit(X_train, y_train)

    print('best parameter:\n', clf.best_params_)
    print('best prediction:{0:.4f}'.format(clf.best_score_))

    pred_con = clf.predict(X_test)
    accuracy_con = accuracy_score(y_test, pred_con)
    print('accuracy:{0:.4f}'.format(accuracy_con))
    get_confusion_matrix(y_test, pred_con)
