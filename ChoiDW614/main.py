from src import API_to_load_data as ald
import bollinger_bands as bb
import dual_momentum as dm
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import make_technical_indicator
import XGBoost_model_training
import random_forest_model_training
import KNN
import Kmeans_cluster

if __name__ == '__main__':
    # XGBoost_model_training.training()
    # random_forest_model_training.training()
    Kmeans_cluster.clustering()


def use_bollinger_band():
    warnings.filterwarnings(action='ignore')
    warnings.filterwarnings(action='default')

    if ald.read_data('sample', 'AAPL', '나스닥'):
        bb.run_bollinger_bands('sample')
    else:
        bb.run_bollinger_bands('sample', False)

    dm.run_dual_momentum()