from src import API_to_load_data as ald
import bollinger_bands as bb
import dual_momentum as dm
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import make_technical_indicator
import model_training

if __name__ == '__main__':
    model_training.training()


def use_bollinger_band():
    warnings.filterwarnings(action='ignore')
    warnings.filterwarnings(action='default')

    if ald.read_data('sample', 'AAPL', '나스닥'):
        bb.run_bollinger_bands('sample')
    else:
        bb.run_bollinger_bands('sample', False)

    dm.run_dual_momentum()
