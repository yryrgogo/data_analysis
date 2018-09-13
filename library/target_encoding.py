import numpy as np
import pandas as pd
import datetime
import glob
import sys
import re
import gc
from multiprocessing import Pool
import multiprocessing
from itertools import combinations
from sklearn.model_selection import StratifiedKFold

sys.path.append('../../../github/module/')
from preprocessing import set_validation, split_dataset, factorize_categoricals
from load_data import pararell_load_data, x_y_split
from utils import get_categorical_features, get_numeric_features
from make_file import make_feature_set, make_npy
from logger import logger_func
from feature_engineering import base_aggregation


#  logger = logger_func()
start_time = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())

ignore_list = [key, target, 'valid_no', 'is_train', 'is_test']


def main():

    level = key
    #  method_list = ['sum', 'mean', 'std']
    method_list = ['mean', 'std']
    #  method_list = ['max', 'min']
    #  method_list = ['sum', 'mean', 'std', 'max', 'min']

    base = pd.read_csv('../data/base.csv')

    select_list = []
    #  categorical = [col for col in data.columns if col.count('cluster')]

    for cat in categorical:
        target_encoding(data, cat, method_list, prefix, select_list=select_list, test=1, impute=1208)
    sys.exit()


if __name__ == '__main__':

    main()
