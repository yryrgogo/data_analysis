import sys
try:
    model_type=sys.argv[2]
except IndexError:
    model_type='lgb'
try:
    learning_rate = float(sys.argv[3])
except IndexError:
    learning_rate = 0.1
try:
    early_stopping_rounds = int(sys.argv[5])
except IndexError:
    early_stopping_rounds = 150
num_iterations = 10000
try:
    experience_code = int(sys.argv[1])
except IndexError:
    experience_code = 0
try:
    seed = int(sys.argv[4])
except IndexError:
    seed = 1208

truncate_flg = False
iter_no = 1000
rank_list = [450, 350, 300, 250]

import gc
import numpy as np
import pandas as pd
import datetime
from datetime import date, timedelta
import glob

import re
import shutil
from sklearn.metrics import log_loss, roc_auc_score
from itertools import combinations
from multiprocessing import Pool
import multiprocessing
import lightgbm as lgb
import xgboost as xgb
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from tqdm import tqdm

sys.path.append('../engineering')
from select_feature import move_to_second_valid

sys.path.append('../model')
from lgbm_clf import validation, prediction, cross_validation
from params_lgbm import train_params

sys.path.append('../../../github/module/')
from preprocessing import set_validation, split_dataset, factorize_categoricals, get_dummies, data_regulize
from load_data import pararell_load_data, x_y_split
from utils import get_categorical_features, get_numeric_features
from logger import logger_func
from make_file import make_feature_set, make_npy
from statistics_info import correlation


def much_feature_validation(base, path, move_path, dummie=0, val_col='valid_no'):

    feature_list = np.array(glob.glob('../features/1_first_valid/*.npy'))
    feature_list = np.sort(feature_list)
    if len(feature_list)>1000:
        feature_list = feature_list[:1000]

    for feature in feature_list:
        shutil.move(feature, '../features/3_winner/')

    logger.info(f'move feature:{len(feature_list)}')

    key_list = []
    for rank in rank_list:
        logger.info(f'rank:{rank}')
        _, _, importance = first_train(base, path, dummie=0, val_col=val_col)
        move_to_second_valid(best_select=importance, rank=rank, key_list=key_list)


def first_train(train, path, dummie=0, val_col='valid_no'):

    cv_feim, col_length = cross_validation(
        logger=logger,
        dataset=train,
        target=target,
        val_col=val_col,
        params=params,
        metric=metric,
        truncate_flg=truncate_flg,
        num_iterations=num_iterations,
        learning_rate=learning_rate,
        early_stopping_rounds=early_stopping_rounds,
        model_type=model_type
    )

    ' 最初のスコア '
    cv_feim.to_csv( f'../valid/{model_type}_feat{col_length}_{metric}{str(first_score)[:7]}_lr{learning_rate}.csv', index=False)
    importance = first[['feature', 'avg_importance', 'rank']].sort_values(by='avg_importance')

    return first_score, train, importance
