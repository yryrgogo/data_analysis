import numpy as np
import pandas as pd
from scipy.stats.mstats import mquantiles
import lightgbm as lgb
import datetime
from tqdm import tqdm
import sys
import xgboost as xgb
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, ridge
import re
import gc
import glob
import os
HOME = os.path.expanduser('~')
sys.path.append(f"{HOME}/kaggle/github/library/")
import utils
from utils import logger_func, get_categorical_features, get_numeric_features, pararell_process, round_size
sys.path.append(f"{HOME}/kaggle/github/model/")
from classifier import cross_validation, data_check
import pickle
logger = logger_func()
pd.set_option('max_columns', 200)
pd.set_option('max_rows', 200)


def x_ray_caliculation(df_valid, col, val, model):
    logger.info(f'''
    #========================================================================
    # X-RAY CALICURATION START : {col}
    #========================================================================''')
    df_valid[col] = val
    pred = model.predict(df_valid)
    p_avg = np.mean(pred)
    return col, val, p_avg

def x_ray_wrapper(args):
    return x_ray_caliculation(*args)

def x_ray(logger, model, df, columns=False, max_sample=50):
    '''
    Explain:
    Args:
        columns: x-rayを出力したいカラムリスト
    Return:
    '''
    x_ray = False
    arg_list = []
    result = pd.DataFrame([])
    if not(columns):
        columns = df.columns
    for col in columns:
        xray_list = []
        val_cnt = df[col].value_counts()
        if len(val_cnt)>max_sample:
            length = max_sample-10
            val_array = val_cnt.head(length).index.values
            percentiles = np.linspace(0.05,0.95,num=10)
            val_percentiles = mquantiles(val_cnt.index.values, prob=percentiles, axis=0)
            max_val = df[col].max()
            min_val = df[col].min()
            r = round_size(max_val, max_val, min_val)
            val_percentiles = np.round(val_percentiles, r)
            val_array = np.hstack((val_array, val_percentiles))
        else:
            length = len(val_cnt)
            val_array = val_cnt.head(length).index.values
        val_array = np.sort(val_array)

        for val in val_array:
            arg_list.append([df, col, val, model])
        xray_list = pararell_process(x_ray_wrapper, arg_list)
        feature_list = []
        value_list = []
        pred_list = []
        for tmp_tuple in xray_list:
            feature_list.append(tmp_tuple[0])
            value_list.append(tmp_tuple[1])
            pred_list.append(tmp_tuple[2])

            result_dict = {
                'feature':feature_list,
                'value':value_list,
                'pred' :pred_list
            }
        tmp_result = pd.DataFrame(data=result_dict)
        if len(result):
            result = pd.concat([result, tmp_result])
        else:
            result = tmp_result.copy()
    print(result)
    sys.exit()
    #  tmp_xray = valid[col].to_frame('value')
    #  tmp_xray['feature'] = col
    #  tmp_xray['x_ray'] = xray_list

    #  if x_ray:
    #      x_ray = pd.concat([x_ray, tmp_xray], axis=0)
    #  else:
    #      x_ray = tmp_xray.copy()

    return x_ray


def main():

    metric = ['auc'][0]
    num_iterations = 3500
    learning_rate = 0.05
    early_stopping_rounds = 150
    model_type = 'lgb'
    fold_type = 'group'
    params = {'bagging_freq': 1, 'bagging_seed': 1012, 'colsample_bytree': 0.01, 'data_random_seed': 1012, 'feature_fraction_seed': 1012, 'lambda_l1': 0.1, 'lambda_l2': 0.5, 'learning_rate': 0.02, 'max_bin': 250, 'max_depth': 5, 'metric': 'auc', 'min_child_samples': 96, 'min_child_weight': 36, 'min_data_in_bin': 96, 'min_split_gain': 0.01, 'num_leaves': 11, 'num_threads': 35, 'objective': 'binary', 'random_seed': 1012, 'subsample': 1.0}

    nrows = 1000
    key = 'c_取引先集約コード'
    target = '翌年トラコンタ購買フラグ__t'
    df = pd.read_csv('../input/20180914_yanmar_drset_10model.csv', nrows=nrows)

    key_cols = [ 'c_取引先集約コード' ,'t_年月' ]
    base_cols = [col for col in df.columns if not(col.count('__')) or col.count('担い手') or col.count('経過')]
    #  feature_cols = [col for col in df.columns if col.count('__') and not(col.count('担い手')) and not(col.count('経過'))]
    feature_cols = [col for col in df.columns if col.count('__')]

    train = df[feature_cols]

    #  result, col_length = cross_validation(
    #      logger=logger,
    #      train=train,
    #      target=target,
    #      params=params,
    #      metric=metric,
    #      fold_type=fold_type,
    #      num_iterations=num_iterations,
    #      learning_rate=learning_rate,
    #      early_stopping_rounds=early_stopping_rounds,
    #      model_type=model_type
    #  )

    ' xray params '
    max_sample = 50
    with open('../output/clf.pickle', 'rb') as f:
        clf = pickle.load(f)

    train = data_check(logger, df=train, target=target)
    x_ray(logger, clf, train)

    sys.exit()

if __name__ == '__main__':
    main()
