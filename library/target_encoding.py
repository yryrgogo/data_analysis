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


def target_encoding(base, df, key, target, method_list, prefix='', test=0, select_list=[], impute=1208, seed=1208):
    '''
    Explain:
        TARGET関連の特徴量を4partisionに分割したデータセットから作る.
        1partisionの特徴量は、残り3partisionの集計から作成する。
        test対する特徴量は、train全てを使って作成する
    Args:
        df(DF)               : 入力データ。カラムにはkeyとvalid_noがある前提
        level(str/list/taple): 目的変数を集計する粒度
        key                  : ユニークカラム名
        target               : 目的変数となるカラム名
        method(str)          : 集計のメソッド
        select_list(list)    : 特定のfeatureのみ保存したい場合はこちらにリストでfeature名を格納
    Return:
        カラム名は{prefix}{target}@{level}
    '''

    ' levelはリストである必要がある '
    if str(type(level)).count('str'):
        level = [level]
    elif str(type(level)).count('tuple'):
        level = list(level)

    train = df[~df[target].isnull()]

    val_col = 'valid_no'
    tmp_val = train[[key, target]].reset_index(drop=True)
    x = train[key].to_frame()
    y = train[target].values

    ' KFold '
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for trn_idx, val_idx in cv.split(x, y):
        cnt+=1

        valid_no = np.zeros(len(val_idx))+cnt
        tmp = pd.DataFrame({'index':val_idx, val_col:valid_no})

        if cnt==1:
            tmp_result = tmp.copy()
        else:
            tmp_result = pd.concat([tmp_result, tmp], axis=0)

    tmp_result.set_index('index', inplace=True)

    ' valid_colをつける '
    df_val = tmp_val.join(tmp_result)
    df = df.merge(df_val, on=key, how='left')
    df[val_col] = df[val_col].where(df[val_col]>=0, -1)

    print(df[val_col].value_counts())
    sys.exit()

    del train, tmp_val, x, y
    gc.collect()

    tmp_base = data[[key, val_col] + level].drop_duplicates()
    if len(base)>0:
        base = base[key].to_frame().merge(tmp_base, on=key, how='left')

    for method in method_list:
        result = pd.DataFrame([])
        valid_list = data[val_col].drop_duplicates().values
        if test == 0:
            valid_list.remove(-1)

        for valid_no in valid_list:

            if valid_no == -1:
                df = data
            else:
                df = data.query('is_train==1')
            '''
            集計に含めないpartisionのDFをdf_val.
            集計するpartisionのDFをdf_aggとして作成
            '''
            df_val = df[df[val_col] == valid_no][level].drop_duplicates()
            #  logger.info(f"\ndf_val: {df_val.shape}")

            df_agg = df[df[val_col] != valid_no][level+[target]]
            #  logger.info(f"\ndf_agg: {df_agg.shape}")

            #  logger.info(f'\nlevel: {level}\nvalid_no: {valid_no}')
            df_agg = base_aggregation(
                data=df_agg,
                level=level,
                feature=target,
                method=method
            )

            ' リークしないようにvalidation側のデータにJOIN '
            tmp_result = df_val.merge(df_agg, on=level, how='left')
            tmp_result[val_col] = valid_no

            if len(result) == 0:
                result = tmp_result
            else:
                result = pd.concat([result, tmp_result], axis=0)
            #  logger.info(f'\ntmp_result shape: {result.shape}')

        result = base.merge(result, on=level+[val_col], how='left')

        for col in result.columns:
            if col.count('bin') and not(col.count(target)):
                result.drop(col, axis=1, inplace=True)

        if impute != 1208:
            print(result.head())
            result.fillna(impute, inplace=True)

        #  logger.info(f'\nresult shape: {result.shape}')
        #  logger.info(f'\n{result.head()}')

        make_npy(result, ignore_list, prefix, select_list=select_list, npy_key=npy_key)


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
