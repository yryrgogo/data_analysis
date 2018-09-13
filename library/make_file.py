import numpy as np
import pandas as pd
import sys, glob
from load_data import pararell_load_data
import os
HOME = os.path.expanduser('~')
sys.path.append(f"{HOME}/kaggle/github/library/")
import utils
from utils import logger_func, get_categorical_features, get_numeric_features, pararell_process
pd.set_option('max_columns', 200)
pd.set_option('max_rows', 200)


def make_npy(result, ignore_list=[], prefix='', suffix='', select_list=[], path='../features/1_first_valid/', logger=False, npy_key='@'):
    '''
    Explain:
        .npyで特徴量を保存する
    Args:
        result:
        ignore_features: npyとして保存しないカラムリスト
        prefix:
        suffix:
    Return:
    '''

    for feature in result.columns:
        if feature.count(npy_key) and feature not in ignore_list:
            filename = f'{prefix}{feature}'
            ' 環境パスと相性の悪い記号は置換する '
            filename = filename.replace('/', '_').replace(':', '_').replace(' ', '_').replace('.', '_').replace('"', '')
            ' .npyをloadして結合するとき、並びが変わらぬ様に昇順ソートしておく '
            #  result = result[[unique_id, feature]].sort_values(by=unique_id)
            #  result.reset_index(drop=True, inplace=True)

            if logger:
                logger.info(result[feature].value_counts())

            if len(select_list)==0:
                np.save(f'{path}{filename}', result[feature].values)

            else:
                if filename in select_list:
                    np.save(f'{path}{filename}', result[feature].values)


def make_feature_set(base, path, use_feature=[]):

    '''
    Explain:
        pathに入ってるfeatureをdatasetにmerge.現状はnpy対応
    Args:
    Return:
    '''

    if len(use_feature)==0:
        use_feature = glob.glob(path)
    p_list = pararell_load_data(use_feature)
    feature_set = pd.concat(p_list, axis=1)
    df = pd.concat([base, feature_set], axis=1)
    return df


def make_raw_feature(data, prefix='', select_list=[], ignore_list=[], extension='pkl', path='../features/1_first_valid', word='', logger=False):

    for tmp_col in data.columns:
        if tmp_col in ignore_list: continue
        if len(select_list)>0:
            if f'{prefix}{tmp_col}' not in select_list:continue
        if len(word)>0:
            if not(tmp_col.count(word)): continue

        new_col = tmp_col.replace('/', '_').replace(':', '_').replace(' ', '_').replace('.', '_').replace('"', '')
        data.rename(columns={tmp_col:new_col}, inplace=True)

        if extension.count('npy'):
            np.save(f'{path}/{prefix}{new_col}.npy', data[new_col].values)
        elif extension.count('csv'):
            data[new_col].to_csv(f'{path}/{prefix}{new_col}.csv')
        elif extension.count('pkl'):
            utils.to_pkl_gzip(path=f'{path}/{prefix}{new_col}.fp', obj=data[new_col].values)

