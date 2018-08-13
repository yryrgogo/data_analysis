import numpy as np
import pandas as pd
import sys, glob
from load_data import pararell_load_data


def make_npy(result, ignore_list=[], prefix='', suffix='', select_list=[], path='../features/1_first_valid/', logger=False):
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
        if feature.count('@') and feature not in ignore_list:
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


def make_feature_set(dataset, path):

    '''
    Explain:
        pathに入ってるfeatureをdatasetにmerge.現状はnpy対応
    Args:
    Return:
    '''

    use_feature = glob.glob(path)
    p_list = pararell_load_data(use_feature, 0)
    df_use = pd.concat(p_list, axis=1)

    dataset = pd.concat([dataset, df_use], axis=1)
    return dataset


def make_raw_feature(data, prefix='', select_list=[], ignore_list=[], extension='npy', path='../features/raw_features/', word=''):

    for col in data.columns:
        if col in ignore_list: continue
        if len(select_list)>0:
            if f'{prefix}{col}' not in select_list:continue
        if len(word)>0:
            if not(col.count(word)): continue

        new_col = col.replace('/', '_').replace(':', '_').replace(' ', '_').replace('.', '_').replace('"', '')
        data.rename(columns={col:new_col}, inplace=True)
        if extension.count('npy'):
            np.save(f'{path}{prefix}{new_col}.npy', data[new_col].values)
        elif extension.count('csv'):
            data[new_col].to_csv(f'{path}{prefix}{new_col}.csv')

