import numpy as np
import pandas as pd
import datetime
import sys
import glob

from preprocessing import set_validation
from convinience import col_part_shape_cnt_check, move_feature, shape_check_move


def make_npy(result, ignore_features=[], prefix='', suffix=''):
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
        if feature.count('@') and feature not in ignore_features:
            filename = f'{prefix}{feature}'
            ' 環境パスと相性の悪い記号は置換する '
            filename = filename.replace(
                '/', '_').replace(':', '_').replace(' ', '_').replace('.', '_')
            ' .npyをloadして結合するとき、並びが変わらぬ様に昇順ソートしておく '
            #  result = result[[unique_id, feature]].sort_values(by=unique_id)
            #  result.reset_index(drop=True, inplace=True)

            print(result.shape)
            np.save(f'../features/1_first_valid/{filename}', result[feature].values)

def base_aggregation(data, level, feature, method, prefix=''):
    '''
    Explain:
        levelの粒度で集約を行う。この関数が受け取るカラムは一つなので、
        複数カラムを集計する時はループで繰り返し引数を渡す
    Args:
        level: str/list/tuple。要素数は何個でもOK
    Return:
        集約したカラムとlevelの2カラムDF。
        集約したカラム名は、{prefix}{元のカラム名}_{メソッド}@{粒度}となっている
    '''

    if str(type(level)).count('str'):
        level = [level]
    elif str(type(level)).count('tuple'):
        level = list(level)

    df = data[level+[feature]]

    result = df.groupby(level)[feature].agg({'tmp': {method}})
    result = result['tmp'].reset_index().rename(
        columns={f'{method}': f'{prefix}{feature}_{method}@{level}'})

    return result


