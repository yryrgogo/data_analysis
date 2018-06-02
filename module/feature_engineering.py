import numpy as np
import pandas as pd
import datetime
import sys
import glob

from preprocessing import set_validation
from convinience import col_part_shape_cnt_check, move_feature, shape_check_move


def make_npy(result, ignore_features, prefix, suffix=''):
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
        if feature.count('@'):
            filename = f'{prefix}{feature}'
            ' 環境パスと相性の悪い記号は置換する '
            filename = filename.replace(
                '/', '_').replace(':', '_').replace(' ', '_').replace('.', '_')
            ' .npyをloadして結合するとき、並びが変わらぬ様に昇順ソートしておく '
            #  result = result[[unique_id, feature]].sort_values(by=unique_id)
            #  result.reset_index(drop=True, inplace=True)

            print(result.shape)
            np.save(
            f'../features/1_first_valid/{filename}', result[feature].values)

