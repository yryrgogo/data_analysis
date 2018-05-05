import pandas as pd
import numpy as np
import sys, re
import gc


def outlier(data, particle, value, out_range=1.64):
    '''
    Explain:
    Args:
        data(DF)        : 外れ値を除外したいデータフレーム
        particle(str)   : 標準偏差を計算する粒度
        value(float)    : 標準偏差を計算する値
        out_range(float): 外れ値とするZ値の範囲.初期値は1.64としている
    Return:
        data(DF): 入力データフレームから外れ値を外したもの
    '''
    tmp = data.groupby(particle, as_index=False)[value].agg(
        {'avg':'mean',
         'std':'std'
         })
    df = data.merge(tmp, on=particle, how='inner')
    param = df[value].values
    avg = df['avg'].values
    std = df['std'].values
    z_value =(param - avg)/std
    df['z'] = z_value

    inner = df[-1*out_range <= df['z']].copy()
    inner = inner[inner['z'] <= out_range]

    out_minus = df[-1*out_range > df['z']]
    minus_max = out_minus.groupby(particle, as_index=False)[value].max()
    out_minus.drop(value, axis=1, inplace=True)
    out_minus = out_minus.merge(minus_max, on=particle, how='inner')

    out_plus = df[df['z'] > out_range]
    plus_min = out_plus.groupby(particle, as_index=False)[value].min()
    out_plus.drop(value, axis=1, inplace=True)
    out_plus = out_plus.merge(plus_min, on=particle, how='inner')

    result = pd.concat([inner, out_minus, out_plus], axis=0)
    result.drop(['avg', 'std', 'z'], axis=1, inplace=True)
    result.reset_index(drop=True, inplace=True)

    del df, out_minus, out_plus
    gc.collect()

    return result
