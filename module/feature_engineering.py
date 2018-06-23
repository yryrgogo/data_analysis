import numpy as np
import pandas as pd
import sys


def base_aggregation(data, level, feature, method, prefix='', suffix=''):
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

    ' levelの型がlistでもtupleでもなければ単体カラムのはずなので、listにする '
    if not(str(type(level)).count('tuple')) and not(str(type(level)).count('list')):
        level = [level]
    elif str(type(level)).count('tuple'):
        level = list(level)

    df = data[level+[feature]]

    result = df.groupby(level)[feature].agg({'tmp': {method}})
    result = result['tmp'].reset_index().rename(
        columns={f'{method}': f'{prefix}{feature}_{method}{suffix}@{level}'})

    return result

def diff_feature(df, first, second):
    ' 大きい値がf1に来るようにする '
    if first<second:
        f1 = second
        f2 = first
    else:
        f1 = first
        f2 = second
    print(f'f1: {f1}')
    print(f'f2: {f2}')
    df[f'{f1}_diff_{f2}@'] = df[f1] - df[f2]
    return df

def division_feature(df, first, second):
    ' 大きい値がf1に来るようにする '
    if first<second:
        f1 = second
        f2 = first
    else:
        f1 = first
        f2 = second
    print(f'f1: {f1}')
    print(f'f2: {f2}')
    df[f'{f1}_div_{f2}@'] = df[f1] / df[f2]
    return df

def product_feature(df, first, second):
    ' 大きい値がf1に来るようにする '
    if first<second:
        f1 = second
        f2 = first
    else:
        f1 = first
        f2 = second
    print(f'f1: {f1}')
    print(f'f2: {f2}')
    df[f'{f1}_pro_{f2}@'] = df[f1] * df[f2]
    return df
