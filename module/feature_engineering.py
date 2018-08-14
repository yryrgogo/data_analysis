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
    if df[first].mean()<df[second].mean():
        f1 = second
        f2 = first
    else:
        f1 = first
        f2 = second
    df[f'{f1}_diff_{f2}@'] = df[f1] - df[f2]
    return df


def division_feature(df, first, second, sort=1):
    ' 大きい値がf1に来るようにする '
    if df[first].mean()<df[second].mean() and sort==1:
        f1 = second
        f2 = first
    else:
        f1 = first
        f2 = second
    df[f'{f1}_div_{f2}@'] = df[f1] / df[f2]
    return df


def product_feature(df, first, second):
    ' 大きい値がf1に来るようにする '
    if df[first].mean()<df[second].mean():
        f1 = second
        f2 = first
    else:
        f1 = first
        f2 = second
    df[f'{f1}_pro_{f2}@'] = df[f1] * df[f2]
    return df


def cat_to_target_bin_enc(df, category, bins=10, target='TARGET'):
    ' カテゴリ変数の種類が多い場合、ターゲットエンコーディングしてからbinにして数を絞る '
    target_avg = df.groupby(category)['TARGET'].mean().reset_index()

    ' positive率でカテゴリをビンにする '
    bin_list = ['TARGET']
    for col in bin_list:
        target_avg[target] = pd.qcut(x=target_avg[col], q=bins, duplicates='drop')
        target_avg.set_index(category, inplace=True)
        target_avg = target_avg.iloc[:, 0]
    df[category] = df[category].map(target_avg)

    ' positive率が低い順に番号をつける '
    bin_sort = df[category].drop_duplicates().sort_values().to_frame()
    bin_sort['index'] = np.arange(len(bin_sort))+1
    bin_sort = bin_sort.set_index(category).iloc[:, 0]
    df[category] = df[category].map(bin_sort)

    return df


