import numpy as np
import pandas as pd
import sys
sys.path.append('../../../module/')
from preprocessing import date_diff


def moving_agg(method, data, level, index, value, window, periods, sort_col=None):

    '''
    Explain:
        移動平均を求める。リーケージに注意
    Args:
        method(sum|avg):
        data(DF)        : 入力データ
        level(list)     : 集計を行う粒度。最終的に欠損値補完を行う粒度が1カラム
                          でなく、複数カラム必要な時はリストで渡す。
                          0からindexの数分を集計粒度とする(順番注意)
        index(int)      : 集計する粒度のカラム数
        value(float)    : 集計する値
        window(int)     : 移動平均して集計する行の数(上の行に遡る数)
        periods(int)    : 集計時に最小いくつの値を必要とするか
        sort_col(column): 移動平均時にソートが必要な場合のカラム。
                          これを入力した場合、戻り値Seriesのindexになる
    Returns:
        result(DF)  : 移動平均による集計結果。groupby as_index=False
    '''

    ' 集計時に日付は邪魔なのでindexに入れてからソートする '
    if not(sort_col is None):
        data = data.set_index(sort_col)
        data = data.sort_index()

    level = level[:index]

    if method=='avg':
        result = data.groupby(level)[value].rolling(
            window=window, min_periods=periods).mean().reset_index()
    elif method=='sum':
        result = data.groupby(level)[value].rolling(
            window=window, min_periods=periods).sum().reset_index()

    result.rename(columns={value: f'{value}_mv_{method}_w{window}_p{periods}@{level}'}, inplace=True)

    return result


def exp_weight_avg(data, level, value, weight, label):

    '''
    Explain:
        重み付き平均。
    Args:
        data(DF)    : 入力データ
        level       : 集計する粒度
        value(float): 集計する値
        weight(int) : 重み付き平均の減衰率
    Return:
        result(Series): 重み付き平均の集計結果。返す行は各粒度の最下行になる
    '''

    N = len(data)
    max_label = data[label].max()

    ' 指数重み付き平均なので、各行の重みを何乗にするか '
    ' labelが日付の場合はdate_diff。そうでない場合はそのとき作る '
    data['diff'] = abs(date_diff(max_label, data[label]))
    data['weight'] = data['diff'].map(lambda x: weight ** x.days)

    ' 各行について、値へ重みをかける '
    data['tmp'] = data['weight'] * data[value]

    '''
    valueがNullになっている行の重みは0にする。（分母の合計に入ってしまう為） 
    重みをかけた行の値はNullになっているが、重みの値はNullになっていない
    '''
    no_null = data[data[value].notnull()]
    null = data[data[value].isnull()]
    if len(null)>0:
        null['weight'] = 0
    if len(null)>0 and len(no_null)>0:
        data = pd.concat([null, no_null], axis=0)
    elif len(null)==0 and len(no_null)>0:
        data = no_null
    elif len(null)>0 and len(no_null)==0:
        data = null

    ' 分子、分母それぞれ合計をとって割る '
    tmp_result = data.groupby(level)['tmp', 'weight'].sum()
    result = tmp_result['tmp']/tmp_result['weight']

    result.name = f'{value}_wg{weight}_avg@{level}'

    return result


