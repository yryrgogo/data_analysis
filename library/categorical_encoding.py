import numpy as np
import pandas as pd
import datetime
import glob
import sys
import re
from multiprocessing import Pool
import multiprocessing
from itertools import combinations

sys.path.append('../../../github/module/')
from preprocessing import set_validation, split_dataset, get_dummies, factorize_categoricals
from convinience_function import get_categorical_features, get_numeric_features
from load_data import pararell_load_data
from feature_engineering import base_aggregation
from make_file import make_npy, make_feature_set

start_time = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())

method_list = ['sum', 'mean', 'std', 'max', 'min']
key = 'SK_ID_CURR'
target = 'TARGET'
ignore_features = [key, target, 'valid_no', 'valid_no_4', 'is_train', 'is_test']


def cat_to_target_bin_enc(df, df_target, category, bins=10):
    '''
    Explain:
        dfでは全インデックスをもった元データをうけとり、そこに
        binにしたfeatureをカラムとして返す。df_targetは01の二値
        をもったデータで-1などでは計算が狂うので注意。
    Args:
        category(column): ターゲットエンコーディングのビンで集約したいカテゴリカラム
    Return:
        元データのカテゴリを集約したカラムに変換したデータ
    '''

    ' カテゴリ変数の種類が多い場合、ターゲットエンコーディングしてからbinにして数を絞る '
    target_avg = df_target.groupby(category)['TARGET'].mean().reset_index()

    ' positive率でカテゴリをビンにする '
    bin_list = ['TARGET']
    for col in bin_list:
        target_avg[target] = pd.qcut(
            x=target_avg[col], q=bins, duplicates='drop')
        target_avg.set_index(category, inplace=True)
        target_avg = target_avg.iloc[:, 0]
    df[category] = df[category].map(target_avg)

    ' positive率が低い順に番号をつける '
    bin_sort = df[category].drop_duplicates().sort_values().to_frame()
    bin_sort['index'] = np.arange(len(bin_sort))+1
    bin_sort = bin_sort.set_index(category).iloc[:, 0]
    df[category] = df[category].map(bin_sort)

    return df


def col_rename(data, level, ignore_features, prefix='', suffix=''):
    '''
    Explain:
        いちいちrenameを書くのが面倒な時に使う。
        {prefix}{元のカラム名}{suffix}@{粒度}のカラム名にrenameしたいとき便利
    Args:
        ignore_features(list): renameしたくないカラムリスト
    Return:
    '''
    for col in data.columns:
        if col not in ignore_features:
            data.rename(
                columns={col: f'{prefix}{col}{suffix}@{level}'}, inplace=True)
    return data


def select_category_value_agg(base, data, level, cat_list=[], num_list=[], method_list=[], ignore_list=[], prefix=''):
    '''
    Explain:
        likelifood encoding
        カテゴリカラムの各内容における行に絞った上でlevel粒度の集計を行い、特徴量を作成する。
        カテゴリカラムの内容を無視して集約するのではなく、カテゴリの内容毎に集計して、
        粒度に対する各カテゴリの統計量を横持ちさせる。
    Args:
    Return:
    '''

    ' dataそのものがbaseとなる場合 '
    if len(base) == 0:
        base = data

    if len(cat_list) == 0:
        cat_list = get_categorical_features(data, ignore_features)

    ' カテゴリカラム '
    for cat in cat_list:
        ' カテゴリカラムにNullがある場合はUnknownとして集計する '
        data[cat] = data[cat].fillna('Unknown')
        ' 集計するカテゴリカラムの中身 '
        for cat_val in data[cat].drop_duplicates().values:

            ' 集計するカテゴリに絞る '
            df_cat = data[data[cat] == cat_val]

            if len(num_list) == 0:
                num_list = get_numeric_features(data, ignore_features)

            ' 集計対象のカラム '
            for val in num_list:
                if df_cat[val].dtype != 'object':

                    for method in method_list:

                        logger.info(
                            f"\ncat: {cat}\ncat_val: {cat_val}\nval: {val}\nmethod: {method}")

                        result = base_aggregation(df_cat, level, val, method)
                        #  if len(result.dropna()) / len(result) < 0.01:
                        #      continue
                        result = base.merge(result, on=level, how='left')
                        ' どのカテゴリカラムでどのVALUEについて集計したか分かるように '
                        prename = f'{prefix}{cat}_{cat_val}_'

                        make_npy(result, ignore_features, prename, method, logger=logger)


def select_level_agg(base, data, level, method_list, ignore_list, num_list=[], prefix=''):
    '''
    Explain:
        likelifood encoding
        集計粒度をlevelで指定して集計を行い、特徴量を作成する。通常の集約関数。
        カテゴリ別に切り分けて集約したい場合は、make_select_category_agg()を使用する。
        .npyでの保存を想定している為、元ファイルのlevelカラムにleft joinして
        から保存する。取り出す時は、元ファイルのlevel粒度にjoinしないと、
        順番が保存されないので注意
    Args:
        level(str/list/tuple): 集計粒度。
        ignore_features(list): 集計を行わないカラムリスト
    Return:
    '''

    if str(type(level)).count('str'):
        level = [level]
    elif str(type(level)).count('tuple'):
        level = list(level)

    ' 集計対象のカラム。int or float '
    for num in num_list:

        ' objectだったら対象外 '
        if data[num].dtype != 'object':

            for method in method_list:

                logger.info(f"\nnum: {num}\nmethod: {method}\n{level}")

                result = base_aggregation(data, level, num, method)

                logger.info(
                    f'\nbase: {base.columns}\nresult: {result.columns}')

                result = base.merge(result, on=level, how='left')
                logger.info(f"\nresult shape: {result.shape}")

                make_npy(result, ignore_list, prefix, method, logger=logger)


def cnt_encoding(base, data, level, cat_list, ignore_list, prefix):
    '''
    Explain:
        対象featureのCOUNT DISTINCTをとり特徴量とする
    Args:
    Return:
    '''
    ' .npy形式の保存を想定。元データの行数とインデックスを保つ為のDF '

    if str(type(level)).count('str'):
        level = [level]
    elif str(type(level)).count('tuple'):
        level = list(level)

    for cnt_col in cat_list:

        ' level粒度でcnt_valをユニークカウント '
        cnt_enc = data[level + [cnt_col]
                       ].groupby(level, as_index=False).count()
        ' 特徴量のカラム名テンプレートにrename '
        cnt_enc = col_rename(cnt_enc, level, ignore_features, '', '_cnt')

        ' 元データの行数とインデックスを保つ為、Left Join '
        result = base.merge(cnt_enc, on=level, how='left')

        ' npyで保存 '
        make_npy(result, ignore_features, prefix, logger=logger)


def main():

    " データの読み込み "
    base = pd.read_csv('../data/base.csv')[key].to_frame()

    ' BASE AGGRIGATION '
    num_list = get_numeric_features(data=data, ignore=ignore_features)
    for num in num_list:
        for method in method_list:
            tmp_result = base_aggregation(data=data, level=key, method=method, prefix=prefix, feature=num)
            result = base.merge(tmp_result, on=key, how='left')
            make_npy(result=result, ignore_list=ignore_features, logger=logger)
    sys.exit()


    #  ' カテゴリの組み合わせをエンコーディングする場合はこちら '
    #  cat_combi = list(combinations(categorical, 2))
    #  categorical = cat_combi

    ' データセットにおけるカテゴリカラムのvalue毎にエンコーディングする '
    select_category_value_agg(base, data, level, cat_list, num_list, method_list, ignore_features, prefix)
    sys.exit()

    #  for level in categorical:
    #      ' 好きな粒度をlevelに入力してエンコーディングする '
    #      make_select_level_agg(data, level, method_list,
    #                            ignore_features, prefix)
    #  sys.exit()

    ' データセットのカテゴリカラムをOneHotエンコーディングし、その平均をとる '
    #  dummie_avg(data, level, ignore_features, prefix)

    cnt_col_list = ['ORGANIZATION_TYPE', 'OCCUPATION_TYPE']
    ' カウントエンコーディング。level粒度で集計し、cnt_valを重複有りでカウント '
    cnt_encoding(base, data, level, cnt_col_list, ignore_features, prefix)


if __name__ == '__main__':

    main()
