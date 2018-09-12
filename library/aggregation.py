import numpy as np
import pandas as pd
import datetime
import glob
import sys
import re
from multiprocessing import Pool
import multiprocessing
from itertools import combinations

import os
HOME = os.path.expanduser('~')
sys.path.append(f"{HOME}/kaggle/github/library/")
import utils
from utils import logger_func, get_categorical_features, get_numeric_features, pararell_process
logger = logger_func()
pd.set_option('max_columns', 200)
pd.set_option('max_rows', 200)
from preprocessing import set_validation, split_dataset, get_dummies, factorize_categoricals
from feature_engineering import base_aggregation, diff_feature, division_feature, product_feature, cnt_encoding, select_category_value_agg
from make_file import make_npy, make_feature_set


start_time = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())

# ===========================================================================
# global variables
# ===========================================================================
dir = "../features/1_first_valid"
key = 'SK_ID_CURR'
target = 'TARGET'
ignore_list = [key, target, 'SK_ID_BUREAU', 'SK_ID_PREV']
prefix = 'nest_'

# ===========================================================================
# DATA LOAD
# ===========================================================================
#  base = pd.read_csv('../data/base.csv')[key].to_frame()
base = pd.read_csv( '../../../kaggle/kaggle_private/home-credit-default-risk/data/base.csv')
df = utils.read_df_pickle(path='')

# ===========================================================================
# 集計方法を選択
# ===========================================================================
agg_code = [ 'base', 'caliculate', 'cnt', 'category', 'combi', 'dummie' ][0]
diff = [True, False][0]
div = [True, False][0]
pro = [True, False][0]
method_list = ['sum', 'mean', 'var', 'max', 'min']


def main():

    '''
    BASE AGGRIGATION
    単一カラムをlevelで粒度指定して基礎集計
    '''
    if agg_code == 'base':

        # =======================================================================
        # 集計するカラムリストを用意
        # =======================================================================
        num_list = get_numeric_features(df=df, ignore=ignore_list)

        # =======================================================================
        # 集計開始
        # =======================================================================
        for num in num_list:
            for method in method_list:
                arg_list.append(df, key, num, method, prefix, '', base)
        ' データセットにおけるカテゴリカラムのvalue毎にエンコーディングする '
        call_list = pararell_process(pararell_wrapper(base_aggregation), arg_list)
        result = pd.concat(call_list, axis=1)

        for col in result.columns:
            if not(col.count('@')) or col in ignore_list:
                continue
            print(col)
            #  utils.to_pickle(path=f"{dir}/{col}.fp", obj=result[col].values)
        sys.exit()


        #  for num in num_list:
        #      for method in method_list:
        #          tmp_result = base_aggregation(df=df, level=key, method=method, prefix=prefix, feature=num, drop=True)
        #          result = base.merge(tmp_result, on=key, how='left')
        #          for col in result.columns:
        #              if not(col.count('@')) or col in ignore_list:
        #                  continue
        #              utils.to_pickle(
        #                  path=f"{dir}/{col}.fp", obj=result[col].values)
                #  make_npy(result=result, ignore_list=ignore_features, logger=logger)

    elif agg_code == 'caliculate':

        '''
        CALICULATION
        複数カラムを四則演算し新たな特徴を作成する
        '''
        f1_list = []
        f2_list = []
        used_lsit = []
        for f1 in f1_list:
            for f2 in f2_list:
                ' 同じ組み合わせの特徴を計算しない '
                if f1 == f2:
                    continue
                if sorted([f1, f2]) in used_list:
                    continue
                used_list.append(sorted([f1, f2]))

                if diff:
                    df = diff_feature(df=df, first=f1, second=f2)
                elif div:
                    df = division_feature(df=df, first=f1, second=f2)
                elif pro:
                    df = product_feature(df=df, first=f1, second=f2)

        for col in df.columns:
            utils.to_pickle(path=f"{dir}/{col}.fp", obj=df[col].values)

    elif agg_code == 'cnt':
        '''
        COUNT ENCODING
        level粒度で集計し、cnt_valを重複有りでカウント
        '''
        cat_list = get_categorical_features(df=df, ignore=ignore_list)

        for category_col in cat_list:
            df = cnt_encoding(df, category_col, ignore_list)
        df = base.merge(df, on=key, how='inner')
        cnt_cols = [col for col in df.columns inf col.count('cntec')]
        for col in cnt_cols:
            utils.to_pickle(path=f"{dir}/{col}.fp", obj=df[col].values)

    elif agg_code == 'category':
        arg_list = []
        ' カテゴリカラム '
        cat_list = get_categorical_features(df=df, ignore=ignore_list)
        num_list = get_numeric_features(df=df, ignore=ignore_list)

        for cat in cat_list:
            for value in num_list:
                for method in method_list:
                    arg_list.append(base, df, key, cat, value,
                                    method, ignore_list, prefix)

        ' データセットにおけるカテゴリカラムのvalue毎にエンコーディングする '
        pararell_process(pararell_wrapper(select_category_value_agg), arg_list)
        #  select_category_value_agg(base, df=df, key=key, category_col=cat, value=value, method, ignore_list, prefix)

    elif agg_code == 'combi':
        combi_num = [1, 2, 3][0]
        cat_combi = list(combinations(categorical, combi_num))

    elif agg_code == 'dummie':

        ' データセットのカテゴリカラムをOneHotエンコーディングし、その平均をとる '
        cat_list = get_categorical_features(data, ignore_features)
        df = get_dummies(df=df, cat_list=cat_list)


def pararell_wrapper(func, args):
    return func(*args)


if __name__ == '__main__':

    main()
