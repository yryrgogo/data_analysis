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
from logger import logger_func
logger = logger_func()
pd.set_option('max_columns', 200)
pd.set_option('max_rows', 200)
from preprocessing import set_validation, split_dataset, get_dummies, factorize_categoricals
from convinience_function import get_categorical_features, get_numeric_features
from load_data import pararell_load_data
from feature_engineering import base_aggregation
from make_file import make_npy, make_feature_set


start_time = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())

method_list = ['sum', 'mean', 'std', 'max', 'min']
key = 'SK_ID_CURR'
target = 'TARGET'
ignore_list = [key, target, 'valid_no', 'valid_no_4', 'is_train', 'is_test']


def main():

    #==============================================================================
    # VARIABLES
    #==============================================================================
    prefix = 'nest_'

    #==============================================================================
    # DATA LOAD
    #==============================================================================
    base = pd.read_csv('../data/base.csv')[key].to_frame()
    data = []

    agg_code = [
        'base'
        ,'cnt'
        ,'cnt'
    ]

    if agg_code=='base':

        ' BASE AGGRIGATION '
        num_list = get_numeric_features(data=data, ignore=ignore_list)
        for num in num_list:
            for method in method_list:
                tmp_result = base_aggregation(data=data, level=key, method=method, prefix=prefix, feature=num)
                result = base.merge(tmp_result, on=key, how='left')
                for col in result.columns:
                    if not(col.count('@')) or col in ignore_list:continue
                    utils.to_pickle(path=f"../features/1_first_valid/{col}.fp", obj=result[col])
                #  make_npy(result=result, ignore_list=ignore_features, logger=logger)

    elif agg_code==''
        ' カテゴリの組み合わせをエンコーディングする場合はこちら '
        combi_num = 1
        #  combi_num = 2
        #  combi_num = 3
        cat_combi = list(combinations(categorical, combi_num))

        for level in cat_combi:
            ' 好きな粒度をlevelに入力してエンコーディングする '
            make_select_level_agg(data, level, method_list, ignore_features, prefix)

    ' データセットにおけるカテゴリカラムのvalue毎にエンコーディングする '
    select_category_value_agg(base, data, level, cat_list, num_list, method_list, ignore_features, prefix)
    sys.exit()


    ' データセットのカテゴリカラムをOneHotエンコーディングし、その平均をとる '
    #  dummie_avg(data, level, ignore_features, prefix)

    cnt_col_list = ['ORGANIZATION_TYPE', 'OCCUPATION_TYPE']
    ' カウントエンコーディング。level粒度で集計し、cnt_valを重複有りでカウント '
    cnt_encoding(base, data, level, cnt_col_list, ignore_features, prefix)


if __name__ == '__main__':

    main()
