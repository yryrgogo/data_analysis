import numpy as np
import pandas as pd
from lgbm_reg import validation, prediction
from incremental_train import exploratory_train, incremental_train
import datetime
from datetime import date, timedelta
import glob, sys, re
from recruit_kaggle_load import RMSLE, load_data, load_submit, make_air_calendar
from sklearn.preprocessing import LabelEncoder
from itertools import combinations

start_time = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())
sys.path.append('../../../module/')
from load_data import pararell_read_csv
from logger import logger_func
logger = logger_func()

print(1)
sys.exit()

""" データセット関連 """
input_path = '../input/*.csv'
submit_path = '../input/sample_submission.csv'
path_list = glob.glob(input_path)
key_list = [ 'air_reserve', 'air_store', 'air_visit', 'air_cal_june_extract' ]

'''データロード'''
air_vi, air_re, air_st, air_cal = load_data(key_list, path_list)

' データセットからそのまま使用する特徴量 '
drop_list = ['dow', 'longitude', 'latitude', 'holiday_flg']
categorical_feature = [
    'air_store_id'
    #  ,'visit_date'
    ,'air_genre_name'
    ,'air_area_name'
    ,'day_of_week'
]
target = 'visitors'
partision = 'air_store_id'


def make_base(partision):

    start_date = pd.to_datetime('2017-03-12')
    #  end_date   = pd.to_datetime('2017-03-12')
    end_date   = pd.to_datetime('2017-04-22')

    base = air_cal[air_cal['visit_date'] <= end_date]
    base = base.merge(air_st[['air_store_id', 'air_genre_name', 'air_area_name']], on='air_store_id', how='left')
    base = base.merge(air_vi[['air_store_id', 'visit_date', 'visitors']], on=['air_store_id', 'visit_date'], how='left')
    base['validation'] = base['visit_date'].map(lambda x: 1 if start_date <= x and x <= end_date else 0)

    f_set = load_feature()
    base = base.merge(f_set, on=['air_store_id', 'air_genre_name', 'air_area_name', 'visit_date'], how='left')
    base.fillna(0, inplace=True)

    base.drop(drop_list, axis=1, inplace=True)

    return base


def load_feature():

    f_set = air_vi[['air_store_id', 'visit_date']].merge(air_st, on='air_store_id', how='left')
    feature_path = '../feature/use_feature/*.csv'
    p_list = pararell_read_csv(feature_path)

    for df in p_list:
        df['visit_date'] = pd.to_datetime(df['visit_date'])
        particle = [col for col in df.columns if (not(col.count('@')) and col.count('air'))][0]
        f_set = f_set.merge(df, on=[particle, 'visit_date'], how='left')

    return f_set


def make_submission():
    submit = load_submit(submit_path)

    submit = submit.merge(air_st, on='air_store_id', how='left').merge(air_cal, on=['air_store_id', 'visit_date'], how='left')

    lbl = LabelEncoder()
    for cat in categorical_feature:
        submit[cat] = lbl.fit_transform(submit[cat])

    return submit


def ready_incremental_train(b_mark):

    valid_list = []
    feature_list = []
    feature_path = glob.glob('../feature/valid_feature/*.csv')

    for i in range(1, len(feature_path), 1):
        tmp_tuple = combinations(feature_path, i)
        for elem in tmp_tuple:
            valid_list.append(elem)

    return valid_list


def valid_feature(dataset):

    feature_path = glob.glob('../feature/valid_feature/*.csv')
    valid_list = []
    feature_list = []
    for path in feature_path:
        logger.info(f'valid feature :{path}')
        feature = pd.read_csv(path)
        feature['visit_date'] = pd.to_datetime(feature['visit_date'])
        valid_set = dataset.merge(feature, on=['air_store_id', 'visit_date'], how='inner')
        exploratory_train(valid_set, target, categorical_feature, 1, 0, partision, 1, 0)

    sys.exit()


def main():

    base = make_base(partision)

    dataset = base

    ' 特徴量の全組み合わせを検証する '
    valid_list = ready_incremental_train(dataset)
    incremental_train(dataset, target, categorical_feature, valid_list, 1)
    sys.exit()

    ' 特徴量を一つずつ検証する '
    #  valid_feature(dataset)

    eda_data = exploratory_train(dataset, target, categorical_feature, 1, 0, partision, 1)
    #  eda_data.to_csv('../eda/{}_eda_prediction.csv'.format(start_time[:11]), index=False)

    #  submit = make_submission()
    #  b_mark.drop('validation', axis=1, inplace=True)

    #  prediction(b_mark, categorical_feature, submit, target)


if __name__ == '__main__':
    main()
