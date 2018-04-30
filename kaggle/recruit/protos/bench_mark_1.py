import numpy as np
import pandas as pd
from lgbm_reg import validation, prediction
from incremental_train import exploratory_train
import datetime
from datetime import date, timedelta
import glob, sys
from recruit_kaggle_load import RMSLE, load_data, load_submit
from sklearn.preprocessing import LabelEncoder

start_time = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())
sys.path.append('../../../module/')
from logger import logger_func
logger = logger_func()


""" データセット関連 """
input_path = '../input/*.csv'
submit_path = '../input/sample_submission.csv'
path_list = glob.glob(input_path)
key_list = [ 'air_reserve', 'air_store', 'air_visit', 'air_calendar' ]

'''データロード'''
air_vi, air_re, air_st, air_cal = load_data(key_list, path_list)

categorical_feature = [
    'air_store_id'
    ,'visit_date'
    ,'air_genre_name'
    ,'air_area_name'
    ,'day_of_week'
]
target = 'visitors'
partision = 'air_store_id'


def bench_mark(partision):

    b_mark = air_cal.merge(air_vi, on=['air_store_id', 'visit_date', 'dow'], how='left').merge(air_st, on='air_store_id', how='left')
    b_mark.fillna(0, inplace=True)

    start_date = pd.to_datetime('2017-03-12')
    end_date   = pd.to_datetime('2017-04-22')

    b_mark = b_mark[b_mark['visit_date'] <= end_date]
    b_mark['validation'] = b_mark['visit_date'].map(lambda x: 1 if start_date <= x and x <= end_date else 0)

    return b_mark


def make_submission():
    submit = load_submit(submit_path)

    submit = submit.merge(air_st, on='air_store_id', how='left').merge(air_cal, on=['air_store_id', 'visit_date'], how='left')

    lbl = LabelEncoder()
    for cat in categorical_feature:
        submit[cat] = lbl.fit_transform(submit[cat])

    return submit


'''Feature Engineeringを行う際のベースとなるデータセット'''
def make_air_base(data):
    data['visit_date_dow'] = data['visit_date'].astype('str') + '-' + data['day_of_week'].str[:3]
    data.to_csv('../input/{}_air_base.csv'.format(start_time[:8]), index=False)
    sys.exit()


def main():

    b_mark = bench_mark(partision)
    #  make_air_base(b_mark)

    eda_data = exploratory_train(b_mark, target, categorical_feature, 1, 0, partision, 1)
    eda_data.to_csv('../eda/{}_eda_prediction.csv'.format(start_time[:11]), index=False)
    sys.exit()

    submit = make_submission()
    b_mark.drop('validation', axis=1, inplace=True)

    prediction(b_mark, categorical_feature, submit, target)


if __name__ == '__main__':
    main()
