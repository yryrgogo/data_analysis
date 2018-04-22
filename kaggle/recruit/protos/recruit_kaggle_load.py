import numpy as np
import pandas as pd
import datetime
import sys
import glob
from sklearn.metrics import mean_squared_log_error
sys.path.append('../../../module')
from load_data import pararell_load_data

start_time = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())

"""Validationは3/12(日)~4/22(土)"""

""" データセット関連 """
input_path = '../input/*.csv'
submit_path = '../input/sample_submission.csv'
path_list = glob.glob(input_path)
key_list = [ 'air_reserve', 'air_store', 'air_visit', 'date_info' ]

""" submit関連
                                id  visitors
0  air_00a91d42b08b08d9_2017-04-23         0
(32019, 2)
"""

""" 評価関数 """
def RMSLE(y_obs, y_pred):
    #  del_idx = np.arange(len(y_obs))[y_obs == 0]
    #  y_obs = np.delete(y_obs, del_idx)
    #  y_pred = np.delete(y_pred, del_idx)
    y_pred = y_pred.clip(min=0.)
    return np.sqrt(mean_squared_log_error(y_obs, y_pred))


def make_submission(dataset):
    print()


def load_data(key_list, path_list):

    """並列でデータセットをロード"""
    p_list = pararell_load_data(key_list, path_list)

    for d_dict in p_list:
        for key, df in d_dict.items():
            if key.count('air_reserve'):
                air_re = df
            elif key.count('air_store'):
                air_st = df
            elif key.count('air_visit'):
                air_vi = df
            elif key.count('date_info'):
                df_date = df.rename(columns = {'calendar_date':'visit_date'})

    air_vi['visit_date'] = pd.to_datetime(air_vi['visit_date'])
    air_vi['dow'] = air_vi['visit_date'].dt.dayofweek

    air_re['visit_date'] = pd.to_datetime(air_re['visit_datetime'].str[:10])
    air_re['reserve_date'] = pd.to_datetime(air_re['reserve_datetime'].str[:10])
    air_re['dow'] = air_re['visit_date'].dt.dayofweek

    df_date['visit_date'] = pd.to_datetime(df_date['visit_date'])
    df_date['dow'] = df_date['visit_date'].dt.dayofweek
    df_date['day_of_week'] = df_date.apply(lambda x:'Special' if x.holiday_flg==1 and (x.day_of_week != 'Saturday' or x.day_of_week != 'Sunday') else x.day_of_week, axis=1)
    df_date['holiday_flg'] = df_date.apply(lambda x:1 if x.holiday_flg==1 or x.day_of_week=='Saturday' or x.day_of_week=='Sunday' else 0, axis=1)

    return air_vi, air_re, air_st, df_date


def load_submit(submit_path):
    submit = pd.read_csv(submit_path)
    submit['air_store_id'] = submit['id'].str[:-11]
    submit['visit_date'] = pd.to_datetime(submit['id'].str[-10:])

    return submit

