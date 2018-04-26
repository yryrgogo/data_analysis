import numpy as np
import pandas as pd
from itertools import combinations
import datetime
from time import time
import sys
import glob
import re
from tqdm import tqdm
from multiprocessing import Pool
import multiprocessing
sys.path.append('../../../module')
from recruit_kaggle_load import load_data, set_validation, date_diff, load_submit

start_time = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())

""" データセット関連 """
input_path = '../input/*.csv'
submit_path = '../input/sample_submission.csv'
path_list = glob.glob(input_path)
key_list = ['air_reserve', 'air_store', 'air_visit', 'air_calendar']

""" データロード """
#  base = pd.read_csv('../input/20180424_air_base.csv')
air_vi, air_re, air_st, air_cal = load_data(key_list, path_list)
data_list = [air_vi, air_re, air_st, air_cal]

""" データチェック"""
#  for d in data_list:
#      print(d.head())
#  sys.exit()

""" 前処理 """
'''validation_noをセット'''
air_vi = set_validation(air_vi)
'''前日のvisitorsをセット'''
air_vi['last_visit'] = air_vi.groupby('air_store_id')['visitors'].shift(1)

''' moving average '''
window_list = [7, 21, 35, 63, 126, 252, 378]
''' weight average '''
weight_list = [0.9, 0.95, 0.98, 0.99]
first_date = air_vi['visit_date'].min()
last_date = air_vi['visit_date'].max()

air_cal = air_cal[air_cal['visit_date'] <= last_date]

''' 学習データの日程（重み付き平均は1日ずつ計算してあげる必要があるので全日付リストが必要） '''
date_list = air_vi['visit_date'].drop_duplicates(
).sort_values().values[100:130]


def moving_avg(data, particle, value, window, periods):

    data = data.set_index('visit_date')
    data = data.sort_index()
    result = data.groupby(particle)[value].rolling(
        window=window, min_periods=periods).mean().reset_index()

    result.rename(columns={value: '{}_@mv_avg_w{}_p{}'.format(value,
        window, periods)}, inplace=True)

    return result


def exp_weight_avg(data, particle, value, weight):

    N = len(data)
    max_date = data['visit_date'].max()

    data['diff'] = abs(date_diff(max_date, data['visit_date']))
    data['weight'] = data['diff'].map(lambda x: weight ** x.days)

    data['{}_@w_avg_{}'.format(value, weight)
         ] = data['weight'] * data['visitors']

    tmp_result = data.groupby(
        particle)['{}_@w_avg_{}'.format(value, weight), 'weight'].sum()
    result = tmp_result['{}_@w_avg_{}'.format(
        value, weight)]/tmp_result['weight']


def date_range(data, start, end):
    return data[(start <= data['visit_date']) & (data['visit_date'] <= end)]


def main():
    """ 移動平均 """
    #  for window in window_list:
    #      mv_avg = moving_avg(air_vi, 'air_store_id', 'last_visit', window, 1)

    """ 重み付き平均 """
    #  for weight in weight_list:
    #      for end_date in date_list:
    #          tmp = date_range(air_vi, first_date, end_date)
    #          exp_weight_avg(tmp, 'air_store_id', 'last_visit', weight)

    """ 曜日&祝日別の集計 """
    vi_date = air_cal.merge(air_vi, on=['air_store_id', 'visit_date', 'dow'], how='left')
    ''' 祝日以外の曜日集計を行う '''
    for i in range(7):
        tmp = vi_date[vi_date['dow']==i][['air_store_id', 'visit_date', 'day_of_week', 'dow', 'visitors']]
        tmp_date = tmp[['air_store_id', 'visit_date', 'day_of_week', 'dow']]
        ''' 祝日はNULLにする '''
        tmp = tmp[tmp['day_of_week'] != 'Special']
        data = tmp_date.merge(tmp, on=['air_store_id', 'visit_date', 'day_of_week', 'dow'], how='left', copy=False)
        ''' 一週前の数字を持たせる '''
        data['lastweek_visitors'] = data['visitors'].shift(1)

        for window in range(1,5,1):
            dow_avg = moving_avg(data, 'air_store_id', 'lastweek_visitors', window, 1)

    ''' 祝日の集計を行う '''
    ' 連休フラグが立っている祝日の集計 '

    ' 連休フラグが立っていない祝日の集計 '

if __name__ == '__main__':

    main()


def one_agg_wrapper(args):
    return one_particle_3_value_agg(*args)


def pararell_process(func, arg_list):

    p = Pool(multiprocessing.cpu_count())
    p_list = p.map(func, arg_list)
    p.close


def make_arg_list(particle_list, t=0, cnt_flg=0):
    return arg_list
