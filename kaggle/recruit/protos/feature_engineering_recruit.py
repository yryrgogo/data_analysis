import numpy as np
import pandas as pd
from itertools import combinations
import datetime
from time import time
import sys, glob, re
from tqdm import tqdm
from multiprocessing import Pool
import multiprocessing
sys.path.append('../../../module')
from recruit_kaggle_load import load_data, set_validation, date_diff

start_time = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())

""" データセット関連 """
input_path = '../input/*.csv'
submit_path = '../input/sample_submission.csv'
path_list = glob.glob(input_path)
key_list = [ 'air_reserve', 'air_store', 'air_visit', 'date_info' ]

""" データロード """
#  base = pd.read_csv('../input/20180424_air_base.csv')
air_vi, air_re, air_st, df_date = load_data(key_list, path_list)
data_list = [air_vi, air_re, air_st, df_date]

""" データチェック"""
#  for d in data_list:
#      print(d.head())
#  sys.exit()

'''validation_noをセット'''
air_vi = set_validation(air_vi)
'''前日のvisitorsをセット'''
air_vi['last_visit'] = air_vi.groupby('air_store_id')['visitors'].shift(1)
''' moving average '''

window_list = [7, 21, 35, 63, 126, 252, 378]


'''
各行に対して重み付き平均を持たせるには？
一度の計算では、maxの日付に対する重み付き平均になるので、
1日ずつしか出せない。全ての行に対してweight_avgを持たせるには、
持たせる日付を決めて、その全ての日付について計算をしていかなければならない
'''

def exp_weight_avg(data, start, end, weight):


    N = len(data)
    max_date = data['visit_date'].max()

    data['diff'] = abs(date_diff(max_date, data['visit_date']))
    data['weight'] = data['diff'].map(lambda x:weight ** x.days)

    data['visitors_@w_avg_{}'.format(weight)] = data['weight'] * data['visitors']


def date_range(data, start, end):
    data = data[( start <= data['visit_date']) & (data['visit_date'] <= end)]
    print(data.head())
    sys.exit()


date_list = air_vi['visit_date'].drop_duplicates().sort_values().values[7:]
print(date_list)
sys.exit()

for weight in [0.9, 0.95, 0.98, 0.99]:
    exp_weight_avg(air_vi, '2017-04-01', air_vi['visit_date'].max(), weight)
sys.exit()


def moving_avg(data, particle, value, window, periods, window_list):

    data.set_index('visit_date', inplace=True)
    data.sort_index(inplace=True)
    result = data.groupby(particle)[value].rolling(window=window, min_periods=periods).mean().reset_index()

    result.rename(columns={'last_visit':'visitors_@mv_avg_{}_{}'.format(window, periods)}, inplace=True)

    return result


def one_agg_wrapper(args):
    return one_particle_3_value_agg(*args)


def pararell_process(func, arg_list):

    p = Pool(multiprocessing.cpu_count())
    p_list = p.map(func, arg_list)
    p.close


def make_arg_list(particle_list, t=0, cnt_flg=0):
    return arg_list


def main():
    print()
    sys.exit()


if __name__ == '__main__':

    main()
