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

first_date = air_vi['visit_date'].min()
last_date = air_vi['visit_date'].max()

''' カレンダーの日付を絞る '''
#  air_cal = air_cal[air_cal['visit_date'] <= last_date]


def main():
    print()


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
