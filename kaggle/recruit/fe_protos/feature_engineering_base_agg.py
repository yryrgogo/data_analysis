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
sys.path.append('../protos/')
from recruit_kaggle_load import recruit_load_data, set_validation, date_diff, load_submit
sys.path.append('../../../module/')
from preprocessing import outlier

start_time = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())

""" データセット関連 """
input_path = '../input/*.csv'
submit_path = '../input/sample_submission.csv'
path_list = glob.glob(input_path)
key_air_vi = 'air_vi_june_ex'
key_air_re = 'air_re'
key_air_cal = 'air_cal_june_ex'
key_area = 'air_area'
key_genre = 'air_genre'
key_list = [ key_air_vi, key_air_cal ,key_area, key_genre]

""" データロード """
data_dict = recruit_load_data(key_list, path_list)
air_vi = data_dict[key_air_vi]
air_cal = data_dict[key_air_cal]
air_area = data_dict[key_area]
air_genre = data_dict[key_genre]
#  air_re = data_dict[key_air_re]

first_date = air_vi['visit_date'].min()
last_date = air_vi['visit_date'].max()

''' カレンダーの日付を絞る '''
air_cal = air_cal[air_cal['visit_date'] <= last_date]


def date_range(data, start, end, include_flg=1):
    '''
    include_flgが0の場合, endの日付は含めずにデータを返す
    '''
    if include_flg == 0:
        return data[(start <= data['visit_date']) & (data['visit_date'] < end)]
    return data[(start <= data['visit_date']) & (data['visit_date'] <= end)]


def squeeze_target(data, particle, size):
    '''
    particleの各要素について、一定数以上のデータがある要素の行のみ残す
    '''

    tmp = data.groupby(particle).size()
    target_id = tmp[tmp >= size].index

    data = data.set_index(particle)
    result = data.loc[target_id, :]
    result = result.reset_index()

    return result


def data_check():
    ' 2つの条件で絞り可視化 '
    rule2 = air_cal[(air_cal['air_store_id'] == 'air_00a91d42b08b08d9') & (
        air_cal['day_of_week'] == 'Sunday')]


def base_agg(data, particle, value, method):
    '''
    Explain:
        Groupbyによる基礎集計を行う関数。粒度と値、集計メソッドを引数とし、結果を返す.
    Args:
        data(DF): 集計を行うデータ
        particle: 集計したい粒度。単変数でもリストでも可
        value(float): 集計したい値
        method(str): 集計関数（sum/mean/std/var/max/min/median）
    Return:
        result(DF): 集計結果。groupbyの粒度カラムはインデックスとせずに返す
    '''

    result = data.groupby(particle, as_index=False)[value].agg(
        {f'{value}_{method}@{particle}': f'{method}'})

    return result


def main():

    '''
    曜日あたりの最大値などを求めるといっても、各レコードに入れるデータが
    リークしてはならない。その為、日付は当日の値だが、各メソッドの結果は
    その前日以上前の値の集計である必要がある
    air_calにはlast_visitorsとして前日の訪問数を持たせてあるので、
    当日以降のデータを切って、last_visitorsを集計すれば問題ない
    '''
    method_list = ['max', 'min', 'std']
    #  particle_list = ['air_store_id', 'day_of_week']
    particle_list = ['air_genre_name', 'day_of_week']

    start = air_cal['visit_date'].values.min()
    # 7日目以降にしとく
    dow_list = air_cal['day_of_week'].drop_duplicates()

    data = air_genre

    for method in method_list:
        result = pd.DataFrame([])
        for dow in dow_list:
            tmp_dow = data[data['day_of_week']==dow]
            date_list = tmp_dow['visit_date'].drop_duplicates().sort_values().values[7:]
            for end in date_list:
                tmp = date_range(tmp_dow, start, end)
                tmp_result = base_agg(tmp, particle_list, 'last_dow_visitors', method)
                tmp_result['visit_date'] = end
                if len(result) ==0:
                    result = tmp_result
                else:
                    result = pd.concat([result, tmp_result], axis=0)
            print(result.shape)
        print(result.shape)
        print(result.tail())
        result.to_csv(f'../feature/valid_feature/last_dow_visitors_{method}@{particle_list}.csv')


if __name__ == '__main__':

    main()
