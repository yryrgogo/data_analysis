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
key_area = 'area_vi_date_june_ex'
key_list = [
    key_air_vi
    ,key_air_cal
    #  ,key_area
]

""" データロード """
data_dict = recruit_load_data(key_list, path_list)
air_vi = data_dict[key_air_vi]
air_cal = data_dict[key_air_cal]
#  air_area = data_dict[key_area]
#  air_re = data_dict[key_air_re]

first_date = air_vi['visit_date'].min()
last_date = air_vi['visit_date'].max()

''' カレンダーの日付を絞る '''
air_cal = air_cal[air_cal['visit_date'] <= last_date]


def date_range(data, start, end, include_flg=1):
    '''
    include_flgが0の場合, endの日付は含めずにデータを返す
    '''
    if include_flg==0:
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


def main():

    #  make_dow_rank(air_cal, 'air_store_id', 3, 1)



def make_dow_rank(data, particle, rank, ratio_flg=0):

    date_list= data['visit_date'].drop_duplicates().sort_values().values

    result = pd.DataFrame([])
    ' 8日目以降で集計する '
    for date in date_list[7:]:

        ' 現在の曜日 '
        dow = data[data['visit_date']==date]['day_of_week'].values[0]
        ' dateは含めない範囲のデータを返す '
        tmp = date_range(data, date_list.min(), date, 0)
        ' 7行以上のデータを持ってるidに絞る '
        target = squeeze_target(tmp, 'air_store_id', 7)

        ' ratio_flgが0なら、その日付時点で最大のdowユニーク数でランクをつける '
        if ratio_flg==0:
            rank = len(data['day_of_week'].drop_duplicates())

        dow_rank = level2_rank(target, particle, 'day_of_week', 'visitors', rank)

        ' holiday_flgをmergeして、TOP3における休日の割合を出す '
        if ratio_flg==1:
            dow_rank = dow_rank.merge(air_cal[[particle, 'day_of_week', 'holiday_flg']].drop_duplicates(), on=[particle, 'day_of_week'], how='inner')

            current = (dow_rank.groupby(particle)['holiday_flg'].sum()/3).reset_index().rename(columns={'holiday_flg': 'holiday_ratio'})

        elif ratio_flg==0:
            current = target[particle].drop_duplicates().to_frame()

        ' ***common part*** '
        current['visit_date'] = date
        current['day_of_week'] = dow

        ' ***dow_rank*** '
        if ratio_flg==0:
            current = current.merge(dow_rank, on=[particle, 'day_of_week'], how='inner')

        ' その日付時点毎の結果を積んでいく '
        if len(result)==0:
            result = current
        else:
            result = pd.concat([result, current], axis=0)

    if ratio_flg==0:
        result.to_csv(f'../feature/valid_feature/dow_rank@{particle}.csv', index=False)
    elif ratio_flg==1:
        result.to_csv(f'../feature/valid_feature/top{rank}_holiday_ratio@{particle}.csv')


def level2_rank(data, level1, level2, value, rank):
    '''
    Explain:
        level1の粒度でなく、level1*level2の粒度についてvalueを降順ソートし、
        level2のlevel1内におけるランクをつける
    Args:
        data(DF)    : データフレーム。level1, level2, valueカラムが必須
        level1(str) : 第一の粒度
        level2(str) : level1の次に細かい第二の粒度
        value(float): level2についてランクをつける際に参考とする値
        rank(int)   : 上位何番までのランクを返すか
    Return:
        result(DF): level1 * level2について、valueを降順とした時の
        level2のランクをつけたデータフレーム。
        columns: level1|level2|rank
    '''

    data = data.fillna(0)
    df_med = data.groupby([level1, level2], as_index=False)[value].median()

    for i in range(rank):
        idx_max = df_med.groupby(level1)[value].idxmax().values
        tmp = df_med.set_index(level2)
        val_max = tmp.groupby(level1)[value].idxmax().reset_index()
        val_max[f'{level2}_rank@{level1}'] = i+1
        df_med.drop(idx_max, axis=0, inplace=True)
        if i==0:
            result = val_max
        else:
            result = pd.concat([result, val_max], axis=0)

    result.rename(columns={value:level2}, inplace=True)
    result.sort_values(by=level1, inplace=True)

    return result


if __name__ == '__main__':

    main()
