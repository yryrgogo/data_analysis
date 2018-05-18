import numpy as np
import pandas as pd
import datetime
from time import time
import sys
import glob
import re
sys.path.append('../protos/')
from recruit_kaggle_load import recruit_load_data, set_validation, load_submit
sys.path.append('../../../module/')
from preprocessing import outlier

start_time = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())

""" データセット関連 """
input_path = '../input/*.csv'
submit_path = '../input/sample_submission.csv'
path_list = glob.glob(input_path)
key_air_vi = 'air_vi_june'
key_air_re = 'air_re'
key_air_st = 'air_st'
key_air_area = 'air_area'
key_air_genre = 'air_genre'
key_air_cal = 'air_cal_june_ex'
key_list = [key_air_st, key_air_vi, key_air_cal, key_air_area, key_air_genre]

""" データロード """
data_dict = recruit_load_data(key_list, path_list)
air_vi = data_dict[key_air_vi]
air_st = data_dict[key_air_st]
air_cal = data_dict[key_air_cal]
air_area = data_dict[key_air_area]
air_genre = data_dict[key_air_genre]
#  air_re = data_dict[key_air_re]

#  """ 前処理 """
#  ' 外れ値を除去 '
#  air_vi = outlier(air_vi, 'air_store_id', 'visitors', 1.64)
#  air_vi.sort_values(by=['air_store_id', 'visit_date'], inplace=True)
#  air_vi.reset_index(drop=True, inplace=True)
#  ' validation_noをセット '
#  air_vi = set_validation(air_vi)
#  '''前日のvisitorsをセット'''
#  air_vi['last_visitors'] = air_vi.groupby('air_store_id')['visitors'].shift(1)

#  air_cal.to_csv('../input/air_cal_june_ex121_out164_valset.csv', index=False)
#  air_area.to_csv('../input/air_area_june_ex121_out164_valset.csv', index=False)
#  air_genre.to_csv('../input/air_genre_june_ex121_out164_valset.csv', index=False)
#  sys.exit()


def consective_holiday(key=None, data=None, level=None):
    """
    Args:
        data:対象期間のカレンダーを持ったDF
    """

    ''' 祝日の集計を行う（土日も祝日の場合はSpecialになってる） '''
    #  tmp = data[[level, 'visit_date', 'holiday_flg', 'day_of_week', 'visitors']]
    #  tmp['n1_holiday_flg'] = tmp.groupby(level)['holiday_flg'].shift(-1)
    #  tmp['n2_holiday_flg'] = tmp.groupby(level)['holiday_flg'].shift(-2)
    #  tmp['n3_holiday_flg'] = tmp.groupby(level)['holiday_flg'].shift(-3)
    #  tmp['b1_holiday_flg'] = tmp.groupby(level)['holiday_flg'].shift(1)
    #  tmp['b2_holiday_flg'] = tmp.groupby(level)['holiday_flg'].shift(2)
    #  tmp['b3_holiday_flg'] = tmp.groupby(level)['holiday_flg'].shift(3)
    #  tmp = tmp.fillna(0)

    #  ' 翌3日の連休数を求める '
    #  tmp['next3_holiday'] = tmp.apply(lambda x:0 if x['n1_holiday_flg']==0 else 1 if x['n2_holiday_flg']==0 else 2 if x['n3_holiday_flg']==0 else 3, axis=1)

    #  ' 前3日の連休数を求める '
    #  tmp['befo3_holiday'] = tmp.apply(lambda x:0 if x['b1_holiday_flg']==0 else 1 if x['b2_holiday_flg']==0 else 2 if x['b3_holiday_flg']==0 else 3, axis=1)

    #  tmp[[level, 'visit_date', 'next3_holiday']].to_csv(f'../feature/valid_feature/next3_holiday@{level}.csv', index=False)
    #  tmp[[level, 'visit_date', 'befo3_holiday']].to_csv(f'../feature/valid_feature/befo3_holiday@{level}.csv', index=False)

    #  tmp.to_csv('../input/air_cal_next_before_holiday.csv', index=False)
    #  tmp.to_csv('../input/air_area_cal_next_before_holiday.csv', index=False)
    #  tmp.to_csv('../input/air_genre_cal_next_before_holiday.csv', index=False)
    #  print('download end')
    #  sys.exit()


    path_list = glob.glob('../input/*.csv')

    for path in path_list:
        if path.count(key):
            tmp = pd.read_csv(path)

    tmp['visit_date'] = pd.to_datetime(tmp['visit_date'])

    ''' 翌連休のある祝日のみで集計する '''
    continuous = tmp[(tmp['day_of_week'] == 'Special')
                     & (tmp['next3_holiday'] > 0)]

    ''' 翌連休のない祝日のみで集計する '''
    discrete = tmp[(tmp['day_of_week'] == 'Special') & (tmp['next3_holiday'] == 0)]

    return continuous, discrete


def main():

    ' visit dataにcalendarで全日程をもたせる '
    #  vi_date = air_cal.merge(air_vi, on=['air_store_id', 'visit_date', 'dow'], how='left')
    air_info = air_cal.merge(air_st, on='air_store_id', how='left')

    df_genre = air_info.groupby(['air_genre_name', 'visit_date', 'day_of_week', 'dow', 'holiday_flg'], as_index=False)['visitors'].mean()
    df_genre.to_csv('../input/air_genre_june_ex121_out164_valset.csv', index=False)

    df_area = air_info.groupby(['air_area_name', 'visit_date', 'day_of_week', 'dow', 'holiday_flg'], as_index=False)['visitors'].mean()
    df_area.to_csv('../input/air_area_june_ex121_out164_valset.csv.csv', index=False)


if __name__ == '__main__':

    main()
