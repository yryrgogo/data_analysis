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
key_air_vi = 'air_vi_ex'
key_air_re = 'air_re'
key_air_st = 'air_st'
key_air_cal = 'air_cal_june_ex'
key_list = [key_air_st, key_air_vi, key_air_cal]

""" データロード """
data_dict = recruit_load_data(key_list, path_list)
air_vi = data_dict[key_air_vi]
air_st = data_dict[key_air_st]
air_cal = data_dict[key_air_cal]
#  air_re = data_dict[key_air_re]


""" 前処理 """
' 外れ値を除去 '
air_vi = outlier(air_vi, 'air_store_id', 'visitors', 1.64)
air_vi.sort_values(by=['air_store_id', 'visit_date'], inplace=True)
air_vi.reset_index(drop=True, inplace=True)
' validation_noをセット '
air_vi = set_validation(air_vi)
'''前日のvisitorsをセット'''
air_vi['last_visitors'] = air_vi.groupby('air_store_id')['visitors'].shift(1)


def main():

    ' visit dataにcalendarで全日程をもたせる '
    vi_date = air_cal.merge(air_vi, on=['air_store_id', 'visit_date', 'dow'], how='left')
    air_info = vi_date.merge(air_st, on='air_store_id', how='left')

    df_area = air_info.groupby(['air_area_name', 'visit_date', 'day_of_week', 'dow'], as_index=False)['visitors', 'last_visitors'].mean()

    vi_date.to_csv('../input/air_vi_date_june_ex121_out164_valset.csv', index=False)
    air_vi.to_csv('../input/air_vi_june_ex121_out164_valset.csv', index=False)
    df_area.to_csv('../input/area_vi_date_june_ex121_out164_valset.csv', index=False)


if __name__ == '__main__':

    main()
