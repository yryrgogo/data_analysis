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


def set_validation(data):
    start_date = pd.to_datetime('2017-03-12')
    end_date   = pd.to_datetime('2017-04-22')
    data['validation'] = data['visit_date'].map(lambda x: 1 if start_date <= x and x <= end_date else 0)

    return data


def make_air_calendar(air_vi, air_cal):

    ' 年末年始を除く、年末年始のセットを作る '
    extract = air_vi[air_vi['visit_date']=='2016-01-06']['visit_date'].values[0]
    first_extract = air_vi[air_vi['visit_date']=='2016-12-24']['visit_date'].values[0]
    last_extract = air_vi[air_vi['visit_date']=='2017-01-06']['visit_date'].values[0]

    air_cal = air_cal[(extract < air_cal['visit_date'])]
    air_cal_1 = air_cal[(first_extract > air_cal['visit_date'])]
    air_cal_2 = air_cal[(air_cal['visit_date'] > last_extract)]
    air_cal = pd.concat([air_cal_1, air_cal_2], axis=0)

    air_cal.to_csv('../input/air_cal_june_extract_year_end.csv', index=False)
    #  air_cal.to_csv('../input/air_calendar_extract_year_end.csv', index=False)
    #  air_cal.to_csv('../input/air_calendar_year_end_start.csv', index=False)
    sys.exit()


    ''' 全日付を持たせたデータ '''
    air_min_date = air_vi.groupby('air_store_id', as_index=False)['visit_date'].min()
    air_id = air_min_date['air_store_id'].values
    min_date = air_min_date['visit_date'].values

    tmp = df_date.copy()
    for i,air,date in zip(range(len(air_id)), air_id, min_date):
        tmp_date = tmp[tmp['visit_date']>=date]
        tmp_date['air_store_id'] = air
        if i==0:
            air_calendar = tmp_date
        else:
            air_calendar = pd.concat([air_calendar, tmp_date], axis=0)


def make_june_calendar():

    date = ['2017-06-01', '2017-06-02', '2017-06-03', '2017-06-04', '2017-06-05']
    dow_no = [3, 4, 5, 6, 0]
    dow = ['Thursday', 'Friday', 'Saturday', 'Sunday', 'Monday']
    holiday = [0, 0, 1, 1, 0]

    june = pd.DataFrame({'visit_date':date, 'day_of_week':dow, 'holiday_flg':holiday, 'dow':dow_no})
    air_id = air_cal['air_store_id'].drop_duplicates().values

    result = pd.DataFrame([])
    for air in air_id:
        june['air_store_id'] = air
        print(june.shape)

        if len(result)==0:
            result = june
        else:
            result = pd.concat([result, june], axis=0)

    result.to_csv('../input/june_air_cal.csv', index=False)

    print(air_cal.shape)
    result = pd.concat([air_cal, result], axis=0)

    result.to_csv('../input/air_cal_june.csv', index=False)
    print(result.shape)


"""並列でデータセットをロード"""
def recruit_load_data(key_list, path_list):

    data_dict = {key:None for key in key_list}
    p_list = pararell_load_data(key_list, path_list)

    for d_dict in p_list:
        for key, df in d_dict.items():
            if key.count('air_re'):
                air_re = df
                air_re['visit_date'] = pd.to_datetime(air_re['visit_datetime'].str[:10])
                air_re['reserve_date'] = pd.to_datetime(air_re['reserve_datetime'].str[:10])
                air_re['dow'] = air_re['visit_date'].dt.dayofweek
                data_dict[key] = air_re

            elif key.count('air_st'):
                air_st = df
                data_dict[key] = air_st

            elif key.count('air_vi'):
                air_vi = df
                air_vi['visit_date'] = pd.to_datetime(air_vi['visit_date'])
                air_vi['dow'] = air_vi['visit_date'].dt.dayofweek
                data_dict[key] = air_vi

            elif key.count('air_cal'):
                air_cal = df
                air_cal['visit_date'] = pd.to_datetime(air_cal['visit_date'])
                data_dict[key] = air_cal

            elif key.count('air_area'):
                air_area = df
                air_area['visit_date'] = pd.to_datetime(air_area['visit_date'])
                data_dict[key] = air_area

            elif key.count('air_genre'):
                air_genre = df
                air_genre['visit_date'] = pd.to_datetime(air_genre['visit_date'])
                data_dict[key] = air_genre

            elif key.count('air_cal_june【'):
                raw_cal = df
                raw_cal['visit_date'] = pd.to_datetime(raw_cal['visit_date'])
                data_dict[key] = raw_cal

    return data_dict


def load_submit(submit_path):
    submit = pd.read_csv(submit_path)
    submit['air_store_id'] = submit['id'].str[:-11]
    submit['visit_date'] = pd.to_datetime(submit['id'].str[-10:])

    return submit

