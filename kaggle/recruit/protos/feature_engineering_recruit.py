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


def moving_agg(method, data, particle, value, window, periods):

    data = data.set_index('visit_date')
    data = data.sort_index()
    if method=='avg':
        result = data.groupby(particle)[value].rolling(
            window=window, min_periods=periods).mean().reset_index()
    elif method=='sum':
        result = data.groupby(particle)[value].rolling(
            window=window, min_periods=periods).sum().reset_index()

    result.rename(columns={value: '{}_@mv_{}_w{}_p{}'.format(value, method,
        window, periods)}, inplace=True)

    return result


def exp_weight_avg(data, particle, value, weight):

    N = len(data)
    max_date = data['visit_date'].max()

    data['diff'] = abs(date_diff(max_date, data['visit_date']))
    data['weight'] = data['diff'].map(lambda x: weight ** x.days)

    data['tmp'.format(value, weight)
         ] = data['weight'] * data['visitors']

    tmp_result = data.groupby(
        particle)['tmp', 'weight'].sum()
    result = tmp_result['tmp'.format(value, weight)]/tmp_result['weight']
    result.name = '{}_@w_avg_{}'.format(value, weight)

    return result


def date_range(data, start, end):
    return data[(start <= data['visit_date']) & (data['visit_date'] <= end)]


def make_special_set(data):

    #  ''' 祝日の集計を行う（土日も祝日の場合はSpecialになってる） '''
    #  tmp = data[['air_store_id', 'visit_date', 'holiday_flg', 'day_of_week']]
    #  tmp['n1_holiday_flg'] = tmp['holiday_flg'].shift(-1)
    #  tmp['n2_holiday_flg'] = tmp['holiday_flg'].shift(-2)
    #  tmp['n3_holiday_flg'] = tmp['holiday_flg'].shift(-3)
    #  tmp['b1_holiday_flg'] = tmp['holiday_flg'].shift(1)
    #  tmp['b2_holiday_flg'] = tmp['holiday_flg'].shift(2)
    #  tmp['b3_holiday_flg'] = tmp['holiday_flg'].shift(3)
    #  tmp = tmp.fillna(0)

    #  ' 翌3日の連休数を求める '
    #  tmp['next3_holiday'] = tmp.apply(lambda x:0 if x['n1_holiday_flg']==0 else 1 if x['n2_holiday_flg']==0 else 2 if x['n3_holiday_flg']==0 else 3, axis=1)

    #  ' 前3日の連休数を求める '
    #  tmp['befo3_holiday'] = tmp.apply(lambda x:0 if x['b1_holiday_flg']==0 else 1 if x['b2_holiday_flg']==0 else 2 if x['b3_holiday_flg']==0 else 3, axis=1)

    #  tmp.to_csv('../input/{}_air_cal_visit.csv'.format(start_time[:11]), index=False)
    #  print('download end')
    #  sys.exit()
    tmp = pd.read_csv('../input/20180429_14_air_cal_visit.csv')
    tmp['visit_date'] = pd.to_datetime(tmp['visit_date'])

    ''' 翌連休のある祝日のみで集計する '''
    continuous = tmp[(tmp['day_of_week'] == 'Special') & (tmp['next3_holiday']>0)]
    continuous = data.merge(continuous, on=['air_store_id', 'visit_date', 'day_of_week', 'holiday_flg'], how='inner', copy=False)
    continuous['last_dow_visitors'] = continuous['visitors'].shift(1)

    ''' 翌連休のない祝日のみで集計する '''
    discrete = tmp[(tmp['day_of_week'] == 'Special') & (tmp['next3_holiday']==0)]
    discrete = data.merge(discrete, on=['air_store_id', 'visit_date', 'day_of_week'], how='inner', copy=False)
    discrete['last_dow_visitors'] = discrete['visitors'].shift(1)

    return continuous, discrete


def main():

    """ 移動平均 """
    #  window_list = [7, 21, 35, 63, 126, 252, 378]
    #  for window in window_list:
    #      mv_avg = moving_agg('avg', air_vi, 'air_store_id', 'last_visit', window, 1)
    #      mv_avg.sort_values(by=['air_store_id', 'visit_date'], inplace=True)
    #      result = mv_avg.iloc[:, 2]
    #      result.to_csv('../feature/{}.csv'.format(result.name), header=True, index=False)
    #  sys.exit()

    """ 重み付き平均 """
    weight_list = [0.9, 0.95, 0.98, 0.99]
    ''' 学習データの日程（重み付き平均は1日ずつ計算してあげる必要があるので全日付リストが必要） '''
    #  date_list = air_vi['visit_date'].drop_duplicates().sort_values().values
    #  for weight in weight_list:
    #      result = pd.DataFrame([])
    #      for end_date in date_list:
    #          tmp = date_range(air_vi, first_date, end_date)
    #          wg_avg = exp_weight_avg(tmp, 'air_store_id', 'last_visit', weight).to_frame().reset_index()
    #          wg_avg['visit_date'] = end_date

    #          if len(result)==0:
    #              result = wg_avg
    #          else:
    #              result = pd.concat([result, wg_avg], axis=0)

    #      result = air_vi[['air_store_id', 'visit_date']].merge(result, on=['air_store_id', 'visit_date'], how='inner')
    #      result.sort_values(by=['air_store_id', 'visit_date'], inplace=True)
    #      print(result.shape)
    #      result = result.iloc[:,2]
    #      result.to_csv('../feature/{}.csv'.format(result.name), header=True, index=False)


    """ 曜日&祝日別の集計
    平均をとった際に影響がない様に、NULLはそのままにする。
    祝日以外の曜日集計を行う
    visit dataにcalendarで全日程をもたせる
    """
    vi_date = air_cal.merge(air_vi, on=['air_store_id', 'visit_date', 'dow'], how='left')

    ' 連休、非連休の祝日データセット作成 '
    continuous, discrete = make_special_set(vi_date)

    #  window_list = [3, 12, 24, 48]
    #  """ 移動平均 """
    #  for window in window_list:
    #      result_dow = pd.DataFrame([])
    #      for i in range(7):
    #          tmp = vi_date[vi_date['dow']==i][['air_store_id', 'visit_date', 'day_of_week', 'dow', 'visitors']]
    #          tmp_date = tmp[['air_store_id', 'visit_date', 'day_of_week', 'dow']]
    #          ''' 祝日はNULLにする '''
    #          tmp = tmp[tmp['day_of_week'] != 'Special']

    #          data = tmp_date.merge(tmp, on=['air_store_id', 'visit_date', 'day_of_week', 'dow'], how='left', copy=False)
    #          ''' 一週前の数字を持たせる '''
    #          data['last_dow_visitors'] = data['visitors'].shift(1)

    #          dow_mv = moving_agg('avg', data, 'air_store_id', 'last_dow_visitors', window, 1)
    #          col_name = [col for col in dow_mv.columns if col.count('@')][0]

    #          ' inner joinでSpecialを除外する '
    #          tmp_result = dow_mv.merge(tmp[['air_store_id', 'visit_date']], on=['air_store_id', 'visit_date'], how='inner')

    #          ' Null埋めする為、店毎の平均値を算出 '
    #          dow_avg = dow_mv.groupby('air_store_id', as_index=False)[col_name].mean()
    #          ' Null埋め '
    #          null = tmp_result[tmp_result[col_name].isnull()]
    #          fill_null = null[['air_store_id', 'visit_date']].merge(dow_avg, on='air_store_id', how='inner')
    #          tmp_result.dropna(inplace=True)

    #          tmp_result = pd.concat([tmp_result, fill_null], axis=0)

    #          if len(result_dow)==0:
    #              result_dow = tmp_result
    #          else:
    #              result_dow = pd.concat([result_dow, tmp_result], axis=0)

    #      '''** dowで集計し、concatした後、連休の方もconcatする **'''
    #      continuous_mv = moving_agg('avg', continuous, 'air_store_id', 'last_dow_visitors', window, 1)

    #      ' feature nameを取得 '
    #      col_name = [col for col in continuous_mv.columns if col.count('@')][0]

    #      ' Null埋めする為、各店舗の平均値を取得 '
    #      continuous_mv_avg = continuous_mv.groupby('air_store_id', as_index=False)[col_name].mean()

    #      ' 移動平均の平均値でNullを埋める '
    #      null = continuous_mv[continuous_mv[col_name].isnull()]
    #      fill_null = null[['air_store_id', 'visit_date']].merge(continuous_mv_avg, on='air_store_id', how='inner')

    #      continuous_mv.dropna(inplace=True)
    #      result_cont = pd.concat([continuous_mv, fill_null], axis=0)

    #      discrete_mv = moving_agg('avg', discrete, 'air_store_id', 'last_dow_visitors', window, 1)

    #      ' feature nameを取得 '
    #      col_name = [col for col in discrete_mv.columns if col.count('@')][0]

    #      ' Null埋めする為、各店舗の平均値を取得 '
    #      discrete_mv_avg = discrete_mv.groupby('air_store_id', as_index=False)[col_name].mean()

    #      ' 移動平均の平均値でNullを埋める '
    #      null = discrete_mv[discrete_mv[col_name].isnull()]
    #      fill_null = null[['air_store_id', 'visit_date']].merge(discrete_mv_avg, on='air_store_id', how='inner')

    #      discrete_mv.dropna(inplace=True)
    #      result_disc = pd.concat([discrete_mv, fill_null], axis=0)

    #      result = pd.concat([result_dow, result_cont, result_disc], axis=0)

    #      ' 日程を元のデータセットと同様にする '
    #      result = air_vi[['air_store_id', 'visit_date']].merge(result, on=['air_store_id', 'visit_date'], how='inner')

    #      result.to_csv('../feature/{}.csv'.format(col_name))

    #  sys.exit()

    """ 重み付き平均 """
    date_list = data['visit_date'].drop_duplicates().sort_values().values
    for i in range(7):
        for weight in weight_list:
            for end_date in date_list:
                data = date_range(data, first_date, end_date)
                dow_wg = exp_weight_avg(data, 'air_store_id', 'last_dow_visitors', weight)

                name = dow_wg.name
                dow_wg = dow_wg.to_frame()
                dow_wg['visit_date'] = end_date

                print(dow_wg)
                sys.exit()

    date_list = continuous['visit_date'].drop_duplicates().sort_values().values

    for weight in weight_list:
        for end_date in date_list:
            continuous = date_range(continuous, first_date, end_date)
            exp_weight_avg(continuous, 'air_store_id', 'last_dow_visitors', weight)

    date_list = discrete['visit_date'].drop_duplicates().sort_values().values

    for weight in weight_list:
        for end_date in date_list:
            discrete = date_range(discrete, first_date, end_date)
            exp_weight_avg(discrete, 'air_store_id', 'last_dow_visitors', weight)


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
