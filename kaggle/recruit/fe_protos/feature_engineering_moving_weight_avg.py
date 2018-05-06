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
from recruit_kaggle_load import load_data, set_validation, date_diff, load_submit
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
#  base = pd.read_csv('../input/20180424_air_base.csv')
data_dict = load_data(key_list, path_list)
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
air_vi['last_visit'] = air_vi.groupby('air_store_id')['visitors'].shift(1)

first_date = air_vi['visit_date'].min()
last_date = air_vi['visit_date'].max()

''' カレンダーの日付を絞る '''
air_cal = air_cal[air_cal['visit_date'] <= last_date]


def moving_agg(method, data, particle, value, window, periods):

    data = data.set_index('visit_date')
    data = data.sort_index()
    if method=='avg':
        result = data.groupby(particle)[value].rolling(
            window=window, min_periods=periods).mean().reset_index()
    elif method=='sum':
        result = data.groupby(particle)[value].rolling(
            window=window, min_periods=periods).sum().reset_index()

    result.rename(columns={value: f'{value}_@mv_{method}_w{window}_p{periods}@{particle}'}, inplace=True)

    return result


def exp_weight_avg(data, particle, value, weight):

    N = len(data)
    max_date = data['visit_date'].max()

    data['diff'] = abs(date_diff(max_date, data['visit_date']))
    data['weight'] = data['diff'].map(lambda x: weight ** x.days)

    data['tmp'] = data['weight'] * data[value]

    ' valueがNullになっている行の重みは0にする。（分母の合計に入ってしまう為） '
    no_null = data[data[value].notnull()]
    null = data[data[value].isnull()]
    if len(null)>0:
        null['weight'] = 0
    if len(null)>0 and len(no_null)>0:
        data = pd.concat([null, no_null], axis=0)
    elif len(null)==0 and len(no_null)>0:
        data = no_null
    elif len(null)>0 and len(no_null)==0:
        data = null
    #  data[value] = data[value].fillna(-10)
    #  data['weight'] = data.apply(lambda x:0 if x[value]<0 else x['weight'], axis=1)
    tmp_result = data.groupby(particle)['tmp', 'weight'].sum()
    result = tmp_result['tmp']/tmp_result['weight']

    result.name = f'{value}_@w_avg_{weight}@{particle}'

    return result


def date_range(data, start, end):
    return data[(start <= data['visit_date']) & (data['visit_date'] <= end)]


def make_special_set(data, particle):
    """
    Args:
        data:対象期間のカレンダーを持ったDF
    """

    ''' 祝日の集計を行う（土日も祝日の場合はSpecialになってる） '''
    #  tmp = data[['air_store_id', 'visit_date', 'holiday_flg', 'day_of_week']]
    #  tmp['n1_holiday_flg'] = tmp.groupby('air_store_id')['holiday_flg'].shift(-1)
    #  tmp['n2_holiday_flg'] = tmp.groupby('air_store_id')['holiday_flg'].shift(-2)
    #  tmp['n3_holiday_flg'] = tmp.groupby('air_store_id')['holiday_flg'].shift(-3)
    #  tmp['b1_holiday_flg'] = tmp.groupby('air_store_id')['holiday_flg'].shift(1)
    #  tmp['b2_holiday_flg'] = tmp.groupby('air_store_id')['holiday_flg'].shift(2)
    #  tmp['b3_holiday_flg'] = tmp.groupby('air_store_id')['holiday_flg'].shift(3)
    #  tmp = tmp.fillna(0)

    #  ' 翌3日の連休数を求める '
    #  tmp['next3_holiday'] = tmp.apply(lambda x:0 if x['n1_holiday_flg']==0 else 1 if x['n2_holiday_flg']==0 else 2 if x['n3_holiday_flg']==0 else 3, axis=1)

    #  ' 前3日の連休数を求める '
    #  tmp['befo3_holiday'] = tmp.apply(lambda x:0 if x['b1_holiday_flg']==0 else 1 if x['b2_holiday_flg']==0 else 2 if x['b3_holiday_flg']==0 else 3, axis=1)

    #  tmp[['air_store_id', 'visit_date', 'next3_holiday']].to_csv('../feature/valid_feature/next3_holiday.csv', index=False)
    #  tmp[['air_store_id', 'visit_date', 'befo3_holiday']].to_csv('../feature/valid_feature/befo3_holiday.csv', index=False)
    #  tmp.to_csv('../input/air_cal_next_before_holiday.csv', index=False)
    #  print('download end')
    #  sys.exit()

    tmp = pd.read_csv('../input/air_cal_next_before_holiday.csv')
    tmp = tmp.merge(air_st, on='air_store_id', how='left')
    tmp['visit_date'] = pd.to_datetime(tmp['visit_date'])

    ''' 翌連休のある祝日のみで集計する '''
    continuous = tmp[(tmp['day_of_week'] == 'Special') & (tmp['next3_holiday']>0)]
    continuous = data.merge(continuous, on=[particle, 'visit_date', 'day_of_week'], how='inner', copy=False)
    continuous['last_dow_visitors'] = continuous.groupby(particle)['visitors'].shift(1)

    ''' 翌連休のない祝日のみで集計する '''
    discrete = tmp[(tmp['day_of_week'] == 'Special') & (tmp['next3_holiday']==0)]
    discrete = data.merge(discrete, on=[particle, 'visit_date', 'day_of_week'], how='inner', copy=False)
    discrete['last_dow_visitors'] = discrete.groupby(particle)['visitors'].shift(1)

    return continuous, discrete


def main():

    ' visit dataにcalendarで全日程をもたせる '
    vi_date = air_cal.merge(air_vi, on=['air_store_id', 'visit_date', 'dow'], how='left')
    air_info = vi_date.merge(air_st, on='air_store_id', how='left')

    df_area = air_info.groupby(['air_area_name', 'visit_date', 'day_of_week', 'dow'], as_index=False)['visitors', 'last_visit'].mean()

    ' 統計量を求める粒度とDFを決定 '
    ptc_list = [
        #  'air_store_id'
        #  ,
        'air_area_name'
    ]
    df_list  = [
        #  vi_date
        #  ,
        df_area
    ]

    for particle, df_input in zip(ptc_list, df_list):

        let_aggregation(df_input, particle)


def let_aggregation(df_input, particle):

    """ 移動平均 """
    #  window_list = [7, 63, 126]
    #  for window in window_list:
    #      mv_avg = moving_agg('avg', df_input, particle, 'last_visit', window, 1)
    #      #  mv_avg = moving_agg('avg', vi_date, 'air_store_id', 'last_visit', window, int(window/2))

    #      col_name = [col for col in mv_avg.columns if col.count('@')][0]
    #      ' Null埋めする為、各店舗の平均値を取得 '
    #      mv_avg_avg = mv_avg.groupby(particle, as_index=False)[col_name].mean()

    #      ' 移動平均の平均値でNullを埋める '
    #      null = mv_avg[mv_avg[col_name].isnull()]
    #      fill_null = null[[particle, 'visit_date']].merge(mv_avg_avg, on=particle, how='inner')

    #      mv_avg.dropna(inplace=True)
    #      result = pd.concat([mv_avg, fill_null], axis=0)

    #      result = df_input[[particle, 'visit_date']].merge(result, on=[particle, 'visit_date'], how='inner')

    #      result.to_csv(f'../feature/valid_feature/{col_name}.csv', header=True, index=False)
    #      print(col_name)
    #      print(result.shape)
    #  sys.exit()

    """ 重み付き平均 """
    weight_list = [0.9, 0.95, 0.98]
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
    #      result = result.iloc[:,2]
    #      result.to_csv('../feature/{}.csv'.format(result.name), header=True, index=False)
    #      print(result.shape)

    #  sys.exit()

    """ 曜日&祝日別の集計
    平均をとった際に影響がない様に、NULLはそのままにする。
    祝日以外の曜日集計を行う
    """

    #  vi_date = vi_date[vi_date['air_store_id']=='air_1c0b150f9e696a5f']
    ' 連休、非連休の祝日データセット作成 '
    continuous, discrete = make_special_set(df_input, particle)

    """ 移動平均 """
    #  window_list = [3, 12, 24, 48]
    #  for window in window_list:
    #      result_dow = pd.DataFrame([])
    #      for i in range(7):
    #          tmp = vi_date[vi_date['dow']==i][['air_store_id', 'visit_date', 'day_of_week', 'dow', 'visitors']]
    #          tmp_date = tmp[['air_store_id', 'visit_date', 'day_of_week', 'dow']]
    #          ''' 祝日のvisitorsは別途集計する為、各曜日におけるvisitorsはNULLにする '''
    #          tmp = tmp[tmp['day_of_week'] != 'Special']
    #          data = tmp_date.merge(tmp, on=['air_store_id', 'visit_date', 'day_of_week', 'dow'], how='left', copy=False)
    #          ''' 一週前の数字を持たせる '''
    #          data['last_dow_visitors'] = data.groupby('air_store_id')['visitors'].shift(1)

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

    #      result.sort_values(by=['air_store_id', 'visit_date'], inplace=True)
    #      result = result[col_name]
    #      result.to_csv('../feature/{}.csv'.format(col_name), index=False, header=True)
    #      print(result.shape)


    """ 重み付き平均 """
    for weight in weight_list:
        result_dow = pd.DataFrame([])
        for i in range(7):
            ' ここではまだその曜日の祝日が残っている '
            tmp = df_input[df_input['dow']==i][[particle, 'visit_date', 'day_of_week', 'dow', 'visitors']]

            ''' 祝日のvisitorsは別途集計する為、各曜日におけるvisitorsはNULLにする '''
            tmp_date = tmp[[particle, 'visit_date', 'day_of_week', 'dow']]
            no_sp = tmp[tmp['day_of_week'] != 'Special']
            data = tmp_date.merge(no_sp, on=[particle, 'visit_date', 'day_of_week', 'dow'], how='left', copy=False)
            data['last_dow_visitors'] = data.groupby(particle)['visitors'].shift(1)

            date_list = data['visit_date'].drop_duplicates().sort_values().values
            '''
            重み付き平均はその期間における直近日の集計値のみが求まる。
            全日程を学習データとするなら、各日時点の重み月平均を求めてあげる
            '''
            tmp_result_dow = pd.DataFrame([])
            for end_date in date_list:
                tmp = date_range(data, first_date, end_date)

                dow_wg = exp_weight_avg(tmp, particle, 'last_dow_visitors', weight)

                ' このnameを特徴量ファイル名とする '
                col_name = dow_wg.name
                dow_wg = dow_wg.to_frame().reset_index()
                dow_wg['visit_date'] = end_date

                ' id * visit_date * feature のDFを作る '
                if len(tmp_result_dow)==0:
                    tmp_result_dow = dow_wg
                else:
                    tmp_result_dow = pd.concat([tmp_result_dow, dow_wg], axis=0)

            '''
            Special週単位で重みを付け集計する為、SpecialはNULLとして残して
            いたが、Specialは別途集計して値を出す為、ここの集約時は除外する
            '''
            ' tmpはSpecialを除いたdowのDF '
            tmp_result_dow = tmp_result_dow.merge(no_sp[[particle, 'visit_date']], on=[particle, 'visit_date'], how='inner')

            ' そのdowの全日程における重み付き平均が求まったら、NULL埋めをする '
            ' Null埋めする為、各店舗の平均値を取得 '
            tmp_wg_avg = tmp_result_dow.groupby(particle, as_index=False)[col_name].mean()

            ' 重み付き平均の平均値でNullを埋める '
            null = tmp_result_dow[tmp_result_dow[col_name].isnull()]
            fill_null = null[[particle, 'visit_date']].merge(tmp_wg_avg, on=particle, how='inner')

            tmp_result_dow.dropna(inplace=True)
            tmp_result_dow = pd.concat([tmp_result_dow, fill_null], axis=0)

            ' NULL埋めまで終えた各dowのDFを積んで完成 '
            if len(result_dow)==0:
                result_dow = tmp_result_dow
            else:
                result_dow = pd.concat([result_dow, tmp_result_dow], axis=0)

        cont_result = pd.DataFrame([])
        date_list = continuous['visit_date'].drop_duplicates().sort_values().values
        for end_date in date_list:
            tmp = date_range(continuous, first_date, end_date)
            continuous_wg = exp_weight_avg(tmp, particle, 'last_dow_visitors', weight)

            col_name = continuous_wg.name

            continuous_wg = continuous_wg.to_frame().reset_index()
            continuous_wg['visit_date'] = end_date

            ' id * visit_date * feature のDFを作る '
            if len(cont_result)==0:
                cont_result = continuous_wg
            else:
                cont_result = pd.concat([cont_result, continuous_wg], axis=0)

        ' Null埋めする為、各店舗の平均値を取得 '
        cont_wg_avg = cont_result.groupby(particle, as_index=False)[col_name].mean()

        ' 重み付き平均の平均値でNullを埋める '
        null = cont_result[cont_result[col_name].isnull()]
        fill_null = null[[particle, 'visit_date']].merge(cont_wg_avg, on=particle, how='inner')

        cont_result.dropna(inplace=True)
        cont_result = pd.concat([cont_result, fill_null], axis=0)

        disc_result = pd.DataFrame([])
        date_list = discrete['visit_date'].drop_duplicates().sort_values().values
        for end_date in date_list:
            tmp = date_range(discrete, first_date, end_date)
            discrete_wg = exp_weight_avg(tmp, particle, 'last_dow_visitors', weight)

            col_name = discrete_wg.name

            discrete_wg = discrete_wg.to_frame().reset_index()
            discrete_wg['visit_date'] = end_date

            ' id * visit_date * feature のDFを作る '
            if len(disc_result)==0:
                disc_result = discrete_wg
            else:
                disc_result = pd.concat([disc_result, discrete_wg], axis=0)

        ' Null埋めする為、各店舗の平均値を取得 '
        disc_wg_avg = disc_result.groupby(particle, as_index=False)[col_name].mean()

        ' 重み付き平均の平均値でNullを埋める '
        null = disc_result[disc_result[col_name].isnull()]
        fill_null = null[[particle, 'visit_date']].merge(disc_wg_avg, on=particle, how='inner')

        disc_result.dropna(inplace=True)
        disc_result = pd.concat([disc_result, fill_null], axis=0)

        result = pd.concat([result_dow, cont_result, disc_result], axis=0)
        result = df_input[[particle, 'visit_date']].merge(result, on=[particle, 'visit_date'], how='inner')
        result.sort_values(by=[particle, 'visit_date'], inplace=True)

        result.to_csv(f'../feature/valid_feature/{col_name}.csv', index=False, header=True)
        print(result.shape)

if __name__ == '__main__':

    main()
