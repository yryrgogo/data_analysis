import dask.dataframe as ddf
import dask.multiprocessing
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_log_error
import datetime
from datetime import date, timedelta
from dateutil.parser import parse
import sys


# load_data********************************
df_train = pd.read_csv('../input/air_visit_data.csv').rename(columns={'air_store_id':'store_id'})
df_train = df_train.loc[:800, df_train.columns]
df_airSI = pd.read_csv('../input/air_store_info.csv').rename(columns={'air_store_id':'store_id'})
df_airR = pd.read_csv('../input/air_reserve.csv').rename(columns={'air_store_id':'store_id'})
df_hpgSI = pd.read_csv('../input/hpg_store_info.csv').rename(columns={'hpg_store_id':'store_id'})
df_hpgR = pd.read_csv('../input/hpg_reserve.csv').rename(columns={'hpg_store_id':'store_id'})
df_sir = pd.read_csv('../input/store_id_relation.csv').set_index('hpg_store_id', drop=False)
df_di = pd.read_csv('../input/date_info.csv').rename(columns={'calendar_date':'visit_date'})
df_submit = pd.read_csv('../input/sample_submission.csv').rename(columns={'hpg_store_id':'store_id'})

# preprocessing****************************
df_submit['store_id'] = df_submit['id'].str[:-11]
df_submit['visit_date'] = pd.to_datetime(df_submit['id'].str[-10:])
df_train['visit_date'] = pd.to_datetime(df_train['visit_date'])
df_train['dow'] = df_train['visit_date'].dt.dayofweek
df_airR['visit_date'] = pd.to_datetime(df_airR['visit_datetime'].str[:10])
df_airR['reserve_date'] = pd.to_datetime(df_airR['reserve_datetime'].str[:10])
df_airR['dow'] = df_airR['visit_date'].dt.dayofweek
df_hpgR['visit_date'] = pd.to_datetime(df_hpgR['visit_datetime'].str[:10])
df_hpgR['reserve_date'] = pd.to_datetime(df_hpgR['reserve_datetime'].str[:10])
df_hpgR['dow'] = df_hpgR['visit_date'].dt.dayofweek
df_hpgR['store_id'] = df_hpgR['store_id'].map(df_sir['air_store_id']).fillna(df_hpgR['store_id'])
df_di['visit_date'] = pd.to_datetime(df_di['visit_date'])
df_di['dow'] = df_di['visit_date'].dt.dayofweek
# SpecialとSaturday/Sundayのholiday_flg=1に
df_di['day_of_week'] = df_di.apply(lambda x:'Special' if x.holiday_flg==1 and (x.day_of_week != 'Saturday' or x.day_of_week != 'Sunday') else x.day_of_week, axis=1)
df_di['holiday_flg'] = df_di.apply(lambda x:1 if x.holiday_flg==1 or x.day_of_week=='Saturday' or x.day_of_week=='Sunday' else 0, axis=1)

# 前3日間の休日数
df_di.sort_values(by='visit_date', ascending=True, inplace=True)
df_di['before_3day_flg'] = df_di['holiday_flg'].rolling(window=4, min_periods=1).sum()
df_di['before_3day_flg'] = df_di.apply(lambda x:x.before_3day_flg-x.holiday_flg, axis=1)
# 後3日間の休日数
df_di.sort_values(by='visit_date', ascending=False, inplace=True)
df_di['after_3day_flg'] = df_di['holiday_flg'].rolling(window=4, min_periods=1).sum()
df_di['after_3day_flg'] = df_di.apply(lambda x:x.after_3day_flg-x.holiday_flg, axis=1)
df_di.sort_values(by='visit_date', ascending=True, inplace=True)

df_train = df_train.merge(df_di, on=['visit_date', 'dow'], how='inner')
df_train = df_train.merge(df_airSI, on=['store_id'], how='inner')
df_airR = df_airR.merge(df_airSI, on=['store_id'], how='inner')


def RMSLE(y_obs, y_pred):
    del_idx = np.arange(len(y_obs))[y_obs == 0]
    y_obs = np.delete(y_obs, del_idx)
    y_pred = np.delete(y_pred, del_idx)
    y_pred = y_pred.clip(min=0.)
    return np.sqrt(mean_squared_log_error(y_obs, y_pred))


def outlier(x):
    return x*1.96


def date_diff(end_date, start_date):
    return (end_date - start_date).days


def store_agg(data, end_date, n_days):
    start_date = end_date - timedelta(days=n_days)
    df_tmp = data[(data['visit_date']>=start_date) & (data['visit_date']<=end_date)].copy()
    result = df_tmp.groupby('store_id', as_index=False)['visitors'].agg(
        {'store_mean_{}'.format(n_days):'mean',
         'store_max_{}'.format(n_days):'max',
         'store_min_{}'.format(n_days):'min',
         'store_median_{}'.format(n_days):'median',
         'store_std_{}'.format(n_days):'std',
         'store_count_{}'.format(n_days):'count'})

    return result


def store_hlf_agg(data, end_date, n_days):
    start_date = end_date - timedelta(days=n_days)
    df_tmp = data[(data['visit_date']>=start_date) & (data['visit_date']<=end_date)].copy()
    result = df_tmp.groupby(['store_id', 'holiday_flg'], as_index=False)['visitors'].agg(
        {'store_hlf_mean_{}'.format(n_days):'mean',
         'store_hlf_max_{}'.format(n_days):'max',
         'store_hlf_min_{}'.format(n_days):'min',
         'store_hlf_median_{}'.format(n_days):'median',
         'store_hlf_std_{}'.format(n_days):'std',
         'store_hlf_count_{}'.format(n_days):'count'})

    return result


def store_dow_agg(data, end_date, n_days):
    start_date = end_date - timedelta(days=n_days)
    df_tmp = data[(data['visit_date']>=start_date) & (data['visit_date']<=end_date)].copy()
    result = df_tmp.groupby(['store_id', 'dow'], as_index=False)['visitors'].agg(
        {'store_dow_mean_{}'.format(n_days):'mean',
         'store_dow_max_{}'.format(n_days):'max',
         'store_dow_min_{}'.format(n_days):'min',
         'store_dow_median_{}'.format(n_days):'median',
         'store_dow_std_{}'.format(n_days):'std',
         'store_dow_count_{}'.format(n_days):'count'})

    return result


def store_weight_mean(data, end_date, n_days):
    start_date = end_date - timedelta(days=n_days)
    df_tmp = data[(data['visit_date']>=start_date) & (data['visit_date']<=end_date)].copy()
    df_tmp['weight'] = df_tmp.apply(lambda x:0.985**date_diff(end_date, x.visit_date), axis=1)
    df_tmp['visitors'] = df_tmp['visitors'] * df_tmp['weight']
    result = df_tmp.groupby(['store_id'], as_index=False)[['visitors', 'weight']].sum()
    result['store_exp_wm_{}'.format(n_days)] = result.visitors / result.weight

    return result


def store_dow_weight_mean(data, end_date, n_days):
    result = None
    start_date = end_date - timedelta(days=n_days)
    df_tmp = data[(data['visit_date']>=start_date) & (data['visit_date']<=end_date)].copy()
    for ratio in [0.9, 0.95, 0.97, 0.98, 0.985, 0.99, 0.999, 0.9999]:
        df_tmp['weight'] = df_tmp.apply(lambda x:ratio**date_diff(end_date, x.visit_date), axis=1)
        df_tmp['store_dow_exp_wm_{}'.format(ratio)] = df_tmp['visitors'] * df_tmp['weight']
        result_tmp = df_tmp.groupby(['store_id', 'dow'], as_index=False)[['store_dow_exp_wm_{}'.format(ratio), 'weight']].sum()
        result_tmp['store_dow_exp_wm_{}'.format(ratio)] = result_tmp['store_dow_exp_wm_{}'.format(ratio)] / result_tmp.weight
        result_tmp.drop('weight', axis=1, inplace=True)
        if result is None:
            result = result_tmp
        else:
            result = result.merge(result_tmp, on=['store_id', 'dow'], how='inner')

    return result


# 連休前や休日前後の違いを的確に捉えられる様に特徴量を充実させる
def store_before_after_holiday_agg(data, end_date, n_days):
    start_date = end_date - timedelta(days=n_days)
    df_tmp = data[(data['visit_date']>=start_date) & (data['visit_date']<=end_date)].copy()
    tmp_result1 = df_tmp.groupby(['store_id', 'before_3day_flg'], as_index=False)['visitors'].agg(
        {'store_bef_hlf_mean':'mean',
         'store_bef_hlf_max':'max',
         'store_bef_hlf_min':'min',
         'store_bef_hlf_median':'median',
         'store_bef_hlf_std':'std',
         'store_bef_hlf_count':'count'})
    tmp_result2 = df_tmp.groupby(['store_id', 'after_3day_flg'], as_index=False)['visitors'].agg(
        {'store_after_hlf_mean':'mean',
         'store_after_hlf_max':'max',
         'store_after_hlf_min':'min',
         'store_after_hlf_median':'median',
         'store_after_hlf_std':'std',
         'store_after_hlf_count':'count'})

    result1 = df_tmp.merge(tmp_result1, on=['store_id', 'before_3day_flg'], how='inner')
    result2 = result1.merge(tmp_result2, on=['store_id', 'after_3day_flg'], how='inner')
    return result2


def genre_weight_mean(data, end_date, n_days):
    start_date = end_date - timedelta(days=n_days)
    df_tmp = data[(data['visit_date']>=start_date) & (data['visit_date']<=end_date)].copy()
    df_tmp['weight'] = df_tmp.apply(lambda x:0.985**date_diff(end_date, x.visit_date), axis=1)
    df_tmp['visitors'] = df_tmp['visitors'] * df_tmp['weight']
    result = df_tmp.groupby(['air_genre_name'], as_index=False)[['visitors', 'weight']].sum()
    result['genre_exp_wm_{}'.format(n_days)] = result.visitors / result.weight

    return result


def genre_dow_weight_mean(data, end_date, n_days):
    start_date = end_date - timedelta(days=n_days)
    df_tmp = data[(data['visit_date']>=start_date) & (data['visit_date']<=end_date)].copy()
    df_tmp['weight'] = df_tmp.apply(lambda x:0.985**date_diff(end_date, x.visit_date), axis=1)
    df_tmp['visitors'] = df_tmp['visitors'] * df_tmp['weight']
    result = df_tmp.groupby(['air_genre_name', 'dow'], as_index=False)[['visitors', 'weight']].sum()
    result['genre_dow_exp_wm_{}'.format(n_days)] = result.visitors / result.weight

    return result


def genre_agg(data, end_date, n_days):
    start_date = end_date - timedelta(days=n_days)
    df_tmp = data[(data['visit_date']>=start_date) & (data['visit_date']<=end_date)].copy()
    result = df_tmp.groupby('air_genre_name', as_index=False)['visitors'].agg(
        {'genre_mean_{}'.format(n_days):'mean',
         'genre_max_{}'.format(n_days):'max',
         'genre_min_{}'.format(n_days):'min',
         'genre_median_{}'.format(n_days):'median',
         'genre_std_{}'.format(n_days):'std',
         'genre_count_{}'.format(n_days):'count'})

    return result


def genre_hlf_agg(data, end_date, n_days):
    start_date = end_date - timedelta(days=n_days)
    df_tmp = data[(data['visit_date']>=start_date) & (data['visit_date']<=end_date)].copy()
    result = df_tmp.groupby(['air_genre_name', 'holiday_flg'], as_index=False)['visitors'].agg(
        {'genre_hlf_mean_{}'.format(n_days):'mean',
         'genre_hlf_max_{}'.format(n_days):'max',
         'genre_hlf_min_{}'.format(n_days):'min',
         'genre_hlf_median_{}'.format(n_days):'median',
         'genre_hlf_std_{}'.format(n_days):'std',
         'genre_hlf_count_{}'.format(n_days):'count'})

    return result


def genre_dow_agg(data, end_date, n_days):
    start_date = end_date - timedelta(days=n_days)
    df_tmp = data[(data['visit_date']>=start_date) & (data['visit_date']<=end_date)].copy()
    result = df_tmp.groupby(['air_genre_name', 'dow'], as_index=False)['visitors'].agg(
        {'genre_dow_mean_{}'.format(n_days):'mean',
         'genre_dow_max_{}'.format(n_days):'max',
         'genre_dow_min_{}'.format(n_days):'min',
         'genre_dow_median_{}'.format(n_days):'median',
         'genre_dow_std_{}'.format(n_days):'std',
         'genre_dow_count_{}'.format(n_days):'count'})

    return result


def store_sales_time(data):
# 時間帯を抽出
    data['visit_time'] = data['visit_datetime'].str[11:13]
#***********そもそも12時間表記と24時間表記が混在してないかチェックが必要
# 00:00~06:00は24:00を加える
    data['visit_time'] = data['visit_time'].map(lambda x: 24+int(x[1]) if x[0]=='0' and int(x[1])<7 else int(x[1]) if (x[0]=='0' and int(x[1])>=7) else int(x))
    df_tmp = data.groupby(['store_id'], as_index=False)['visit_time'].agg({'store_max_time':'max', 'store_min_time':'min'})

    return df_tmp


def genre_sales_time(data):
# 時間帯を抽出
    data['visit_time'] = data['visit_datetime'].str[11:13]
#***********そもそも12時間表記と24時間表記が混在してないかチェックが必要
# 00:00~06:00は24:00を加える
    data['visit_time'] = data['visit_time'].map(lambda x: 24+int(x[1]) if x[0]=='0' and int(x[1])<7 else int(x[1]) if (x[0]=='0' and int(x[1])>=7) else int(x))
    df_tmp = data.groupby(['air_genre_name'], as_index=False)['visit_time'].agg({'store_max_time':'max', 'store_min_time':'min'})

    return df_tmp




#  tmp = store_agg(df_train, df_train.visit_date.max(), 1000)
#  tmp = store_hlf_agg(df_train, df_train.visit_date.max(), 1000)
#  tmp = store_dow_agg(df_train, df_train.visit_date.max(), 1000)
#  tmp = store_weight_mean(df_train, df_train.visit_date.max(), 1000)
#  tmp = store_dow_weight_mean(df_train, df_train.visit_date.max(), 1000)
#  tmp = store_before_after_holiday_agg(df_train, df_train.visit_date.max(), 1000)
#  tmp = genre_weight_mean(df_train, df_train.visit_date.max(), 1000)
#  tmp = genre_dow_weight_mean(df_train, df_train.visit_date.max(), 1000)
#  tmp = store_sales_time(df_airR)
tmp = genre_sales_time(df_airR)

print(tmp)
