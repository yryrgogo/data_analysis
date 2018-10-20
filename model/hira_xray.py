import lightgbm
import sys
import os
HOME = os.path.expanduser('~')
sys.path.append(f"{HOME}/kaggle/github/library/")
import utils
from utils import logger_func, get_categorical_features, get_numeric_features, pararell_process, round_size, mkdir_p
import datetime
start_time = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())
import numpy as np
import pandas as pd
from scipy.stats.mstats import mquantiles
from tqdm import tqdm
import re
import gc
import glob
sys.path.append(f"{HOME}/kaggle/github/model/")
import utils
from Estimator import cross_validation, data_check, sc_metrics
import pickle
logger = logger_func()
pd.set_option('max_columns', 200)
pd.set_option('max_rows', 200)


def read_model(model_path_list, model_num):
    for path in model_path_list:
        if path.count(f'div{mc}') and path.count(ec) and path.count(mtype) and path.count(f'model_{model_num}'):
            with open(path, 'rb') as f:
                model = pickle.load(f)
                break
    return model

def read_score(feim_path_list, model_num):
    for path in model_path_list:
        if path.count(f'div{mc}') and path.count(ec) and path.count(mtype):
            cv_score = pd.read_csv(path)['cv_score'].values[0]
    return cv_score


def pararell_xray_caliculation(col, val, model_num):
    model_path_list = glob.glob(f'{model_dir}/*.pickle')
    model = read_model(model_path_list, model_num)
    train[col] = val
    pred = model.predict(train)
    del model
    gc.collect()
    p_avg = np.mean(pred)

    logger.info(f'''
#========================================================================
# CALICULATION PROGRESS... COLUMN: {col} | VALUE: {val} | X-RAY: {p_avg}
#========================================================================''')
    return col, val, p_avg

def single_xray_caliculation(col, val, model, df_xray):

    df_xray[col] = val
    df_xray.sort_index(axis=1, inplace=True)
    pred = model.predict(df_xray)
    gc.collect()
    p_avg = np.mean(pred)

    logger.info(f'''
#========================================================================
# CALICULATION PROGRESS... COLUMN: {col} | VALUE: {val} | X-RAY: {p_avg}
#========================================================================''')
    return col, val, p_avg


def pararell_xray_wrapper(args):
    return pararell_xray_caliculation(*args)

def x_ray(logger, model_num, train, col_list=[], max_point=20, ignore_list=[]):
    '''
    Explain:
    Args:
        col_list  : x-rayを出力したいカラムリスト
        max_point : x-rayを可視化するデータポイント数
        ex_feature: データポイントの取得方法が特殊なfeature
    Return:
    '''
    result = pd.DataFrame([])
    if len(col_list)==0:
        col_list = train.columns
    for i, col in enumerate(col_list):
        if col in ignore_list:
            continue
        xray_list = []

        #========================================================================
        # Get X-RAY Data Point
        # 1. 対象カラムの各値のサンプル数をカウントし、割合を算出。
        # 2. 全体においてサンプル数の少ない値は閾値で切ってX-RAYを算出しない
        #========================================================================
        # TODO: Numericはnuniqueが多くなるので、丸められるようにしたい
        val_cnt = train[col].value_counts().reset_index().rename(columns={'index':col, col:'cnt'})
        val_cnt['ratio'] = val_cnt['cnt']/len(train)

        ex_feature = 'elpse'
        if col.count(ex_feature) or len(val_cnt)<=15:
            threshold = 0
        else:
            threshold = 0.005
        val_cnt = val_cnt.query(f"ratio>={threshold}") # サンプル数の0.5%未満しか存在しない値は除く
        ' xray params '
        max_sample = 30

        # max_sampleよりnuniqueが大きい場合、max_sampleに取得ポイント数を絞る.
        # 合わせて10パーセンタイルをとり, 分布全体のポイントを最低限取得できるようにする
        if len(val_cnt)>max_sample:
            length = max_sample-10
            data_points = val_cnt.head(length).index.values
            percentiles = np.linspace(0.05, 0.95, num=10)
            val_percentiles = mquantiles(val_cnt.index.values, prob=percentiles, axis=0)
            max_val = train[col].max()
            min_val = train[col].min()
            # 小数点以下が大きい場合、第何位までを計算するか取得して丸める
            r = round_size(max_val, max_val, min_val)
            val_percentiles = np.round(val_percentiles, r)
            data_points = np.hstack((data_points, val_percentiles))
        else:
            length = len(val_cnt)
            data_points = val_cnt.head(length).index.values # indexにデータポイント, valueにサンプル数が入ってる
        data_points = np.sort(data_points)

        #========================================================================
        # 一番計算が重くなる部分
        # multi_processにするとprocess数分のデータセットをメモリに乗せる必要が
        # あり, Overheadがめちゃでかく大量のメモリを食う。また、各データポイントに
        # ついてpredictを行うので、毎回全CPUを使っての予測が行われる。
        # また、modelオブジェクトも重いらしく引数にmodelを載せると死ぬ
        # TODO DataFrameでなくnumpyでデータセットを保存してみる？
        #========================================================================
        if Pararell:
            arg_list = []

            train.sort_index(axis=1, inplace=True)
            # TODO numpyに変換
            # tmp_set = train~
            for point in data_points:
                arg_list.append([col, point, model_num])
            try: cpu_cnt
            except AttributeError: cpu_cnt=-1
            xray_values = pararell_process(pararell_xray_wrapper, arg_list, cpu_cnt=cpu_cnt)
        else:
            xray_values = []
            model_path_list = glob.glob(f'{model_dir}/*.pkl')
            model = read_model(model_path_list, model_num)
            df_xray = train.drop(target, axis=1)
            df_xray.sort_index(axis=1, inplace=True)
            # TODO numpyに変換
            # tmp_set = df_xray~
            for point in data_points:
                one_xray = single_xray_caliculation(col=col, val=point, model=model, df_xray=df_xray)
                xray_values.append(one_xray)

        del tmp_set
        gc.collect()

        #========================================================================
        # 計算済みの1つのfeatureのX-RAYの結果を統合する為の前準備
        #========================================================================
        feature_list = []
        value_list = []
        xray_list = []
        result_dict = {}
        for xray_tuple in xray_values:
            feature_list.append(xray_tuple[0])
            value_list.append(xray_tuple[1])
            xray_list.append(xray_tuple[2])
        result_dict = {
            'feature':feature_list,
            'value':value_list,
            'xray' :xray_list
        }
        #========================================================================
        # 各featureのX-RAYの結果を統合
        #========================================================================
        tmp_result = pd.DataFrame(data=result_dict)
        if len(result):
            result = pd.concat([result, tmp_result], axis=0)
        else:
            result = tmp_result.copy()

    return result


def xray_concat(df, suffix):
    fold = 5
    for model_num in range(1, fold+1, 1):
        #========================================================================
        # 1modelのx-rayを計算
        #========================================================================
        N = 300000
        train = df.sample(N, random_state=model_num)
        tmp_result = x_ray(logger, model_num, train)
        tmp_result.rename(columns = {'xray': f'x_ray_{i+1}'}, inplace=True)
        tmp_result.drop_duplicates(inplace=True)

        if len(result):
            result = result.merge(tmp_result, on=['feature', 'value'], how='inner')
        else:
            result = tmp_result.copy()

    xray_cols = [col for col in result.columns if col.count('x_ray_')]
    result['x_ray_avg'] = result[xray_cols].mean(axis=1)
