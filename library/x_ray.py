import numpy as np
import pandas as pd
from scipy.stats.mstats import mquantiles
import datetime
from time import sleep
from tqdm import tqdm
import sys
import re
import gc
import glob
import os
HOME = os.path.expanduser('~')
sys.path.append(f"{HOME}/kaggle/github/library/")
import utils
from utils import logger_func, get_categorical_features, get_numeric_features, pararell_process, round_size
sys.path.append(f"{HOME}/kaggle/github/model/")
from Estimator import cross_validation, data_check
import pickle
logger = logger_func()
pd.set_option('max_columns', 200)
pd.set_option('max_rows', 200)
start_time = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())

#========================================================================
# Global Variables 
#========================================================================
global train # For Pararell Processing
key = 'c_取引先集約コード'
target = '翌年トラコンタ購買フラグ__t'
ignore_list = [ 'c_取引先集約コード', 't_年月', target]
key_cols = [ 'c_取引先集約コード' ,'t_年月' ]
eno_code = 'cp_営農タイプ'
model_code = 'div_ターゲット'


def x_ray_caliculation(col, val, model):
    train[col] = val
    pred = model.predict(train)
    p_avg = np.mean(pred)
    return col, val, p_avg

def x_ray_wrapper(args):
    return x_ray_caliculation(*args)

def x_ray(logger, model, train, columns=False, max_sample=50):
    '''
    Explain:
    Args:
        columns: x-rayを出力したいカラムリスト
    Return:
    '''
    x_ray = False
    result = pd.DataFrame([])
    if not(columns):
        columns = train.columns
    for i, col in enumerate(columns):
        xray_list = []

        #========================================================================
        # MAKE X-RAY GET POINT
        #========================================================================
        val_cnt = train[col].value_counts()
        if len(val_cnt)>max_sample:
            length = max_sample-10
            val_array = val_cnt.head(length).index.values
            percentiles = np.linspace(0.05,0.95,num=10)
            val_percentiles = mquantiles(val_cnt.index.values, prob=percentiles, axis=0)
            max_val = train[col].max()
            min_val = train[col].min()
            r = round_size(max_val, max_val, min_val)
            val_percentiles = np.round(val_percentiles, r)
            val_array = np.hstack((val_array, val_percentiles))
        else:
            length = len(val_cnt)
            val_array = val_cnt.head(length).index.values
        val_array = np.sort(val_array)

        #========================================================================
        # PARARELL PROCESSING READY & START
        #========================================================================
        arg_list = []
        for val in val_array:
            arg_list.append([col, val, model])

        logger.info(f'''
#========================================================================
# X-RAY CALICURATION START : {col}
#========================================================================''')

        xray_values = pararell_process(x_ray_wrapper, arg_list)
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

        tmp_result = pd.DataFrame(data=result_dict)
        if len(result):
            result = pd.concat([result, tmp_result], axis=0)
            logger.info(f'''
#========================================================================
# {i+1}/len(columns) FEATURE. CURRENT RESULT SHAPE : {result.shape}
#========================================================================''')
        else:
            result = tmp_result.copy()
            logger.info(f'''
#========================================================================
# {i+1}/len(columns) FEATURE. CURRENT RESULT SHAPE : {result.shape}
#========================================================================''')

    return result


def xray_main(df, suffix):
    global train

    metric = ['auc'][0]
    num_iterations = 3500
    learning_rate = 0.05
    early_stopping_rounds = 150
    model_type = 'lgb'
    fold_type = 'group'
    fold = 5

    params = {'bagging_freq': 1, 'bagging_seed': 1012, 'colsample_bytree': 1.0, 'data_random_seed': 1012, 'feature_fraction_seed': 1012, 'lambda_l1': 0.1, 'lambda_l2': 0.5, 'learning_rate': 0.02, 'max_bin': 250, 'max_depth': 5, 'metric': 'auc', 'min_child_samples': 96, 'min_child_weight': 36, 'min_data_in_bin': 96, 'min_split_gain': 0.01, 'num_leaves': 11, 'num_threads': 35, 'objective': 'binary', 'random_seed': 1012, 'subsample': 1.0}

    ' TEST用 '
    #  nrows = 1000
    #  df = pd.read_csv('../input/20180914_yanmar_drset_10model.csv', nrows=nrows)

    df.set_index(key, inplace=True)

    feature_cols = [col for col in df.columns if col.count('__') or col.count('担い手') or col.count('経過')]
    #  feature_cols = [col for col in df.columns if col.count('__') and col not in ignore_list]
    train = df[feature_cols]
    categorical_list = get_categorical_features(df=train, ignore_list=ignore_list) # For categorical decode

    result, col_length, model_list, df_cat_decode = cross_validation(
        logger=logger,
        train=train,
        target=target,
        params=params,
        metric=metric,
        fold_type=fold_type,
        num_iterations=num_iterations,
        learning_rate=learning_rate,
        early_stopping_rounds=early_stopping_rounds,
        model_type=model_type,
        ignore_list=ignore_list,
        x_ray=True
    )

    # 念のためモデルとカテゴリのdecode_mapを保存しておく
    for i, model in enumerate(model_list):
        with open(f'../output/model_{i}@{suffix}.pickle', 'wb') as f:
            pickle.dump(obj=model, file=f)
    df_cat_decode.to_csv(f'../output/df_cat_decode@{suffix}.csv', index=False)
    return

    ' xray params '
    model_list = []
    max_sample = 50
    for i in range(fold):
        with open(f'../output/model_{i}.pickle', 'rb') as f:
            model = pickle.load(f)
            model_list.append(model)

    train, _ = data_check(logger, df=train, target=target)

    result = pd.DataFrame([])
    for i, model in enumerate(model_list):
        tmp_result = x_ray(logger, model, train)
        tmp_result.rename(columns = {'xray': f'x_ray_{i+1}'}, inplace=True)

        if len(result):
            result = result.merge(tmp_result, on=['feature', 'value'], how='inner')
            logger.info(f'''
#========================================================================
# CURRENT RESULT SHAPE {i+1}/{len(model_list)}  : {result.shape}
#========================================================================''')
        else:
            result = tmp_result.copy()
        logger.info(f'''
#========================================================================
# CURRENT RESULT SHAPE {i+1}/{len(model_list)}  : {result.shape}
#========================================================================''')
        break

    #  df_cat_decode = pd.read_csv('../output/20180918_0747_df_cat_decode.csv')
    for cat in categorical_list:
        cat_cols = [col for col in df_cat_decode.columns if col.count(cat)]
        decode_dict = df_cat_decode[cat_cols].drop_duplicates().set_index(f'{cat}').to_dict()[f"origin_{cat}"]
        tmp = result.query(f"feature=='{cat}'")
        tmp['value'] = tmp['value'].map(decode_dict)
        #  print(decode_dict)
        #  print(tmp)
        #  sys.exit()

        tmp_result = result.query(f"feature!='{cat}'")
        result = pd.concat([tmp, tmp_result], axis=0)

    result.to_csv(f"../output/{start_time[:12]}_xray_test1000_{suffix}.csv", index=False)


if __name__ == '__main__':

    #  df = pd.read_csv('../input/20180914_yanmar_drset_10model.csv', nrows=100)
    df = pd.read_csv('../input/20180918_yanmar_dr_16model_add_eino.csv')
    model_code_list = df[model_code].drop_duplicates().values
    eno_code_list = df[eno_code].drop_duplicates().values
    base_cols = [col for col in df.columns if not(col.count('__')) or col.count('担い手') or col.count('経過')]
    diary_cols = [col for col in df.columns if col.count('__d') or col.count('担い手') or col.count('経過')]
    sales_cols = [col for col in df.columns if col.count('__sf') or col.count('担い手') or col.count('経過')]
    diary_cols += [key, target]
    sales_cols += [key, target]
    feature_set_list = [diary_cols, sales_cols]

    for mc in model_code_list:
        tmp_tmp_df = df.query(f"{model_code}=='{mc}'")
        #  xray_main(tmp_tmp_df, suffix='')
        #  sys.exit()
        for ec in eno_code_list:
            for feature_set in feature_set_list:
                tmp_df = tmp_tmp_df.query(f"{eno_code}=='{ec}'")[feature_set]
                if len(tmp_df[target].drop_duplicates())==0:
                    continue
                for col in feature_set:
                    if col.count('__d'):
                        suffix = f'diary_div{mc}_{ec}'
                    elif col.count('__sf'):
                        suffix = f'sales_div{mc}_{ec}'

                for col in tmp_df.columns:
                    if col.count('経過') and not(col.count(str(mc))):
                        tmp_df.drop(col, axis=1, inplace=True)

                print(suffix)
                for col in tmp_df.columns:
                    print(col)
                #  xray_main(tmp_df, suffix=suffix)

