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
Train = [True, False][1]
Pararell = [True, False][0]
do_type = ['xray', 'pred_concat'][0]
eno_code_list = ['稲作', '畑作']
#  eno_code_list = ['None']

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


def read_model(model_path, model_num):
    for path in model_path:
        if path.count(f'div{mc}') and path.count(ec) and path.count(mtype) and path.count(f'model_{model_num}'):
            with open(path, 'rb') as f:
                model = pickle.load(f)
                break
    return model


def x_ray_caliculation(col, val, model_num):
    model_path = glob.glob('../output/20180918_yanmar_10model/*.pickle')
    model = read_model(model_path, model_num)
    df_xray = train.copy()
    df_xray[col] = val
    pred = model.predict(df_xray)
    del model, df_xray
    gc.collect()
    p_avg = np.mean(pred)

    logger.info(f'''
#========================================================================
# CALICULATION PROGRESS... COLUMN: {col} | VALUE: {val} | X-RAY: {p_avg}
#========================================================================''')

    return col, val, p_avg


def x_ray_wrapper(args):
    return x_ray_caliculation(*args)


def x_ray(logger, model_num, train, columns=False, max_sample=30):
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
        if col in ignore_list:
            continue
        xray_list = []

        #========================================================================
        # MAKE X-RAY GET POINT
        #========================================================================
        val_cnt = train[col].value_counts().reset_index().rename(columns={'index':col, col:'cnt'})
        val_cnt['ratio'] = val_cnt['cnt']/len(train)

        if col.count('経過') or len(val_cnt)<=15:
            threshold = 0
        else:
            threshold = 0.005
        val_cnt = val_cnt.query(f"ratio>={threshold}") # サンプル数の0.5%未満しか存在しない値は除く

        if len(val_cnt)>max_sample:
            length = max_sample-10
            val_array = val_cnt.head(length).index.values
            percentiles = np.linspace(0.05, 0.95, num=10)
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

        logger.info(f'''
#========================================================================
# MODEL DIVISION                 : {suffix}
# X-RAY CALICURATION START       : {col}
# X-RAY CALICURATION VALUE COUNT : {len(val_array)}
# MULTI PROCESSING               : {Pararell}
#========================================================================''')
        if Pararell:
            #========================================================================
            # MULTI PROCESSING READY & START
            #========================================================================
            arg_list = []

            for val in val_array:
                #  arg_list.append([col, val, model]) # 重すぎ
                arg_list.append([col, val, model_num])

            xray_values = pararell_process(x_ray_wrapper, arg_list)

        else:
            #========================================================================
            # SINGLE PROCESSING 
            #========================================================================
            xray_values = []
            for val in val_array:
                tmp_val = x_ray_caliculation(col=col, val=val, model=model)
                xray_values.append(tmp_val)

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
# {i+1}/{len(columns)} FEATURE. CURRENT RESULT SHAPE : {result.shape}
#========================================================================''')
        else:
            result = tmp_result.copy()
            logger.info(f'''
#========================================================================
# {i+1}/{len(columns)} FEATURE. CURRENT RESULT SHAPE : {result.shape}
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

    params = {'bagging_freq': 1,
              'bagging_seed': 1012,
              'feature_fraction_seed': 1012,
              'data_random_seed': 1012,
              'random_seed': 1012,
              'colsample_bytree': 1.0,
              'lambda_l1': 0.1,
              'lambda_l2': 0.5,
              'learning_rate': 0.02,
              'max_bin': 255,
              'max_depth': 4,
              'metric': 'auc',
              #  'min_child_samples': 96,
              #  'min_child_weight': 36,
              #  'min_data_in_bin': 96,
              'min_split_gain': 0.01,
              'num_leaves': 8,
              'num_threads': 35,
              'objective': 'binary',
              'subsample': 1.0}


    df.set_index(key, inplace=True)
    feature_cols = [col for col in df.columns if col.count('__') or col.count('担い手') or col.count('経過')]
    train = df[feature_cols]
    categorical_list = get_categorical_features(df=df, ignore_list=ignore_list) # For categorical decode

    if Train:

        cv_feim, col_length, model_list, df_cat_decode = cross_validation(
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
        cv_feim.to_csv(f'../output/feature_importance@{suffix}.csv', index=False)

        return

    ' xray params '
    max_sample = 30

    train, _ = data_check(logger, df=train, target=target)

    result = pd.DataFrame([])
    for i in range(fold):

        if do_type=='pred_concat':
            logger.info(f'''
#========================================================================
# PREDICTION CONCAT START
#========================================================================''')
            pred_result = prediciont_concat(i)
            return pred_result


        elif do_type=='xray':
            logger.info(f'''
#========================================================================
# X-RAY START
#========================================================================''')

            tmp_result = x_ray(logger, i, train)
            tmp_result.rename(columns = {'xray': f'x_ray_{i+1}'}, inplace=True)

            if len(result):
                result = result.merge(tmp_result, on=['feature', 'value'], how='inner')
                logger.info(f'''
#========================================================================
# CURRENT RESULT SHAPE {i+1}/{fold}  : {result.shape}
#========================================================================''')
            else:
                result = tmp_result.copy()
            logger.info(f'''
#========================================================================
# CURRENT RESULT SHAPE {i+1}/{fold}  : {result.shape}
#========================================================================''')

    #========================================================================
    # CATEGORICAL DECODE
    #========================================================================
    cat_decode_path = glob.glob('../output/20180918_yanmar_10model/df_cat_decode*.csv')
    #  cat_decode_path = glob.glob('../output/df_cat_decode*.csv')
    for path in cat_decode_path:
        if path.count(f'div{mc}') and path.count(ec) and path.count(mtype):
            df_cat_decode = pd.read_csv(path)
            break

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

    xray_cols = [col for col in result.columns if col.count('x_ray_')]
    result['x_ray_avg'] = result[xray_cols].mean(axis=1)
    result['data_div'] = suffix[:5]
    result['model_type'] = suffix
    result['eino_type'] = suffix[-2:]
    result['model_div'] = f'div_{mc}'

    #========================================================================
    # X-RAYの算出に失敗していたら中止する
    #========================================================================
    if len(result[result['x_ray_avg'].isnull()])>0:
        print(result)
        sys.exit()

    #========================================================================
    # RESULT SAVE
    #========================================================================
    result.to_csv(f"../output/{start_time[:12]}_yanmar_xray_{suffix}.csv", index=False)


def load_data(path):
    return pd.read_csv(path)

def xray_concat():
    path_key = '../output/*xray*.csv'
    path_list = glob.glob(path_key)

    #========================================================================
    # 中身チェック
    #  for path in path_list:
        #  if not(path.count('div2')):continue
        #  df = pd.read_csv(path)
        #  df.set_index('feature', inplace=True)
        #  index = [col for col in df.index if col.count('経過')]
        #  df = df.loc[index, :]
    #========================================================================

    func = load_data
    p_list = pararell_process(func, path_list)
    df = pd.concat(p_list, axis=0)

    feature_list = df['feature'].values
    type_list = []

    ' Categoricalは別シートにするため、分けられるようにする '
    for feature in feature_list:
        if feature.count('担い手') or feature.count('カテゴリ'):
            type_list.append('Category')
        elif feature.count('経過'):
            type_list.append('Elapsed_month')
        else:
            type_list.append('Numeric')
    df['feature_type'] = type_list

    ' Categoricalなfeature valueのカラムを分ける '
    df_cat = df.query("feature_type=='Category'").rename(columns={'value':'value_category'})
    df_num = df.query("feature_type!='Category'")
    df = pd.concat([df_cat, df_num], axis=0)

    df.to_csv(f'../output/{start_time[:12]}_yanmar_xray_10model.csv', index=False)


def prediciont_concat(model_num):
    model_path = glob.glob('../output/20180918_yanmar_10model/*.pickle')
    model = read_model(model_path, model_num)
    pred = model.predict(train)
    del model
    gc.collect()
    return pred


def importance_concat():
    #  path_list = glob.glob('../output/20180918_yanmar_10model/*importance*.csv')
    path_list = glob.glob('../output/20180918_yanmar_16model/*importance*.csv')
    feim = pd.DataFrame([])
    for path in path_list:
        tmp = pd.read_csv(path)
        use_cols = [col for col in tmp.columns if col.count('avg') or not(col.count('importance'))]
        tmp = tmp[use_cols]
        filename = re.search(r'@([^/.]*).csv', path).group(1)[:-5]
        tmp['model_div'] = filename[-4:]
        tmp['data_div'] = filename[:5]
        tmp['filename'] = filename
        tmp['avg_importance'] = tmp['avg_importance'] / tmp['avg_importance'].max()
        if len(feim):
            feim = pd.concat([feim, tmp], axis=0)
        else:
            feim = tmp.copy()
    feim.to_csv(f'../output/{start_time[:12]}_yanmar_feature_importance_score_16model.csv', index=False)


def Calicurate_Feature_Importance(col, model_num):

    # Model Load
    model_path = glob.glob('../output/20180918_yanmar_10model/*.pickle')
    model = read_model(model_path, model_num)

    seed_list = [1212, 1111, 2222]
    value_list = train[col].values
    for seed in seed_list:
        np.random.seed(seed)
        tmp_values = np.random.shuffle(value_list)

        pred = model.predict(train)
        del model
        gc.collect()
        p_avg = np.mean(pred)

    logger.info(f'''
#========================================================================
# CALICULATION PROGRESS... COLUMN: {col} | VALUE: {val} | X-RAY: {p_avg}
#========================================================================''')

    return col, val, p_avg


def x_ray_wrapper(args):
    return x_ray_caliculation(*args)




if __name__ == '__main__':

    #  importance_concat()
    #  sys.exit()

    #  xray_concat()
    #  sys.exit()

    logger.info('''
# DATA LOADING...''')
    #  df = pd.read_csv('../input/20180918_yanmar_dr_16model_add_eino.csv', nrows=10000)
    #  df = pd.read_csv('../input/20180918_yanmar_dr_16model_add_eino.csv')
    df = pd.read_csv('../input/20180918_yanmar_drset_10_model.csv')
    logger.info(f'''
#========================================================================
# DATA SHAPE : {df.shape}
#========================================================================''')
    model_code_list = df[model_code].drop_duplicates().values

    base_cols = [col for col in df.columns if not(col.count('__')) or col.count('担い手') or col.count('経過')]
    diary_cols = [col for col in df.columns if col.count('__co') or col.count('__d') or col.count('担い手') or col.count('経過')]
    sales_cols = [col for col in df.columns if col.count('__co') or col.count('__sf') or col.count('担い手') or col.count('経過')]

    diary_cols += [key, target]
    sales_cols += [key, target]
    feature_set_list = {'diary':diary_cols, 'sales':sales_cols}

    mc_df = pd.DataFrame([])
    for mc in tqdm(model_code_list):
        #========================================================================
        # FOR TEST
        #  if mc!=2:continue
        #========================================================================

        tmp_tmp_df = df.query(f"{model_code}=='{mc}'")

        ec_df = pd.DataFrame([])
        for ec in eno_code_list:

            #========================================================================
            # FOR TEST
            #  if ec!='稲作':continue
            #========================================================================

            mtype_df = pd.DataFrame([])
            for mtype, feature_set in feature_set_list.items():
                if len(eno_code_list)>=2:
                    tmp_df = tmp_tmp_df.query(f"{eno_code}=='{ec}'")
                else:
                    tmp_df = tmp_tmp_df

                ' For Predicion Concat RawTable '
                if do_type=='pred_concat':
                    base_df = tmp_df.copy()

                tmp_df = tmp_df[feature_set]

                for col in feature_set:
                    if col.count('__d'):
                        suffix = f'diary_div{mc}_{ec}'
                    elif col.count('__sf'):
                        suffix = f'sales_div{mc}_{ec}'

                for col in tmp_df.columns:
                    if col.count('__co') and not(col.count(str(mc))):
                        tmp_df.drop(col, axis=1, inplace=True)
                        logger.info(f'''
# DROP COLUMN : {col}''')

                #========================================================================
                # X-RAY SAMPLING ABOUT MAX300000~500000 (1000000 is too much)
                if do_type=='xray':
                    if len(tmp_df)>250000:
                        tmp_df = tmp_df.sample(250000)
                #========================================================================

                result_pred = xray_main(tmp_df, suffix=suffix)

                if do_type=='pred_concat':
                    if len(mtype_df):
                        mtype_df[f'Pred_{mtype}'] = result_pred
                    else:
                        mtype_df = base_df.copy()
                        mtype_df[f'Pred_{mtype}'] = result_pred
                    del base_df

                del tmp_df
                gc.collect()
            if do_type=='pred_concat':
                if len(ec_df):
                    ec_df = pd.concat([ec_df, mtype_df], axis=0)
                else:
                    ec_df = mtype_df.copy()
                del mtype_df
                gc.collect()

        if do_type=='pred_concat':
            if len(mc_df):
                mc_df = pd.concat([mc_df, ec_df], axis=0)
            else:
                mc_df = ec_df.copy()
            del ec_df
            gc.collect()

    if do_type=='pred_concat':
        mc_df.to_csv(f'../output/{start_time[:12]}_pred_concat_table.csv', index=False)
