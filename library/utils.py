"""
gsutil -m cp ../feature/* gs://homecredit_ko
gsutil -m rsync -d -r ../feature gs://homecredit_ko
"""
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from glob import glob
import os
import subprocess
import sys
from socket import gethostname
HOSTNAME = gethostname()

from tqdm import tqdm
from sklearn.model_selection import KFold
from time import time, sleep
from datetime import datetime
import gc
import pickle
import gzip
from multiprocessing import Pool
import multiprocessing

from logging import StreamHandler, DEBUG, Formatter, FileHandler, getLogger


# =============================================================================
# global variables
# =============================================================================
COMPETITION_NAME = 'home-credit-default-risk'

# =============================================================================
# def
# =============================================================================
def start(fname):
    global st_time
    st_time = time()
    print(f"""
#==============================================================================
# START!!! {fname}    PID: {os.getpid()}    time: {datetime.today()}
#==============================================================================
""")
    send_line(f'{HOSTNAME}  START {fname}  time: {elapsed_minute():.2f}min')
    return


def reset_time():
    global st_time
    st_time = time()
    return


def end(fname):
    print(f"""
#==============================================================================
# SUCCESS !!! {fname}
#==============================================================================
""")
    print('time: {:.2f}min'.format(elapsed_minute()))
    send_line(f'{HOSTNAME}  FINISH {fname}  time: {elapsed_minute():.2f}min')
    return


def elapsed_minute():
    return (time() - st_time)/60


def to_feather(df, path):

    if df.columns.duplicated().sum() > 0:
        raise Exception(
            f'duplicated!: { df.columns[df.columns.duplicated()] }')
    df.reset_index(inplace=True, drop=True)
    df.columns = [c.replace('/', '-').replace(' ', '-') for c in df.columns]
    for c in df.columns:
        df[[c]].to_feather(f'{path}_{c}.f')
    return


def to_pickle(path, obj):
    with open(path, 'wb') as f:
        pickle.dump(obj=obj, file=f)
        print(f"""
#==============================================================================
# PICKLE TO SUCCESS !!! {path}
#==============================================================================
""")


def read_pickle(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
        print(f"""
#==============================================================================
# PICKLE READ SUCCESS !!! {path}
#==============================================================================
""")
        return obj


def to_pkl_gzip(obj, path):
    #  df.to_pickle(path)
    with open(path, 'wb') as f:
        pickle.dump(obj=obj, file=f)
    os.system('gzip -f ' + path)
    os.system('rm ' + path)
    return


def read_pkl_gzip(path):
    with gzip.open(path, mode='rb') as fp:
        data = fp.read()
    return pickle.loads(data)


def to_df_pickle(df, path, fname='', split_size=3, index=False):
    """
    path = '../output/mydf'

    write '../output/mydf/0.p'
          '../output/mydf/1.p'
          '../output/mydf/2.p'

    """
    print(f'shape: {df.shape}')

    if index:
        df.reset_index(drop=True, inplace=True)
        gc.collect()
    mkdir_p(path)

    ' データセットを切り分けて保存しておく '
    kf = KFold(n_splits=split_size)
    for i, (train_index, val_index) in enumerate(tqdm(kf.split(df))):
        df.iloc[val_index].to_pickle(f'{path}/{fname}{i:03d}.p')
    return


def read_df_pickle(path, col=None, use_tqdm=True):
    if col is None:
        if use_tqdm:
            df = pd.concat([pd.read_pickle(f)
                            for f in tqdm(sorted(glob(path)))])
        else:
            print(f'reading {path}')
            df = pd.concat([pd.read_pickle(f)
                            for f in sorted(glob(path))])
    else:
        df = pd.concat([pd.read_pickle(f)[col]
                        for f in tqdm(sorted(glob(path)))])
    return df

# def to_feathers(df, path, split_size=3, inplace=True):
#    """
#    path = '../output/mydf'
#
#    wirte '../output/mydf/0.f'
#          '../output/mydf/1.f'
#          '../output/mydf/2.f'
#
#    """
#    if inplace==True:
#        df.reset_index(drop=True, inplace=True)
#    else:
#        df = df.reset_index(drop=True)
#    gc.collect()
#    mkdir_p(path)
#
#    kf = KFold(n_splits=split_size)
#    for i, (train_index, val_index) in enumerate(tqdm(kf.split(df))):
#        df.iloc[val_index].to_feather(f'{path}/{i:03d}.f')
#    return
#
# def read_feathers(path, col=None):
#    if col is None:
#        df = pd.concat([pd.read_feather(f) for f in tqdm(sorted(glob(path+'/*')))])
#    else:
#        df = pd.concat([pd.read_feather(f)[col] for f in tqdm(sorted(glob(path+'/*')))])
#    return df


def load_train(col=None):
    if col is None:
        return read_pickles('../data/train')
    else:
        return read_pickles('../data/train', col)


def load_test(col=None):
    if col is None:
        return read_pickles('../data/test')
    else:
        return read_pickles('../data/test', col)


def merge(df, col):
    trte = pd.concat([load_train(col=col),  # .drop('TARGET', axis=1),
                      load_test(col=col)])
    df_ = pd.merge(df, trte, on='SK_ID_CURR', how='left')
    return df_


def check_feature():

    sw = False
    files = sorted(glob('../feature/train*.f'))
    for f in files:
        path = f.replace('train_', 'test_')
        if not os.path.isfile(path):
            print(f)
            sw = True

    files = sorted(glob('../feature/test*.f'))
    for f in files:
        path = f.replace('test_', 'train_')
        if not os.path.isfile(path):
            print(f)
            sw = True

    if sw:
        raise Exception('Miising file :(')
    else:
        print('All files exist :)')

# =============================================================================
#
# =============================================================================


def reduce_mem_usage(df):
    col_int8 = []
    col_int16 = []
    col_int32 = []
    col_int64 = []
    col_float16 = []
    col_float32 = []
    col_float64 = []
    col_cat = []
    for c in tqdm(df.columns, mininterval=20):
        col_type = df[c].dtype

        if col_type != object:
            c_min = df[c].min()
            c_max = df[c].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    col_int8.append(c)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    col_int16.append(c)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    col_int32.append(c)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    col_int64.append(c)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    col_float16.append(c)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    col_float32.append(c)
                else:
                    col_float64.append(c)
        else:
            col_cat.append(c)

    if len(col_int8) > 0:
        df[col_int8] = df[col_int8].astype(np.int8)
    if len(col_int16) > 0:
        df[col_int16] = df[col_int16].astype(np.int16)
    if len(col_int32) > 0:
        df[col_int32] = df[col_int32].astype(np.int32)
    if len(col_int64) > 0:
        df[col_int64] = df[col_int64].astype(np.int64)
    if len(col_float16) > 0:
        df[col_float16] = df[col_float16].astype(np.float16)
    if len(col_float32) > 0:
        df[col_float32] = df[col_float32].astype(np.float32)
    if len(col_float64) > 0:
        df[col_float64] = df[col_float64].astype(np.float64)
    if len(col_cat) > 0:
        df[col_cat] = df[col_cat].astype('category')


def check_var(df, var_limit=0, sample_size=None):
    if sample_size is not None:
        if df.shape[0] > sample_size:
            df_ = df.sample(sample_size, random_state=71)
        else:
            df_ = df
#            raise Exception(f'df:{df.shape[0]} <= sample_size:{sample_size}')
    else:
        df_ = df

    var = df_.var()
    col_var0 = var[var <= var_limit].index
    if len(col_var0) > 0:
        print(f'remove var<={var_limit}: {col_var0}')
    return col_var0


def check_corr(df, corr_limit=1, sample_size=None):
    if sample_size is not None:
        if df.shape[0] > sample_size:
            df_ = df.sample(sample_size, random_state=71)
        else:
            raise Exception(f'df:{df.shape[0]} <= sample_size:{sample_size}')
    else:
        df_ = df

    corr = df_.corr('pearson').abs()  # pearson or spearman
    a, b = np.where(corr >= corr_limit)
    col_corr1 = []
    for a_, b_ in zip(a, b):
        if a_ != b_ and a_ not in col_corr1:
            #            print(a_, b_)
            col_corr1.append(b_)
    if len(col_corr1) > 0:
        col_corr1 = df.iloc[:, col_corr1].columns
        print(f'remove corr>={corr_limit}: {col_corr1}')
    return col_corr1


def remove_feature(df, var_limit=0, corr_limit=1, sample_size=None, only_var=True):
    col_var0 = check_var(df,  var_limit=var_limit, sample_size=sample_size)
    df.drop(col_var0, axis=1, inplace=True)
    if only_var == False:
        col_corr1 = check_corr(df, corr_limit=corr_limit,
                               sample_size=sample_size)
        df.drop(col_corr1, axis=1, inplace=True)
    return


def __get_use_files__():

    return


def get_use_files(prefixes=[], is_train=True):

    unused_files = [f.split('/')[-1]
                    for f in sorted(glob('../feature_unused/*.f'))]
    unused_files += [f.split('/')[-1]
                     for f in sorted(glob('../feature_var0/*.f'))]
    unused_files += [f.split('/')[-1]
                     for f in sorted(glob('../feature_corr1/*.f'))]

    if is_train:
        all_files = sorted(glob('../feature/train*.f'))
        unused_files = ['../feature/train_'+f for f in unused_files]
    else:
        all_files = sorted(glob('../feature/test*.f'))
        unused_files = ['../feature/test_'+f for f in unused_files]

    if len(prefixes) > 0:
        use_files = []
        for prefix in prefixes:
            use_files += glob(f'../feature/*{prefix}*')
        all_files = (set(all_files) & set(use_files)) - set(unused_files)

    else:
        for f in unused_files:
            if f in all_files:
                all_files.remove(f)

    all_files = sorted(all_files)

    print(f'got {len(all_files)}')
    return all_files


# =============================================================================
# other API
# =============================================================================
def submit(file_path, comment='from API'):
    os.system(f'kaggle competitions submit -c {COMPETITION_NAME} -f {file_path} -m "{comment}"')
    sleep(20)  # tekito~~~~
    tmp = os.popen(f'kaggle competitions submissions -c {COMPETITION_NAME} -v | head -n 2').read()
    col, values = tmp.strip().split('\n')
    message = 'SCORE!!!\n'
    for i, j in zip(col.split(','), values.split(',')):
        message += f'{i}: {j}\n'
        print(f'{i}: {j}') # TODO: comment out later?
    #  send_line(message.rstrip())


import requests


def send_line(message):

    line_notify_token = '5p5sPTY7PrQaB8Wnwp6aadfiqC8m2zh6Q8llrfNisGT'
    line_notify_api = 'https://notify-api.line.me/api/notify'

    payload = {'message': message}
    headers = {'Authorization': 'Bearer ' + line_notify_token}
    requests.post(line_notify_api, data=payload, headers=headers)


def stop_instance():
    """
    You need to login first.
    >> gcloud auth login
    """
    send_line('stop instance')
    os.system(
        f'gcloud compute instances stop {os.uname()[1]} --zone us-east1-b')


def mkdir_p(path):
    try:
        os.stat(path)
    except:
        os.mkdir(path)

def logger_func():
    logger = getLogger(__name__)
    log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s]\
    [%(funcName)s] %(message)s ')
    handler = StreamHandler()
    handler.setLevel('INFO')
    handler.setFormatter(log_fmt)
    logger.addHandler(handler)

    mkdir_p('../output')
    handler = FileHandler('../output/py_train.py.log', 'a')
    handler.setLevel(DEBUG)
    handler.setFormatter(log_fmt)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)

    logger.info('start')

    return logger


def x_y_split(data, target):
    x = data.drop(target, axis=1)
    y = data[target].values
    return x, y


def load_file(path):
    if path.count('.csv'):
        return pd.read_csv(path)
    elif path.count('.npy'):
        filename = re.search(r'/([^/.]*).npy', path).group(1)
        tmp = pd.Series(np.load(path), name=filename)
        return tmp


def pararell_load_data(path_list):
    p = Pool(multiprocessing.cpu_count())
    p_list = p.map(load_file, path_list)
    p.close

    return p_list


def path_info(path):

    path_dict = {}
    path_dict['filename'] = re.search(r'/([^/.]*).csv', path).group(1)  # Linux
    path_dict['particle'] = re.search(r'feature_([0-9]+)@', path).group(1)
    path_dict['time'] = re.search(r'@([^.]*)@', path).group(1)
    path_dict['elem'] = re.search(r'\D@([^.]*).csv', path).group(1)

    return path_dict


" 並列処理 "
def pararell_process(func, arg_list):
    process = Pool(multiprocessing.cpu_count())
    #  p = Pool(len(arg_list))
    callback = process.map(func, arg_list)
    process.close()
    process.terminate()
    return callback


" 機械学習でよく使う系 "
def row_number(df, level):
    '''
    Explain:
        levelをpartisionとしてrow_numberをつける。
        順番は入力されたDFのままで行う、ソートが必要な場合は事前に。
    Args:
    Return:
    '''
    index = np.arange(1, len(df)+1, 1)
    df['index'] = index
    min_index = df.groupby(level)['index'].min().reset_index()
    df = df.merge(min_index, on=level, how='inner')
    df['row_no'] = df['index_x'] - df['index_y'] + 1
    df.drop(['index_x', 'index_y'], axis=1, inplace=True)
    return df


#  カテゴリ変数を取得する関数
def get_categorical_features(df, ignore_list=[]):
    obj = [col for col in list(df.columns) if (df[col].dtype == 'object') and col not in ignore_list]
    return obj

#  カテゴリ変数を取得する関数
def get_datetime_features(df, ignore_list=[]):
    dt = [col for col in list(df.columns) if str(df[col].dtype).count('time') and col not in ignore_list]
    return dt

#  連続値カラムを取得する関数
def get_numeric_features(df, ignore_list=[]):
    num = [col for col in list(df.columns) if (str(df[col].dtype).count('int') or str(df[col].dtype).count('float')) and col not in ignore_list]
    return num


def round_size(value, max_val, min_val):
    range_val = int(max_val - min_val)
    origin_size = len(str(value))
    dtype = str(type(value))
    if dtype.count('float') or dtype.count('decimal'):
        origin_num = str(value)
        str_num = str(int(value))
        int_size = len(str_num)
        d_size = origin_size - int_size -1
        if int_size>1:
            if origin_num[-1]=='0':
                if range_val>10:
                    dec_size=0
                else:
                    dec_size=2
            else:
                dec_size = 1
        else:
            if d_size>3:
                dec_size = 3
            else:
                dec_size = 2
    elif dtype.count('int'):
            dec_size = 0
    else:
        raise ValueError('date type is int or float or decimal.')

    return dec_size
