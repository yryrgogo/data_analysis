import pandas as pd
import numpy as np
import sys
from tqdm import tqdm
import gc


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
