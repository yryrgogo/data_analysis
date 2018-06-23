import pandas as pd
import numpy as np
import sys


" 機械学習でよく使う系 "
#  カテゴリ変数を取得する関数
def get_categorical_features(data, ignore):
    obj = [col for col in list(data.columns) if data[col].dtype == 'object' and col not in ignore]
    return obj


#  連続値カラムを取得する関数
def get_numeric_features(data, ignore):
    num = [col for col in list(data.columns) if (str(data[col].dtype).count('int') or str(data[col].dtype).count('float')) and col not in ignore]
    return num


def row_number(df, col_name):
    '''
    Explain:
        col_nameをpartisionとしてrow_numberをつける。
        順番は入力されたDFのままで行う、ソートが必要な場合は事前に。
    Args:
    Return:
    '''
    index = np.arange(1, len(df)+1, 1)
    df['index'] = index
    min_index = df.groupby(col_name)['index'].min().reset_index()
    df = df.merge(min_index, on=col_name, how='inner')
    df['row_no'] = df['index_x'] - df['index_y'] + 1
    df.drop(['index_x', 'index_y'], axis=1, inplace=True)
    return df


" 並列処理 "
def pararell_process(func, arg_list):
    p = Pool(multiprocessing.cpu_count())
    p_list = p.map(func, arg_list)
    p.close
    return p_list


"  評価関数  "
def RMSLE(y_obs, y_pred):
    #  del_idx = np.arange(len(y_obs))[y_obs == 0]
    #  y_obs = np.delete(y_obs, del_idx)
    #  y_pred = np.delete(y_pred, del_idx)
    y_pred = y_pred.clip(min=0.)
    return np.sqrt(mean_squared_log_error(y_obs, y_pred))


' データをチェックする系 '
def col_heta_shape_cnt_check(data, col=0, heta=0, shape=0, count=0, nunique=0, all_check=0):

    if all_check=1:
        col=1
        part=1
        shape=1
        count=1
        nunique=1

    if col == 1:
        for col in data.columns:
            print(col)

    if part == 1:
        print(data.head())
        print(data.tail())

    if shape == 1:
        print(data.shape)

    if count == 1:
        print(data.count())

    if nunique == 1:
        print(data.nunique())


def dframe_dtype(data):
    ' データフレームのデータ型をカラム毎に確認する '
    for col in data.columns:
        print(f'column: {col} dtype: {data[col].dtype}')


def list_check(check_list):
    ' listの中身を一覧で '
    for ele in check_list:
        print(ele)
    sys.exit()


def null_check(data):
    for col in data.columns:
        print(col)
        print(len(data[data[col].isnull()]))
    sys.exit()


def main():
    print()
    #  move_feature()


if __name__ == '__main__':
    main()
