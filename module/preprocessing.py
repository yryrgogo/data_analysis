import pandas as pd
import numpy as np
import sys, re, glob
import gc
from sklearn.model_selection import StratifiedKFold
from load_data import pararell_load_data


unique_id = 'SK_ID_CURR'
seed = 1208


""" 日時操作系 """
def date_diff(start, end):
    diff = end - start
    return diff


def date_range(data, start, end, include_flg=1):
    '''
    include_flgが0の場合, endの日付は含めずにデータを返す
    '''
    if include_flg == 0:
        return data[(start <= data['visit_date']) & (data['visit_date'] < end)]
    return data[(start <= data['visit_date']) & (data['visit_date'] <= end)]


""" 前処理系 """
def outlier(data, level, value, out_range=1.64):
    '''
    Explain:
    Args:
        data(DF)        : 外れ値を除外したいデータフレーム
        level(str)      : 標準偏差を計算する粒度
        value(float)    : 標準偏差を計算する値
        out_range(float): 外れ値とするZ値の範囲.初期値は1.64としている
    Return:
        data(DF): 入力データフレームから外れ値を外したもの
    '''
    tmp = data.groupby(level, as_index=False)[value].agg(
        {'avg': 'mean',
         'std': 'std'
         })
    df = data.merge(tmp, on=level, how='inner')
    param = df[value].values
    avg = df['avg'].values
    std = df['std'].values
    z_value = (param - avg)/std
    df['z'] = z_value

    inner = df[-1*out_range <= df['z']].copy()
    inner = inner[inner['z'] <= out_range]

    out_minus = df[-1*out_range > df['z']]
    minus_max = out_minus.groupby(level, as_index=False)[value].max()
    out_minus.drop(value, axis=1, inplace=True)
    out_minus = out_minus.merge(minus_max, on=level, how='inner')

    out_plus = df[df['z'] > out_range]
    plus_min = out_plus.groupby(level, as_index=False)[value].min()
    out_plus.drop(value, axis=1, inplace=True)
    out_plus = out_plus.merge(plus_min, on=level, how='inner')

    result = pd.concat([inner, out_minus, out_plus], axis=0)
    result.drop(['avg', 'std', 'z'], axis=1, inplace=True)
    result.reset_index(drop=True, inplace=True)

    del df, out_minus, out_plus
    gc.collect()

    return result


def impute_avg(data=None, unique_id=None, level=None, index=1, value=None):
    '''
    Explain:
        平均値で欠損値補完を行う
    Args:
        data(DF)       : NULLを含み、欠損値補完を行うデータ
        level(list)    : 集計を行う粒度。最終的に欠損値補完を行う粒度が1カラム
                         でなく、複数カラム必要な時はリストで渡す。
                         ただし、欠損値補完の集計を行う粒度がリストと異なる場合、
                         次のindex変数にリストのうち集計に使うカラム数を入力する
                         (順番注意)
        index(int)     : 欠損値補完の際に集計する粒度カラム数
        value(float)   : 欠損値補完する値のカラム名
    Return:
        result(DF): 欠損値補完が完了したデータ
    '''

    ' 元データとの紐付けをミスらない様にインデックスをセット '
    data.set_index(unique_id, inplace=True)

    ' Null埋めする為、level粒度の平均値を取得 '
    use_cols = level + [value]
    data = data[use_cols]
    imp_avg = data.groupby(level, as_index=False)[value].mean()

    ' 平均値でNull埋め '
    null = data[data[value].isnull()]
    null = null.reset_index()
    fill_null = null.merge(imp_avg, on=level[:index], how='inner')

    ' インデックスをカラムに戻して、Null埋めしたDFとconcat '
    data = data[data[value].dropna()].reset_index()
    result = pd.concat([data, fill_null], axis=0)

    return result


def lag_feature(data, value, lag, level=[]):
    '''
    Explain:
        時系列データにおいて、リーケージとなる特徴量を集計する際、shiftによって
        リーケージを避け、ラグ特徴量を作成する
    Args:
        data(DF)    : valueやlevelを含んだデータフレーム
        value(float): ラグをとる特徴量。主にリークになる様な特徴量
        lag(int)    : shiftによってずらす行の数
                     （正：前の行数分データをとる、 負：後の行数分データをとる）
        level(list) : 粒度を指定してラグをとる場合、その粒度を入れたリスト。
                      このlevelでgroupbyをした上でラグをとる
    Return:
        data: 最初の入力データにラグをとった特徴カラムを付与して返す
    '''

    if len(level)==0:
        data[f'shift{lag}_{value}'] = data[value].shift(lag)
    else:
        data[f'shift{lag}_{value}@{level}'] = data.groupby(level)[value].shift(lag)

    return data


def dframe_dtype(data):
    for col in data.columns:
        print(data[col].dtype)


#  カテゴリ変数を取得する関数
def get_categorical_features(data, ignore):
    obj = [col for col in list(data.columns) if data[col].dtype == 'object' and col not in ignore]
    return obj


#  連続値カラムを取得する関数
def get_numeric_features(data, ignore):
    num = [col for col in list(data.columns) if (str(data[col].dtype).count('int') or str(data[col].dtype).count('float')) and col not in ignore]
    return num


# カテゴリ変数をファクトライズ (整数に置換)する関数
def factorize_categoricals(data, cats):
    for col in cats:
        data[col], _ = pd.factorize(data[col])
    return data


# カテゴリ変数のダミー変数 (二値変数化)を作成する関数
def get_dummies(data, cats, drop=1):
    input_col = data.columns
    for col in cats:
        data = pd.concat([data, pd.get_dummies(data[col], prefix=col)], axis=1)
    if drop==1:
        data = data.drop(input_col, axis=1)
    return data


def split_dataset(dataset, val_no):
    """
    時系列用のtrain, testデータを切る。validation_noを受け取り、
    データセットにおいてその番号をもつ行をTestデータ。その番号を
    もたない行をTrainデータとする。

    Args:
        dataset(DF): TrainとTestに分けたいデータセット
        val_no(int): 時系列においてleakが発生しない様にデータセットを
                     切り分ける為の番号。これをもとにデータを切り分ける

    Return:
        train(df): 学習用データフレーム(validationカラムはdrop)
        test(df) : 検証用データフレーム(validationカラムはdrop)
    """

    train = dataset[dataset['valid_no'] != val_no].copy()
    test = dataset[dataset['valid_no'] == val_no].copy()

    train.drop(['valid_no'], axis=1, inplace=True)
    test.drop(['valid_no'], axis=1, inplace=True)

    return train, test


def set_validation(data, target, holdout_flg=0):
    '''
    Explain:
        データセットにvalidation番号を振る。繰り返し検証を行う際、
        validationを固定したいのでカラムにする。
    Args:
    Return:
    '''
    #  start_date = pd.to_datetime('2017-03-12')
    #  end_date = pd.to_datetime('2017-04-22')
    #  data['validation'] = data['visit_date'].map(lambda x: 1 if start_date <= x and x <= end_date else 0)

    if holdout_flg==1:
        ' 全体をStratifiedKFoldで8:2に切って、8をCV.2をHoldoutで保存する '

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        x = data.drop(target, axis=1)
        y = data[target].values

        for trn_idx, val_idx in cv.split(x, y):
            data.iloc[trn_idx].to_csv('../data/cv_app_train.csv', index=False)
            data.iloc[val_idx].to_csv('../data/holdout_app_train.csv', index=False)
            sys.exit()

    else:
        ' データ4分割してvalidation番号をつける '
        cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed)
        data = data[[unique_id, target]]
        x = data[unique_id].to_frame()
        y = data[target].values
        cnt=0

        for trn_idx, val_idx in cv.split(x, y):
            cnt+=1

            valid_no = np.zeros(len(val_idx))+cnt
            tmp = pd.DataFrame({'index':val_idx, 'valid_no':valid_no})

            if cnt==1:
                tmp_result = tmp
            else:
                tmp_result = pd.concat([tmp_result, tmp], axis=0)

        tmp_result.set_index('index', inplace=True)

        result = data.join(tmp_result)
        result.drop(target, axis=1, inplace=True)
        print(result.shape)
        print(result.head())

    return result


def squeeze_target(data, level, size):
    '''
    levelの各要素について、一定数以上のデータがある要素の行のみ残す
    '''

    tmp = data.groupby(level).size()
    target_id = tmp[tmp >= size].index

    data = data.set_index(level)
    result = data.loc[target_id, :]
    result = result.reset_index()
    return result


def make_feature_set(dataset, path):

    '''
    Explain:
        pathに入ってるfeatureをdatasetにmerge.現状はnpy対応
    Args:
    Return:
    '''

    use_feature = glob.glob(path)
    p_list = pararell_load_data(use_feature, 0)
    df_use = pd.concat(p_list, axis=1)

    dataset = pd.concat([dataset, df_use], axis=1)
    return dataset


def row_number(df, level):
    '''
    Explain:
        各level粒度毎にrow_numberをつける。
        順番は入力されたDFのままなので、ソートが必要な場合は事前に。
    Args:
    Return:
    '''
    index = np.arange(1, len(df)+1, 1)
    df['index'] = index
    min_index = df.groupby(level)['index'].min().reset_index()
    df = df.merge(min_index, on=level, how='inner')
    df['index'] = df['index_x'] - df['index_y'] + 1
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


