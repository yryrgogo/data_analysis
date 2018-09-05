import pandas as pd
import numpy as np
import sys, re, glob
import gc
from sklearn.model_selection import StratifiedKFold
from utils import pararell_load_data
from multiprocessing import Pool
import multiprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler


sc = StandardScaler()
mm = MinMaxScaler()

""" 前処理系 """

' データセットを標準化、欠損値、無限値の中央値置き換え '
def data_regulize(df, na_flg=0, inf_flg=0, sc_flg=0, mm_flg=0, float16_flg=0, ignore_feature_list=[], logger=False):

    if inf_flg==1:
        df = inf_replace(data=df, logger=logger, ignore_feature_list=ignore_feature_list)
    if na_flg==1:
        df = impute_avg(data=df, logger=logger, ignore_feature_list=ignore_feature_list)

    ' 正規化 / 標準化 '
    if sc_flg==1:
        #  df[col] = sc.fit_transform(df[col])
        avg = df[col].mean()
        se = df[col].std()
        df[col] = (df[col] - avg) / se
    elif mm_flg==1:
        df = max_min_regularize(df, ignore_feature_list=ignore_feature_list, logger=logger)
    if float16_flg==1:
        df = df.astype('float16')

    return df


def impute_avg(data, logger=False, drop=False, ignore_feature_list=[]):
    for col in data.columns:
        if col in ignore_feature_list:continue
        if len(data[col][data[col].isnull()])>0:
            if drop:
                data.drop(col, axis=1, inplace=True)
                if logger:
                    logger.info(f'drop: {col}')
                continue
            data[col] = data[col].fillna(data[col].mean())
            if logger:
                logger.info(f'{col} impute length: {len(data[col][data[col].isnull()])}')

    return data


def inf_replace(data, logger=False, drop=False, ignore_feature_list=[]):
    for col in data.columns:
        if col in ignore_feature_list:continue

        ' count of inf '
        inf_plus = np.where(data[col].values == float('inf') )
        inf_minus = np.where(data[col].values == float('-inf') )
        logger.info(f'{col} >> inf count: {len(inf_plus)} | -inf count: {len(inf_minus)}')

        data[col].replace(np.inf, np.nan, inplace=True)
        data[col].replace(-1*np.inf, np.nan, inplace=True)
        logger.info(f'*****inf replace SUCCESS!!*****')
    return data

    #      for i in range(len(inf_plus[0])):
    #          logger.info(f'inf include: {col}')
    #          data[col].values[inf_plus[0][i]] = np.nan
    #          data[col].values[inf_minus[0][i]] = np.nan
    #          logger.info(f'-inf include: {col}')

    #  return data


def max_min_regularize(data, ignore_feature_list=[], logger=False):
    for col in data.columns:
        if col in ignore_feature_list:continue
        #  try:
        #      data[col] = mm.fit_transform(data[col].values)
        #  except TypeError:
        #      if logger:
        #          logger.info('TypeError')
        #          logger.info(data[col].drop_duplicates())
        #  except ValueError:
        #      if logger:
        #          logger.info('ValueError')
        #          logger.info(data[col].shape)
        #          logger.info(data[col].head())
        c_min = data[col].min()
        if c_min<0:
            data[col] = data[col] + np.abs(c_min)
        c_max = data[col].max()
        data[col] = data[col] / c_max

    return data


' 外れ値除去 '
def outlier(df, value=False, out_range=1.96, print_flg=False, replace_value=False, drop=False, replace_inner=False, logger=False, plus_replace=True, minus_replace=True, plus_limit=False, minus_limit=False, z_replace=False):
    '''
    Explain:
    Args:
        data(DF)        : 外れ値を除外したいデータフレーム
        value(float)    : 標準偏差を計算する値
        out_range(float): 外れ値とするZ値の範囲.初期値は1.64としている
    Return:
        data(DF): 入力データフレームから外れ値を外したもの
    '''

    std = df[value].std()
    avg = df[value].mean()

    tmp_val = df[value].values
    z_value = (tmp_val - avg)/std
    df['z'] = z_value

    inner = df[df['z'].between(left=-1*out_range, right=out_range)]
    plus_out  = df[df['z']>out_range]
    minus_out = df[df['z']<-1*out_range]

    if logger:
        length = len(df)
        in_len = len(inner)
        logger.info(f'''
#==========================================
# value         : {value}
# out_range     : {out_range}
# replace_value : {replace_value}
# plus_replace  : {plus_replace}
# minus_replace : {minus_replace}
# z_replace     : {z_replace}
# plus_limit    : {plus_limit}
# minus_limit   : {minus_limit}
# drop          : {drop}
# all max       : {df[value].max()}
# inner  max    : {inner[value].max()}
# all min       : {df[value].min()}
# inner  min    : {inner[value].min()}
# all length    : {length}
# inner length  : {in_len}
# diff length   : {length-in_len}
# plus out len  : {len(plus_out)}
# minus out len : {len(minus_out)}
#==========================================
        ''')

    # replace_valueを指定してz_valueを使い置換する場合
    if replace_value:
        if z_replace:
            if plus_replace:
                df[value] = df[value].where(df['z']<=out_range, replace_value)
            if minus_replace:
                df[value] = df[value].where(df['z']>=-out_range, replace_value)
        if plus_limit:
            df[value] = df[value].where(df['z']<=plus_limit, replace_value)
        if minus_limit:
            df[value] = df[value].where(df['z']>=-minus_limit, replace_value)


    # 外れ値を除去する場合
    elif drop:
        if plus_replace:
            df = df[df['z']>=-1*out_range]
        elif minus_replace:
            df = df[df['z']<=out_range]
    # replace_valueを指定せず、innerのmax, minを使い有意水準の外を置換する場合
    elif replace_inner:
        inner_max = inner[value].max()
        inner_min = inner[value].min()
        if plus_replace:
            df[value] = df[value].where(df['z']<=out_range, inner_max)
        elif minus_replace:
            df[value] = df[value].where(df['z']>=-out_range, inner_min)

    plus_out_val  = df[df['z']>out_range][value].drop_duplicates().values
    minus_out_val = df[df['z']<-1*out_range][value].drop_duplicates().values
    logger.info(f'''
#==========================================
# RESULT
# plus out value  : {plus_out_val}
# minus out value : {minus_out_val}
#==========================================
    ''')

    del plus_out, minus_out
    gc.collect()

    return df


def contraction(data, value, limit, max_flg=1, nan_flg=0):
    '''
    Explain:
        収縮法。limitより大きいor小さい値をlimitの値で置換する。
    Args:
        data    :
        value   :
        limit   : 収縮を適用する閾値
        max_flg : limitより大きい値を収縮する場合は1。小さい値を収縮する場合は0。
        null_flg: limitの値ではなくNaNに置換する場合は1。
    Return:
    '''

    if max_flg==1 and nan_flg==0:
        data[value] = data[value].map(lambda x: limit if x > limit else x)
    elif max_flg==0 and nan_flg==0:
        data[value] = data[value].map(lambda x: limit if x < limit else x)
    elif max_flg==1 and nan_flg==1:
        data[value] = data[value].map(lambda x: np.nan if x > limit else x)
    elif max_flg==0 and nan_flg==1:
        data[value] = data[value].map(lambda x: np.nan if x < limit else x)

    return data


#  def impute_avg(data=None, unique_id=none, level=None, index=1, value=None):
#      '''
#      Explain:
#          平均値で欠損値補完を行う
#      Args:
#          data(DF)       : nullを含み、欠損値補完を行うデータ
#          level(list)    : 集計を行う粒度。最終的に欠損値補完を行う粒度が1カラム
#                           でなく、複数カラム必要な時はリストで渡す。
#                           ただし、欠損値補完の集計を行う粒度がリストと異なる場合、
#                           次のindex変数にリストのうち集計に使うカラム数を入力する
#                           (順番注意)
#          index(int)     : 欠損値補完の際に集計する粒度カラム数
#          value(float)   : 欠損値補完する値のカラム名
#      Return:
#          result(DF): 欠損値補完が完了したデータ
#      '''

#      ' 元データとの紐付けをミスらない様にインデックスをセット '
#      #  data.set_index(unique_id, inplace=true)

#      ' Null埋めする為、level粒度の平均値を取得 '
#      use_cols = level + [value]
#      data = data[use_cols]
#      imp_avg = data.groupby(level, as_index=False)[value].mean()

#      ' 平均値でNull埋め '
#      null = data[data[value].isnull()]
#      #  null = null.reset_index()
#      fill_null = null.merge(imp_avg, on=level[:index], how='inner')

#      ' インデックスをカラムに戻して、Null埋めしたDFとconcat '
#      data = data[data[value].dropna()]
#      result = pd.concat([data, fill_null], axis=0)

#      return result


def lag_feature(data, col_name, lag, level=[]):
    '''
    Explain:
        対象カラムのラグをとる
        時系列データにおいてリーケージとなる特徴量を集計する際などに使用
    Args:
        data(DF)    : valueやlevelを含んだデータフレーム
        col_name    : ラグをとる特徴量のカラム名
        lag(int)    : shiftによってずらす行の数
                     （正：前の行数分データをとる、 負：後の行数分データをとる）
        level(list) : 粒度を指定してラグをとる場合、その粒度を入れたリスト。
                      このlevelでgroupbyをした上でラグをとる
    Return:
        data: 最初の入力データにラグをとった特徴カラムを付与して返す
    '''

    if len(level)==0:
        data[f'shift{lag}_{value}'] = data[col_name].shift(lag)
    else:
        data[f'shift{lag}_{value}@{level}'] = data.groupby(level)[col_name].shift(lag)

    return data


# カテゴリ変数をファクトライズ (整数に置換)する関数
def factorize_categoricals(data, cats):
    for col in cats:
        data[col], _ = pd.factorize(data[col])
    return data


# カテゴリ変数のダミー変数 (二値変数化)を作成する関数
def get_dummies(data, cat_list, drop=1):
    for col in cat_list:
        data = pd.concat([data, pd.get_dummies(data[col], prefix=col)], axis=1)
    if drop==1:
        data = data.drop(cat_list, axis=1)
    return data


def split_dataset(dataset, val_no, val_col='valid_no'):
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

    train = dataset[dataset[val_col] != val_no]
    test = dataset[dataset[val_col] == val_no]

    for col in train.columns:
        if col.count('valid_no'):
            train.drop(col, axis=1, inplace=True)
    for col in test.columns:
        if col.count('valid_no'):
            test.drop(col, axis=1, inplace=True)

    return train, test


def set_validation(data, target, unique_id, val_col='valid_no', fold=5, seed=1208, holdout_flg=0):
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
        ' データをfold数に分割してvalidation番号をつける '
        cv = StratifiedKFold(n_splits=fold, shuffle=True, random_state=seed)
        data = data[[unique_id, target]].reset_index(drop=True)
        x = data[unique_id].to_frame()
        y = data[target].values
        cnt=0

        for trn_idx, val_idx in cv.split(x, y):
            cnt+=1

            valid_no = np.zeros(len(val_idx))+cnt
            tmp = pd.DataFrame({'index':val_idx, val_col:valid_no})

            if cnt==1:
                tmp_result = tmp.copy()
            else:
                tmp_result = pd.concat([tmp_result, tmp], axis=0)

        tmp_result.set_index('index', inplace=True)


        result = data.join(tmp_result)
        result.drop(target, axis=1, inplace=True)
        print(result.shape)
        print(result.head())

    return result


def squeeze_target(data, col_name, size):
    '''
    col_nameの各要素について、一定数以上のデータがある要素の行のみ残す
    '''

    tmp = data.groupby(col_name).size()
    target_id = tmp[tmp >= size].index

    data = data.set_index(col_name)
    result = data.loc[target_id, :]
    result = result.reset_index()
    return result

