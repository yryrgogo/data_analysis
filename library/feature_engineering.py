import gc
import numpy as np
import pandas as pd
import sys
from sklearn.model_selection import StratifiedKFold
import os
HOME = os.path.expanduser('~')
sys.path.append(f"{HOME}/kaggle/github/library/")
import utils
from utils import logger_func, get_categorical_features, get_numeric_features, pararell_process
pd.set_option('max_columns', 200)
pd.set_option('max_rows', 200)


def base_aggregation(df, level, feature, method, prefix='', suffix='', base=[]):
    '''
    Explain:
        levelの粒度で集約を行う。この関数が受け取る集計対象カラムは一つなので、
        複数カラムを集計する時はループで繰り返し引数を渡す
    Args:
        level: str/list/tuple。要素数は何個でもOK
    Return:
        集約したカラムとlevelの2カラムDF。
        集約したカラム名は、{prefix}{元のカラム名}_{メソッド}@{粒度}となっている
    '''

    ' levelの型がlistでもtupleでもなければ単体カラムのはずなので、listにする '
    if not(str(type(level)).count('tuple')) and not(str(type(level)).count('list')):
        level = [level]
    elif str(type(level)).count('tuple'):
        level = list(level)

    df = df[level+[feature]]

    result = df.groupby(level)[feature].agg({'tmp': {method}})
    result = result['tmp'].reset_index().rename(columns={f'{method}': f'{prefix}{feature}_{method}{suffix}@'})
    if len(base):
        try:
            result = base[level].to_frame().merge(result, on=level, how='left')[f'{prefix}{feature}_{method}{suffix}@']
        except AttributeError:
            result = base[level].merge(result, on=level, how='left')[f'{prefix}{feature}_{method}{suffix}@']


    return result


def diff_feature(df, first, second):
    ' 大きい値がf1に来るようにする '
    if df[first].mean()<df[second].mean():
        f1 = second
        f2 = first
    else:
        f1 = first
        f2 = second
    df[f'{f1}_diff_{f2}@'] = df[f1] - df[f2]
    return df


def division_feature(df, first, second, sort=1):
    ' 大きい値がf1に来るようにする '
    if df[first].mean()<df[second].mean() and sort==1:
        f1 = second
        f2 = first
    else:
        f1 = first
        f2 = second
    df[f'{f1}_div_{f2}@'] = df[f1] / df[f2]
    return df


def product_feature(df, first, second):
    ' 大きい値がf1に来るようにする '
    if df[first].mean()<df[second].mean():
        f1 = second
        f2 = first
    else:
        f1 = first
        f2 = second
    df[f'{f1}_pro_{f2}@'] = df[f1] * df[f2]
    return df


def cat_to_target_bin_enc(df, category, bins=10, target='TARGET'):
    ' カテゴリ変数の種類が多い場合、ターゲットエンコーディングしてからその近さでbinにして数を絞る '
    target_avg = df.groupby(category)['TARGET'].mean().reset_index()

    ' positive率でカテゴリをビンにする '
    bin_list = ['TARGET']
    for col in bin_list:
        target_avg[target] = pd.qcut(x=target_avg[col], q=bins, duplicates='drop')
        target_avg.set_index(category, inplace=True)
        target_avg = target_avg.iloc[:, 0]
    df[category] = df[category].map(target_avg)

    ' positive率が低い順に番号をつける '
    bin_sort = df[category].drop_duplicates().sort_values().to_frame()
    bin_sort['index'] = np.arange(len(bin_sort))+1
    bin_sort = bin_sort.set_index(category).iloc[:, 0]
    df[category] = df[category].map(bin_sort)

    return df


def num_cat_encoding(df, bins=0, isfill=False, origin_drop=True):
    '''
    Explain:
        Numeric to binning
    Args:
    Return:
    '''

    bin_list = get_numeric_features(df=df, ignore=ignore_features)

    logger.info(df.shape)
    for col in bin_list:
        # 必要ならNullは中央値で埋める
        if isfill:
            df[col] = df[col].replace(np.inf, np.nan)
            df[col] = df[col].replace(-1*np.inf, np.nan)
            df[col] = df[col].fillna(df[col].median())
        # binにする数よりユニーク数が少ない場合は除外
        length = len(df[col].drop_duplicates())
        if length<bins:
            continue
        df[f'bin{bins}_{col}'] = pd.qcut(x=df[col], q=bins, duplicates='drop')
        if origin_drop:
            df.drop(col, axis=1, inplace=True)


def cat_to_target_bin_enc(df, df_target, category, bins=10):
    '''
    Explain:
        dfでは全インデックスをもった元データをうけとり、そこに
        binにしたfeatureをカラムとして返す。df_targetは01の二値
        をもったDF. -1など三値入ると計算が狂うので注意。
    Args:
        category(column): ターゲットエンコーディングのビンで集約したいカテゴリカラム
    Return:
        元データのカテゴリを集約したカラムに変換したデータ
    '''

    ' カテゴリ変数の種類が多い場合、ターゲットエンコーディングしてからbinにして数を絞る '
    target_avg = df_target.groupby(category)['TARGET'].mean().reset_index()

    ' positive率でカテゴリをビンにする '
    bin_list = ['TARGET']
    for col in bin_list:
        target_avg[target] = pd.qcut(
            x=target_avg[col], q=bins, duplicates='drop')
        target_avg.set_index(category, inplace=True)
        target_avg = target_avg.iloc[:, 0]
    df[category] = df[category].map(target_avg)

    ' positive率が低い順に番号をつける '
    bin_sort = df[category].drop_duplicates().sort_values().to_frame()
    bin_sort['index'] = np.arange(len(bin_sort))+1
    bin_sort = bin_sort.set_index(category).iloc[:, 0]
    df[category] = df[category].map(bin_sort)

    return df


def col_rename(data, level, ignore_features, prefix='', suffix=''):
    '''
    Explain:
        いちいちrenameを書くのが面倒な時に使う。
        {prefix}{元のカラム名}{suffix}@{粒度}のカラム名にrenameしたいとき便利
    Args:
        ignore_features(list): renameしたくないカラムリスト
    Return:
    '''
    for col in data.columns:
        if col not in ignore_features:
            data.rename(
                columns={col: f'{prefix}{col}{suffix}@{level}'}, inplace=True)
    return data


def select_category_value_agg(base, df, key, category_col, value, method, path='../features/1_first_valid', ignore_list=[], prefix='', null_val='XNA'):
    '''
    Explain:
        likelifood encoding
        カテゴリカラムの各内容における行に絞った上でlevel粒度の集計を行い、特徴量を作成する。
        カテゴリカラムの内容を無視して集約するのではなく、カテゴリの内容毎に集計して、
        粒度に対する各カテゴリの統計量を横持ちさせる。
    Args:
    Return:
    '''

    df = df[[level, category_col, value]]
    ' カテゴリカラムにNullがある場合はXNAとして集計する '
    df[category_col].fillna(null_val, inplace=True)

    ' 集計するカテゴリカラムの中身 '
    for cat_val in df[category_col].drop_duplicates().values:

        ' 集計するカテゴリに絞る '
        df_cat = df.query("{category_col} == '{cat_val}'")

        if df_cat[value].dtype != 'object':

            logger.info(f"\ncat: {category_col}\ncat_val: {cat_val}\nval: {value}\nmethod: {method}")

            result = base_aggregation(df_cat, level, value, method)
            result = base.merge(result, on=level, how='left')

            feature_list = [col for col in result.columns if col.count('@')]
            for feature in feature_list:
                utils.to_pickle(path=f'{path}/{prefix}{feature}.fp', obj=result[feature].values)


def cnt_encoding(df, category_col, ignore_list):

    cnt_enc = df[category_col].value_counts().reset_index().rename(columns={'index':category_col, category_col:f'cnt_{category_col}@'})
    result = df.merge(cnt_enc, on=category_col, how='left').drop(category_col, axis=1)
    return result


def exclude_feature(col_name, feature):
    if np.var(feature)==0:
        logger.info(f'''
        #========================================================================
        # ***WARNING!!*** VARIANCE 0 COLUMN : {col_name}
        #========================================================================''')
        return True
    return False


def target_encoding(logger, base, train, test, key, target, level, method='mean', path='../features/1_first_valid', prefix='', select_list=[], ignore_list=[], seed=1208):
    '''
    Explain:
        TARGET関連の特徴量を4partisionに分割したデータセットから作る.
        1partisionの特徴量は、残り3partisionの集計から作成する。
        test対する特徴量は、train全てを使って作成する
    Args:
        df(DF)               : 入力データ。カラムにはkeyとvalid_noがある前提
        key                  : ユニークカラム名
        level(str/list/taple): ターゲットを集計する粒度
        target               : エンコーディングに使うカラム名
        method(str)          : 集計のメソッド
        select_list(list)    : 特定のfeatureのみ保存したい場合はこちらにリストでfeature名を格納
    Return:
        カラム名は{prefix}{target}@{level}
    '''
    val_col = 'valid_no'

    ' levelはリストである必要がある '
    if str(type(level)).count('str'):
        level = [level]
    elif str(type(level)).count('tuple'):
        level = list(level)

    ' KFold '
    if fold_type=='stratified':
        folds = StratifiedKFold(n_splits=fold, shuffle=True, random_state=seed) #1
        kfold = folds.split(train,y)
    elif fold_type=='group':
        if group_col_name=='':raise ValueError(f'Not exist group_col_name.')
        folds = GroupKFold(n_splits=fold)
        kfold = folds.split(train, y, groups=train[group_col_name].values)

    base_train = train[key].to_frame()
    result = pd.DataFrame()
    # Train内のTE
    for n_fold, (trn_idx, val_idx) in enumerate(kfold):

        x_train, y_train = train.iloc[trn_idx, :], y.iloc[trn_idx]
        x_val, y_val = train.iloc[val_idx, :], y.iloc[val_idx]

        x_train = x_train.groupby(level)[target].agg({f'TE_{target}@{level}':f'{method}'}).reset_index()
        tmp_result = x_val.drop(target, axis=1).merge(x_train, on=level, how='left')

        if len(result) == 0:
            result = tmp_result.copy()
        else:
            result = pd.concat([result, tmp_result], axis=0)

        gc.collect()

    result = base_train.merge(result, on=key, how='inner')

    # Train内のTE
    train = train.groupby(level)[target].agg({f'TE_{target}@{level}':f'{method}'}).reset_index()
    test_result = test.merge(train, on=level, how='left')

    logger.info(f'''
#========================================================================
# COMPLETE TARGET ENCODING!!
# FEATURE : TE_{target}@{level}
# LENGTH  : Train{len(result)} / Test{len(test_result)}
#========================================================================''')

    return result[f'TE_{target}@{level}'].values, test_result[f'TE_{target}@{level}'].values
