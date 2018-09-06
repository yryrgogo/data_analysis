import numpy as np
import pandas as pd
import sys


def base_aggregation(df, level, feature, method, prefix='', suffix=''):
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
    result = result['tmp'].reset_index().rename(
        columns={f'{method}': f'{prefix}{feature}_{method}{suffix}@{level}'})

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
    ' カテゴリ変数の種類が多い場合、ターゲットエンコーディングしてからbinにして数を絞る '
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


def num_cat_encoding(df, bins=0):
    '''
    Explain:
    連続値を離散値orカテゴリカルに変換する
    Args:
    Return:
    '''

    if bins>0:

        bin_list = get_numeric_features(df=df, ignore=ignore_features)

        logger.info(df.shape)
        for col in bin_list:
            df[col] = df[col].replace(np.inf, np.nan)
            df[col] = df[col].replace(-1*np.inf, np.nan)
            df[col] = df[col].fillna(df[col].median())
            length = len(df[col].drop_duplicates())
            #  print(df[col].drop_duplicates())
            #  continue
            #  sys.exit()
            if length<bins:
                continue
            df[f'bin{bins}_{col}'] = pd.qcut(x=df[col], q=bins, duplicates='drop')
            df.drop(col, axis=1, inplace=True)
            #  df.rename(columns={col:f'bin{bin}_{col}'}, inplace=True)

    app = pd.read_csv('../df/application_summary_set.csv')
    #  print(app.columns)
    #  print(app['bin10_a_ORGANIZATION_TYPE'].drop_duplicates())
    #  sys.exit()

    label_list = ['a_REGION_RATING_CLIENT_W_CITY', 'a_HOUSE_HOLD_CODE@']
    cat_list = get_categorical_features(df=app, ignore=ignore_features) + label_list
    cat_list = [col for col in cat_list if not(col.count('bin')) or (col.count('TION_TYPE'))]
    #  cat_list = ['a_HOUSE_HOLD_CODE@']
    #  cat_list = [col for col in cat_list if not(col.count('FLAG')) and not(col.count('GEND'))]
    bin_list = [col for col in df.columns if col.count('bin')]
    #  bin_list = [col for col in df.columns if (col.count('bin20') or col.count('bin10') )]
    df = df.merge(app, on=unique_id, how='inner')

    categorical_list = []
    for cat in cat_list:
        for num in bin_list:
            #  encode_list = [cat, elem_3, elem, elem_2]
            encode_list = [cat, num, 'a_CODE_GENDER']

            length = len(df[encode_list].drop_duplicates())
            cnt_id = len(df[unique_id].drop_duplicates())
            if length>100 or length<60 or cnt_id/length<3000:
                continue
            categorical_list.append(encode_list)

    method_list = ['mean', 'std']
    select_list = []
    val_col = 'valid_no_4'

    base = pd.read_csv('../df/base.csv')
    for cat in tqdm(categorical_list):
        length = len(df[cat].drop_duplicates())
        prefix = f'new_len{length}_'
        #  prefix = f'abp_vc{length}_'
        target_encoding(base=base, df=df, unique_id=unique_id, level=cat, method_list=method_list,
                        prefix=prefix, select_list=select_list, test=1, impute=1208, val_col=val_col, npy_key=target)

