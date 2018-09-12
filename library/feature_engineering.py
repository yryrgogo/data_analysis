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
            if length<bins:
                continue
            df[f'bin{bins}_{col}'] = pd.qcut(x=df[col], q=bins, duplicates='drop')
            df.drop(col, axis=1, inplace=True)
            #  df.rename(columns={col:f'bin{bin}_{col}'}, inplace=True)

    app = pd.read_csv('../df/application_summary_set.csv')

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


def select_category_value_agg(base, data, level, cat_list=[], num_list=[], method_list=[], ignore_list=[], prefix='', null_val='XNA'):
    '''
    Explain:
        likelifood encoding
        カテゴリカラムの各内容における行に絞った上でlevel粒度の集計を行い、特徴量を作成する。
        カテゴリカラムの内容を無視して集約するのではなく、カテゴリの内容毎に集計して、
        粒度に対する各カテゴリの統計量を横持ちさせる。
    Args:
    Return:
    '''

    ' dataそのものがbaseとなる場合 '
    if len(base) == 0:
        base = data

    if len(cat_list) == 0:
        cat_list = get_categorical_features(data, ignore_features)

    ' カテゴリカラム '
    for cat in cat_list:
        ' カテゴリカラムにNullがある場合はUnknownとして集計する '
        data[cat] = data[cat].fillna(null_val)
        ' 集計するカテゴリカラムの中身 '
        for cat_val in data[cat].drop_duplicates().values:

            ' 集計するカテゴリに絞る '
            df_cat = data[data[cat] == cat_val]

            if len(num_list) == 0:
                num_list = get_numeric_features(data, ignore_features)

            ' 集計対象のカラム '
            for val in num_list:
                if df_cat[val].dtype != 'object':

                    for method in method_list:

                        logger.info(
                            f"\ncat: {cat}\ncat_val: {cat_val}\nval: {val}\nmethod: {method}")

                        result = base_aggregation(df_cat, level, val, method)
                        #  if len(result.dropna()) / len(result) < 0.01:
                        #      continue
                        result = base.merge(result, on=level, how='left')
                        ' どのカテゴリカラムでどのVALUEについて集計したか分かるように '
                        prename = f'{prefix}{cat}_{cat_val}_'

                        make_npy(result, ignore_features, prename, method, logger=logger)


def cnt_encoding(df, category_col, ignore_list):

    cnt_enc = df[category_col].value_counts().reset_index().rename(columns={'index':category_col, category_col:f'cntec_{category_col}@'})

    ' 元データの行数とインデックスを保つ為、Left Join '
    result = df.merge(cnt_enc, on=category_col, how='left').drop(category_col, axis=1)
    return result
