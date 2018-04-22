import numpy as np
import pandas as pd
import gc
import re
import datetime
from tqdm import tqdm
from lgbm_reg import validation


start_time = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())

# PATH*************************************

base_score = {
    1: 0.89654587,
    2: 0.900488924,
    3: 0.88991557,
    4: 0.895650121,  # 各validationの平均
}


def path_info(path):

    path_dict = {}
    path_dict['filename'] = re.search(r'/([^/.]*).csv', path).group(1)  # Linux
    path_dict['particle'] = re.search(r'feature_([0-9]*)@', path).group(1)
    path_dict['time'] = re.search(r'@([^.]*)@', path).group(1)
    path_dict['elem'] = re.search(r'\D@([^.]*).csv', path).group(1)

    return path_dict


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

    train = dataset[dataset['validation'] != val_no].copy()
    test = dataset[dataset['validation'] == val_no].copy()

    train.drop(['validation'], axis=1, inplace=True)
    test.drop(['validation'], axis=1, inplace=True)

    return train, test


# 学習結果のfeature importanceを軸に、featureと予測結果の関係をまとめる
def feature_summarize(ftim, dataset, use_cols, target, ft_set_name, number, val_no):
    """
    予測結果を検証する為に、feature_importanceや特徴量同士の相関などを
    まとめる。

    Args:
        ftim    (df)    : feature_importanceをDFにし、cv_scoreなどを加えた
        dataset (df)    : 元のデータセット
        use_cols(list)  : 学習に使用した特徴量リスト
        #  target(str)     : 
        ft_set_name(str): 学習に使用した特徴量リストを出力したファイル名
        number  (int)   : 全体の学習における通し番号
        val_no  (int)   : CVにおいて何番目のvalidationか

    Returns:result
    通し番号、CV番号、特徴量数、相関係数、特徴量リストファイル名を追加したDF
    """

    ftim['no'] = number
    ftim['val_no'] = val_no
    ftim['num_of_ft'] = len(use_cols)
    ftim['add_feature'] = ft_set_name

    # feature_importanceを合計1としてスケーリング
    tmp = ftim['importance'].sum()
    ftim['importance'] = ftim['importance']/tmp
    ftim.sort_values(by='importance', ascending=False, inplace=True)

    # 基準とするfeatureを設定し、各featureの重要度の相対値をとる
    # （新しいfeatureと既存のgood featureを比較）
    comp = ftim[ftim['feature'] == 'dow']['importance'].values[0]
    ftim['comp_dow'] = ftim['importance']/comp

    corr = top_corr(dataset, use_cols, 5)
    corr_0 = top_corr(dataset[dataset[target] == 0], use_cols, 5, '_0')
    corr_1 = top_corr(dataset[dataset[target] == 1], use_cols, 5, '_1')

    result = ftim.merge(corr, on='feature', how='left')
    result = result.merge(corr_0, on='feature', how='left')
    result = result.merge(corr_1, on='feature', how='left')

    return result


def exploratory_train(dataset, target, categorical_feature, max_val_no=1, number=0, pts_score='0', test_viz=0, base_score={}):
    """
    時系列モデルの検証、学習、予測結果のEDAを行う。
    Partision、Validationの番号をデータセットにカラムとして持たせ、その番号
    毎に学習と予測を行い、今の特徴量セットの分析を進める。

    Args:
        dataset(df)    : 元のデータセット
        max_val_no(int): 複数のValidation_noがある場合、その最大値を入れる（番号は1からの連番が前提）
        number  (int)  : 全体の学習における通し番号。EDA用に出力するファイルを見やすくする為。

    Returns:result_val
        入力されたデータセットにおける指定Validation_noの予測結果をそれぞれ分析する為のデータフレーム。
    """

    list_score = []
    list_diff = []
    # validation毎の結果をまとめるdf
    result_val = pd.DataFrame([])
    # validation毎に結果をまとめて結合していく
    for val_no in tqdm(range(1, max_val_no+1, 1)):

        # 学習用、テスト用データセットを作成
        train, test = split_dataset(dataset, val_no)
        # 学習
        ftim = validation(train, test, target, categorical_feature, pts_score, test_viz)

        # feature_importanceとまとめてスコアを返してるので、一番目のみ取り出す
        score_cv = ftim['score_cv'].values[0]
        list_score.append(score_cv)

        if len(base_score) > 0:
            score_diff = score_cv - base_score[val_no-1]
        else:
            score_diff = 0
        list_diff.append(score_diff)
        ftim['score_diff'] = score_diff

        use_cols = list(train.columns)
        ft_set_name = '{}_{}features.csv'.format(
            start_time[:11], len(use_cols))

        "EDA用のデータセットを作成する"
        feature_summary = feature_summarize(
            ftim, dataset, use_cols, target, ft_set_name, number, val_no)

        # validation毎にサマリーを格納し結合していく
        if len(result_val) == 0:
            result_val = feature_summary
        else:
            result_val = pd.concat([result_val, feature_summary], axis=0)

        del train, test
        gc.collect()

    # 全Validationの平均スコアを格納する
    result_val['score_mean'] = np.mean(list_score)
    result_val['diff_mean'] = np.mean(list_diff)
    # 学習に使用した特徴量リストを確認する為のファイルを出力する
    pd.Series(use_cols, name='features').to_csv(
        '../output/{}_use_{}features.csv'.format(start_time[:11], len(use_cols)), index=False)

    return result_val


def top_corr(data, use_cols, num, suffix=''):
    """
    Explain:
        対象featureと相関の高い上位featureとその係数を取得する
        また、各パーティションにより相関が変わるかなども確認する
        (パーティション毎の相関などを求めたい際は、class毎にDFを切った上でわたす)
        変数名に対する横持ちでデータフレームが返ってくるので、featureカラムを
        keyとして結合する

    Args:
        data (DF)       : 相関を計算したいカラムをまとめたデータフレーム
        use_cols (list) : 相関係数を取り出したい変数名のリスト
        num             : 上位何番目まで相関係数とその変数を取り出すか
        suffix          : 複数の相関を一つのDFにまとめたい時、区別用の文字を末尾につける

    Returns:
        result_corr(DF):
        上位相関係数とその変数をTableau可視化用フォーマットにしたDF。
        変数名に対して横持ちでvalueが入っているので、
        mergeする時はカラム名:featureをkeyとする

    """

    # 相関を計算
    df = data.corr(method='pearson')
    df = df[df < 1]  # 相関が1となる行は表示しない

    # 結果を格納するデータフレーム
    result_corr = pd.DataFrame([])

    for ft in use_cols:

        # 相関係数上位を取り出しDFに変換、インデックスを振りなおす
        # reset_indexにより、変数名がカラム名:indexとしてDFに入る
        top = df[ft].sort_values(ascending=False)[
            :num].to_frame().reset_index()

        # インデックスにするリストを作る:要素はカラム名。相関の順位とカテゴリか係数かをわかるように
        idx_cat = []
        idx_cor = []
        for i in range(1, num+1, 1):
            idx_cat.append('corr_{}_cat'.format(i)+suffix)
            idx_cor.append('corr_{}'.format(i)+suffix)
        idx = idx_cat + idx_cor

        # 相関上位変数名とその相関係数を一つのリストにまとめる
        val = list(top['index']) + list(top[ft])
        top_cor = pd.Series(val, index=idx)
        top_cor['feature'] = ft  # どの変数について相関を求めたか
        top_cor = top_cor.to_frame().T  # 横持ちにする

        # 各変数について、横持ちを積み上げていく
        if len(result_corr) == 0:
            result_corr = top_cor
        else:
            result_corr = pd.concat([result_corr, top_cor], axis=0)

    return result_corr


def incremental_train(dataset, feature_list, max_val_no):
    """ Summary line
    用意した特徴量の最適な組み合わせを発見する為、インクリメンタルな学習を行う。
    戻り値はなく、全学習＆予測結果のまとめをCSVで出力して終わり。

    Args:
        dataset(DF)       : 候補となる全特徴量を含んだデータフレーム。
        feature_list(list): 学習を試したい特徴量の組み合わせリスト。この特徴量
                           セットを順に流し、学習結果とその分析結果をまとめる。
    """

    number = 0
    result = pd.DataFrame([])

    for feature_set in tqdm(feature_list):

        number += 1
        tmp_result = exploratory_train(
            dataset[feature_set], max_val_no, number)

        if len(result) == 0:
            result = tmp_result
        else:
            result = pd.concat([result, tmp_result], axis=0)

    result.to_csv('../output/{}_ftim.csv'.format(start_time[:11]), index=False)
