import gc
from itertools import combinations
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import log_loss, roc_auc_score
import datetime
from tqdm import tqdm
from logging import StreamHandler, DEBUG, Formatter, FileHandler, getLogger
import sys
import glob
import re
sys.path.append('../module')
from load_data import load_data, x_y_split


start_time = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())

# PATH*************************************
log_path = '../output/py_train.py.log'
base_path = '../input/speed.csv'
feature_path = '../features/*.csv'
path_list = glob.glob(feature_path)

target = 'is_attributed'

base = pd.read_csv(base_path)

# model params
#  metric = 'logloss'
metric = 'auc'
categorical_feature = ['o', 'd', 'a', 'c']
base_feature = categorical_feature + \
    ['hm', 'h', 'rc', 'i_v', 'pts', 'val', target]
early_stopping_rounds = 10000
seed = 2018
fix_params = {
    'objective': 'binary',
    'num_leaves': 127,
    'learning_rate': 0.1,
    'n_estimators': 130,
    'feature_fraction': 0.7,
    'random_state': seed
}

base_auc = {
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


def sc_metrics(test, pred):
    if metric == 'logloss':
        return logloss(test, pred)
    elif metric == 'auc':
        return roc_auc_score(test, pred)
    else:
        print('score caliculation error!')


def validation(train, test, feature_set=[]):

    if metric == 'auc':
        best_score = 0
    elif metric == 'logloss':
        best_score = 100

    x_train, y_train = x_y_split(train, target)
    x_test, y_test = x_y_split(test, target)

    use_cols = x_train.columns

    logger.info('metric:{}'.format(metric))
    logger.info('train columns: {} \n{}'.format(len(use_cols), use_cols))

    clf = lgb.LGBMClassifier(**fix_params)
    clf.fit(x_train, y_train,
            eval_set=[(x_test, y_test)],
            eval_metric=metric,
            early_stopping_rounds=early_stopping_rounds,
            categorical_feature=categorical_feature)

    y_pred = clf.predict_proba(x_test, num_iteration=clf.best_iteration_)[:, 1]

    sc_score = sc_metrics(y_test, y_pred)

    logger.info('prediction score: {}'.format(sc_score))

    """feature importance output file"""
    ftim = pd.DataFrame({'feature': use_cols, 'importance': clf.feature_importances_})
    ftim['score_cv'] = sc_score

    """ パーティション毎にスコアを出力し、格納する """
    #  df_pts = test[['partision', target]].to_frame()
    #  df_pts['predicion'] = y_pred

    #  for pts in df_pts['partision'].drop_duplicates().values:
    #      tmp = df_pts[df_pts['partision']==pts]
    #      y_test_pts = tmp[target].values
    #      y_pred_pts = tmp['predicion'].values
    #      pts_score = sc_metrics(y_test_pts, y_pred_pts)
    #      ftim['score_{}'.format(pts)] = pts_score

    return ftim


def split_dataset(dataset, val_no):

    train = dataset[dataset['pts'] != val_no].copy()
    test = dataset[dataset['val'] == val_no].copy()

    train.drop(['val', 'pts'], axis=1, inplace=True)
    test.drop(['val', 'pts'], axis=1, inplace=True)

    return train, test


# 学習結果のfeature importanceを軸に、featureと予測結果の関係をまとめる
def feature_summarize(ftim, dataset, use_cols, ft_set_name, cnt, val_no):

    """
    予測結果を検証する為に、feature_importanceや特徴量同士の相関などを
    まとめる。

    Args:
        ftim    (df)    :feature_importanceをDFにし、cv_scoreなどを加えた
        dataset (df)    :元のデータセット
        use_cols(list)  :学習に使用した特徴量リスト
        ft_set_name(str):学習に使用した特徴量リストを出力したファイル名
        cnt     (int)   :全体の学習における通し番号
        val_no  (int)   :CVにおいて何番目のvalidationか

    Returns:result
    通し番号、CV番号、特徴量数、相関係数、特徴量リストファイル名を追加したDF
    """

    ftim['no'] = cnt
    ftim['val_no'] = val_no
    ftim['num_of_ft'] = len(use_cols)
    ftim['add_feature'] = ft_set_name

    # feature_importanceを合計1としてスケーリング
    tmp = ftim['importance'].sum()
    ftim['importance'] = ftim['importance']/tmp
    ftim.sort_values(by='importance', ascending=False, inplace=True)

    # 基準とするfeatureを設定し、各featureの重要度の相対値をとる
    # （新しいfeatureと既存のgood featureを比較）
    ch = ftim[ftim['feature'] == 'c']['importance'].values[0]
    ftim['comp_c'] = ftim['importance']/ch

    corr = top_corr(dataset, use_cols, 5)
    corr_0 = top_corr(dataset[dataset[target] == 0], use_cols, 5, '_0')
    corr_1 = top_corr(dataset[dataset[target] == 1], use_cols, 5, '_1')

    result = ftim.merge(corr, on='feature', how='left')
    result = result.merge(corr_0, on='feature', how='left')
    result = result.merge(corr_1, on='feature', how='left')

    return result


def incremental_train(dataset):
    global cnt

    list_score = []
    list_diff  = []
    # validation毎の結果をまとめるdf
    result_val = pd.DataFrame([])
    # validation毎に結果をまとめて結合していく
    for val_no in range(1, 4, 1):
        cnt += 1

        # 学習用、テスト用データセットを作成
        train, test = split_dataset(dataset, val_no)
        # 学習
        ftim = validation(train, test)

        # feature_importanceとまとめてスコアを返してるので、一番目のみ取り出す
        score_cv = ftim['score_cv'].values[0]
        list_score.append(score_cv)
        score_diff = score_cv - base_auc[val_no-1]
        list_diff.append(score_diff)
        ftim['score_diff'] = score_diff

        use_cols = list(train.columns)
        ft_set_name = '{}_{}features.csv'.format(
            start_time[:11], len(use_cols))

        "EDA用のデータセットを作成する"
        feature_summary = feature_summarize(
            ftim, dataset, use_cols, ft_set_name, cnt, val_no, score_diff)

        # validation毎にサマリーを格納し結合していく
        if len(result_val) == 0:
            result_val = feature_summary
        else:
            result_val = pd.concat([result_val, feature_summary], axis=0)

        del train, test
        gc.collect()

    # 全CVの平均スコアを格納する
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


def create_feature_set(base):

    feature_list = []
    for path in path_list[:20]:
        if path.count('dp') or path.count('std') or path.count('sur') or path.count('avg'):
            tmp = pd.read_csv(path, compression='gzip',
                              squeeze=True, dtype='float32')
        else:
            tmp = pd.read_csv(path, compression='gzip',
                              squeeze=True, dtype='int32')

        if tmp.std() == 0:
            continue

        feature_list.append(tmp)
    df_ft = pd.concat([base]+feature_list, axis=1)

    return df_ft


def main():
    base = pd.read_csv(base_path)
    base.rename(columns={'dl': target}, inplace=True)
    base[target] = base[target].map(lambda x: 1 if x >= 1 else 0)
    base = base.sort_values(by='key_dhm').reset_index(drop=True)
    df = base[base_feature].copy()
    del base

    feature_set = create_feature_set(df)
    feature_set_list = [feature_set]

    result = pd.DataFrame([])
    for feature_set in feature_set_list:
        tmp_result = incremental_train(feature_set)

        if len(result) == 0:
            result = tmp_result
        else:
            result = pd.concat([result, tmp_result], axis=0)

    result.to_csv('../output/{}_ftim.csv'.format(start_time[:11]), index=False)


if __name__ == '__main__':

    # logger
    logger = getLogger(__name__)
    log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s]\
    [%(funcName)s] %(message)s ')
    handler = StreamHandler()
    handler.setLevel('INFO')
    handler.setFormatter(log_fmt)
    logger.addHandler(handler)

    handler = FileHandler(log_path, 'a')
    handler.setLevel(DEBUG)
    handler.setFormatter(log_fmt)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)

    logger.info('start')
    cnt = 0

    main()
