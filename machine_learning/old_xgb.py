import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
from ccjc_load_data import create_train, load_train_data, load_test_data
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from sklearn.metrics import log_loss, roc_auc_score, roc_curve, auc
import xgboost as xgb

from logging import StreamHandler, DEBUG, Formatter, FileHandler, getLogger


logger = getLogger(__name__)

DIR = 'result_tmp/'
SUBMIT_FILE = 'submit.csv'


if __name__ == '__main__':

    # for log visualization
    log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s]\
    [%(funcName)s] %(message)s ')
    handler = StreamHandler()
    handler.setLevel('INFO')
    handler.setFormatter(log_fmt)
    logger.addHandler(handler)

    handler = FileHandler(DIR + 'train.py.log', 'a')
    handler.setLevel(DEBUG)
    handler.setFormatter(log_fmt)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)

    logger.info('start')

    df, df_test = create_train()

    x_train = df.drop('w2_plus_flg', axis=1)
    y_train = df['w2_plus_flg'].values

    use_cols = x_train.columns.values[]

    logger.debug('train columns: {} {}'.format(use_cols.shape, use_cols))
    logger.info('data preparation end {}'.format(x_train.shape))

    # fixed random_state
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    # パラメータを探す parameter grid
    all_params = {'max_depth':[3,5,7,9],
                # 0.1~0.01 -> try!! priority is engineering
                  'learning_rate':[0.1],
#汎化性能 過学習させないが,精度とトレードオフ  
                  'min_child_weight':[3,5,10],
# 自動で止める閾値的なもの              
                  'n_estimators':[10000],
# 使う特徴量を絞る
                  'colsample_bytree':[0.8, 0.9],
# 学習の木の深さによっても使う特徴を減らしていく
                  'colsample_bylevel':[0.8, 0.9],
                  'reg_alpha':[0,0.1],
                  'max_delta_step':[0.1],
                  'seed':[0]}
    max_score = 0
    max_params = None

    # tqdmを入れると進捗を教えてくれる。全てのfor文に入れてもいいくらい。
    for params in tqdm(list(ParameterGrid(all_params))):
        logger.info('params: {}'.format(params))

        list_auc_score = []
        list_logloss_score = []
        list_best_iterations = []

        # クロスバリデーション トレーニングセットとバリデーションセットを切って、
        # トレーニングセットで学習してバリデーションセットで評価する
        # それを全スプリットに対して行って、平均値を見てその汎化性能を測り、
        # kaggleのリーダーボードのスコアリングの様にする
        # リーダーボードと同じスコアリングになる様、splitを切るのが大事
        # 手元CVスコアとリーダーボードスコアに差があるときはどこかに偏りがあるので、そこは頑張って修正する
        # StraightfieldKFoldは行のインデックスが返ってくる
        for train_idx, valid_idx in cv.split(x_train, y_train):
            trn_x = x_train.iloc[train_idx, :]
            val_x = x_train.iloc[valid_idx, :]

            trn_y = y_train[train_idx]
            val_y = y_train[valid_idx]

            # main(part) parameter: penalty(l2, l1), C（正則化項(Cはあんま良くない？)）,
            # interceptは切片
            # **でdictをkeywordargsとして渡せる
            #  
            clf = xgb.sklearn.XGBClassifier(**params)
            #xgbは誤差に対して各ラウンドでフィットを重ねていく
            # fitの中にCVセットを食わせることができる
# 他のCVを常に監視して、一番良いところでイテレーションをストップさせる
            clf.fit(trn_x,
                    trn_y,
                    eval_set=[(val_x, val_y)],
                  early_stopping_rounds=100,
                  eval_metric='auc'
                    )

            # predicted_probaは2列で出力される
            pred = clf.predict_proba(val_x, ntree_limit=clf.best_ntree_limit)[:, 1]
            sc_logloss = log_loss(val_y, pred)
            sc_auc = roc_auc_score(val_y, pred)

            list_logloss_score.append(sc_logloss)
            list_auc_score.append(sc_auc)
            list_best_iterations.append(clf.best_iteration)
            # 学習の結果がパッと分かるようにログをとる
            # best_ntree_limit == .best_iteration + 1 なので注意
            logger.debug('   logloss: {}, auc: {}'.format(sc_logloss, sc_auc))

        # 数字の見方として、loglossは小さい方がいいが、AUCは大きい方が良い。
        # ベクトルを揃えた方が見る時に混乱しないので符号を変えとく→minimizeで統一
        # error -> min , score -> max
        params['n_estimators'] = int(np.mean(list_best_iterations))
        sc_logloss = np.mean(list_logloss_score)
        sc_auc = np.mean(list_auc_score)
        if max_score < sc_auc:
            max_score = sc_auc
            max_params = params
        logger.info('logloss: {}, auc: {}'.format(sc_logloss, sc_auc))
        logger.info('current min score: {}, params: {}'.format(max_score, max_params))

    logger.info('max params: {}'.format(max_params))
    logger.info('max auc: {}'.format(max_score)
    best_params = pd.DataFrame(max_params)
    best_params.to_csv(DIR + 'best_params.csv')

    clf = xgb.sklearn.XGBClassifier(**max_params)
    clf.fit(x_train, y_train)

    logger.info('train end')

    x_test = df_test[use_cols]

    logger.info('test data load end {}'.format(x_test.shape))
    #predict単体でやるとラベルになってしまう。確率値を取り出す時はproba
    pred_test = clf.predict_proba(x_test)[:, 1]

    pred_test.to_csv(DIR + 'submit.csv', index=False)

    logger.info('end')
