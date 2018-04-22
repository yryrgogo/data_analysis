import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.metrics import log_loss, roc_auc_score
import datetime
from tqdm import tqdm
import sys
from logger import logger_func
sys.path.append('../../../module')
from load_data import load_data, x_y_split

start_time = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())
logger = logger_func()


"""Model Parameter"""
metric = 'logloss'
#  metric = 'auc'
categorical_feature = [ 'o', 'd', 'a', 'c', 'hm', 'h' ]

early_stopping_rounds = 10000
seed = 2018
all_params = {
    'objective': ['binary'],
    'num_leaves': [127, 255, 511],
    'learning_rate': [0.05, 0.1],
    'learning_rate': [0.1],
    'n_estimators': [100, 300, 500],
    'feature_fraction': [0.7, 0.8, 0.9],
    'random_state': [seed]
}

fix_params = {
    'objective': 'binary',
    'num_leaves': 511,
    'learning_rate': 0.1,
    'n_estimators': 130,
    'feature_fraction': 0.7,
    'random_state': seed
}



def sc_metrics(test, pred):
    if metric == 'logloss':
        return logloss(test, pred)
    elif metric == 'auc':
        return roc_auc_score(test, pred)
    else:
        print('score caliculation error!')


def cross_validation(train, test, feature_set=[]):

    list_score = []
    list_best_iterations = []
    best_params = None

    if metric == 'auc':
        best_score = 0
    elif metric == 'logloss':
        best_score = 100

# feature_setが決まっていたらそれのみで学習させる
    if len(feature_set) == 0:
        train_set = set(list(train.columns.values))
        test_set = set(list(test.columns.values))
        use_cols = list(train_set & test_set)
    elif len(feature_set) > 0:
        use_cols = feature_set

    x_train, y_train = x_y_split(train[use_cols], target)
    x_test, y_test = x_y_split(test[use_cols], target)
    use_cols.remove(target)

    now = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())
    logger.info('tuning start {}'.format(now))
    logger.debug('tuning start {}'.format(now))

    logger.info('Pre Setting \nmetric:{}'.format(metric))
    logger.info('train columns: {} \n{}'.format(len(use_cols), use_cols))
    logger.debug('train columns: {} \n{}'.format(len(use_cols), use_cols))

    for params in tqdm(list(ParameterGrid(all_params))):
        logger.info('params: {}'.format(params))

        clf = lgb.LGBMClassifier(**params)
        clf.fit(x_train, y_train,
                eval_set=[(x_test, y_test)],
                eval_metric=metric,
                early_stopping_rounds=early_stopping_rounds,
                categorical_feature=categorical_feature)

        y_pred = clf.predict_proba(
            x_test, num_iteration=clf.best_iteration_)[:, 1]
        sc_score = sc_metrics(y_test, y_pred)

        list_score.append(sc_score)
        list_best_iterations.append(clf.best_iteration_)
        logger.info('{}: {}'.format(metric, sc_score))
        logger.debug('   {}: {}'.format(metric, sc_score))

        logger.info('CV end')

        params['n_estimators'] = int(np.mean(list_best_iterations))
        sc_score = np.mean(list_score)
        if metric == 'logloss':
            if best_score > sc_score:
                best_score = sc_score
                best_params = params
        elif metric == 'auc':
            if best_score < sc_score:
                best_score = sc_score
                best_params = params

        logger.info('{}: {}'.format(metric, sc_score))
        logger.info('current {}: {}  best params: {}'.format(
            metric, best_score, best_params))
        logger.debug('current {}: {}  best params: {}'.format(
            metric, best_score, best_params))

    logger.info('CV best score : {}'.format(best_score))
    logger.debug('CV best score : {}'.format(best_score))
    logger.info('CV best params: {}'.format(best_params))
    logger.debug('CV best params: {}'.format(best_params))

    # params output file
    df_params = pd.DataFrame(best_params, index=['params'])
    df_params.to_csv('../output/{}_best_params_{}_{}_{}.csv'.format(
        start_time[:11], metric, best_score, fs_name), index=False)

    logger.info('tuning end')
    now = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())
    logger.debug('tuning end {}'.format(now))

    clf = lgb.LGBMClassifier(**best_params)
    clf.fit(x_train, y_train,
            eval_set=[(x_test, y_test)],
            eval_metric=metric,
            early_stopping_rounds=early_stopping_rounds,
            categorical_feature=categorical_feature)

    y_pred = clf.predict_proba(x_test, num_iteration=clf.best_iteration_)[:, 1]

    # feature importance output file
    feim_result = pd.Series(clf.feature_importances_, name='importance')
    feature_name = pd.Series(use_cols, name='feature')
    features = pd.concat([feature_name, feim_result], axis=1)
    features.sort_values(by='importance', ascending=False, inplace=True)

    sc_score = sc_metrics(y_test, y_pred)
    list_score.append(sc_score)
    features.to_csv('../output/{}_feature_importances_{}_{}_{}.csv'.format(
        start_time[:11], metric, sc_score, fs_name), index=False)

    mean_score = np.mean(list_score)
    logger.info('CV & TEST mean {}: {}  best_params: {}'.format(
        metric, mean_score, best_params))
    logger.debug('CV & TEST mean {}: {}  best_params: {}'.format(
        metric, mean_score, best_params))

    test['prediction'] = y_pred

    test.to_csv('../output/{}_result_viz_{}_{}.csv'.format(
        start_time[:11], metric, sc_score), index=False)


def validation(train, test, pts_score='0', test_viz=0):

    x_train, y_train = x_y_split(train, target)
    x_test, y_test = x_y_split(test, target)

    use_cols = x_train.columns

    logger.info('metric:{}'.format(metric))
    logger.info('train columns: {} \nuse columns: {}'.format(len(use_cols), use_cols))

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
    if pts_score!='0':
        df_pts = test[[pts_score, target]].to_frame()
        df_pts['predicion'] = y_pred

        for pts in df_pts[pts_score].drop_duplicates().values:
            tmp = df_pts[df_pts[pts_score]==pts]
            y_test_pts = tmp[target].values
            y_pred_pts = tmp['predicion'].values
            pts_score = sc_metrics(y_test_pts, y_pred_pts)
            ftim['score_{}'.format(pts)] = pts_score


    """特徴量セットに予測結果をJOINして出力"""
    if test_viz==1:
        test['prediction'] = y_pred
        test.to_csv('../output/{}_test_viz_{}_{}.csv'.format(
            start_time[:11], metric, sc_score), index=False)

    return ftim


def prediction(train, pred, feature_set=[]):
    global submit


    x_train, y_train = x_y_split(train[use_cols+[target]], target)
    use_cols = x_train.columns.values

    pred = pred[use_cols]

    clf = lgb.LGBMClassifier(**fix_params)
    clf.fit(x_train, y_train,
            eval_metric=metric,
            categorical_feature=categorical_feature)

    prediction = clf.predict_proba(pred)[:, 1]
    pred['prediction'] = prediction

    #  submit = 

    logger.info('submit data downloading...')
    #  submit.to_csv('../submit/{}_submit.csv'.format(start_time[:11]), index=False)
    logger.info('submit create complete!!')
