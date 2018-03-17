import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import ParameterGrid, StratifiedKFold, train_test_split
from sklearn.metrics import log_loss, roc_auc_score
import datetime
from tqdm import tqdm
from logging import StreamHandler, DEBUG, Formatter, FileHandler, getLogger
import sys
sys.path.append('../module')
from load_data import load_data, x_y_split, extract_set


# *********load_data*********
rawdata, fs_name = load_data(input_path, fn_list, None, None)

# indexカラムにより範囲指定してデータセットを作成
#  df_range = extract_set(data, index, row_list)

#  sys.exit()

# feature selection
#  feim = pd.read_csv('../feim/.csv')
#  feim.sort_values(by='importance', ascending=False, inplace=True)
#  fe_list1 = list(feim['feature'].values[:100])
#  fe_list2 = list(feim['feature'].values[:200])


def sc_metrics(test, pred):
    if metric == 'logloss':
        return logloss(test, pred)
    elif metric == 'auc':
        return roc_auc_score(test, pred)
    else:
        print('score caliculation error!')


def tuning(data, feature_set=[]):

    df = data.copy()
    list_score = []
    list_best_iterations = []
    best_params = None
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    if metric == 'auc':
        best_score = 0
    elif metric == 'logloss':
        best_score = 100

# feature_setが決まっていたらそれのみで学習させる
    if len(feature_set) == 0:
        use_cols = df.columns.values
    elif len(feature_set) > 0:
        use_cols = feature_set

    train, test = train_test_split(
        df[use_cols], test_size=test_size, random_state=seed)
    x_train, y_train = x_y_split(train, target)
    x_test, y_test = x_y_split(test, target)

    now = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())
    logger.info('tuning start {}'.format(now))
    logger.debug('tuning start {}'.format(now))

    logger.info('Pre Setting \nmetric:{}'.format(metric))
    logger.info('train columns: {} \n{}'.format(use_cols.shape, use_cols))
    logger.debug('train columns: {} \n{}'.format(use_cols.shape, use_cols))

    for params in tqdm(list(ParameterGrid(all_params))):
        logger.info('params: {}'.format(params))

        for train_idx, valid_idx in tqdm(cv.split(x_train, y_train)):
            trn_x = x_train.iloc[train_idx, :]
            val_x = x_train.iloc[valid_idx, :]

            trn_y = y_train[train_idx]
            val_y = y_train[valid_idx]

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
            logger.debug('{}: {}'.format(metric, sc_score))
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

    y_pred = clf.predict(x_test, num_iteration=clf.best_iteration_)

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
    logger.info('CV & TEST mean {}: {}'.format(metric, mean_score))
    logger.debug('CV & TEST mean {}: {}'.format(metric, mean_score))


def prediction(data, feature_set=[]):

    df = data.copy()

# feature_setが決まっていたらそれのみで学習させる
    if len(feature_set) == 0:
        use_cols = df.columns.values
    elif len(feature_set) > 0:
        use_cols = feature_set

    x, y = x_y_split(df[use_cols], target)

    now = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())

    clf = lgb.LGBMClassifier(**fix_params)
    clf.fit(x, y,
            #  eval_set=[(x_test, y_test)],
            eval_metric=metric,
            #  early_stopping_rounds=early_stopping_rounds,
            categorical_feature=categorical_feature)

    y_pred = clf.predict_proba(x)[:, 1]
    sc_score = sc_metrics(y, y_pred)

    result = df[use_cols]
    result['pred'] = y_pred
    result['obs'] = y
    result.to_csv(
        '../output/{}_pred_and_obs_viz_{}.csv'.format(start_time[:11], fs_name), index=False)


def main():

    tuning(rawdata)
    sys.exit()
    #  for f in feature_list:
    #  prediction(rawdata)


if __name__ == '__main__':

    # Pre Setting
    start_time = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())

    # path
    input_path = '../input/*.csv'

    # dataset info
    fn_list = ['feature_set', 'submit']
    fn_list = []
    target = 'result'
    test_size = 0.2
    seed = 2018

    # model params
    #  metric = 'logloss'
    metric = 'auc'
    categorical_feature = []
    early_stopping_rounds = 1000
    all_params = {
        'objective': ['binary'],
        'num_leaves': [31],
        #  'num_leaves': [31, 63, 127, 255, 511, 1023],
        #  'learning_rate': [0.05, 0.01],
        'learning_rate': [0.05],
        #  'n_estimators': [100, 300, 500],
        'n_estimators': [100],
        #  'feature_fraction': [0.7, 0.8, 0.9],
        'feature_fraction': [0.7],
        'random_state': [seed]
    }

    fix_params = {
        'objective': 'binary',
        #  'num_leaves': 1023,
        'num_leaves': 511,
        'learning_rate': 0.05,
        'n_estimators': 120,
        'feature_fraction': 0.7,
        'random_state': seed
    }

    # logger
    logger = getLogger(__name__)
    log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s]\
    [%(funcName)s] %(message)s ')
    handler = StreamHandler()
    handler.setLevel('INFO')
    handler.setFormatter(log_fmt)
    logger.addHandler(handler)

    handler = FileHandler('../output/py_train.py.log', 'a')
    handler.setLevel(DEBUG)
    handler.setFormatter(log_fmt)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)

    logger.info('start')

    main()
