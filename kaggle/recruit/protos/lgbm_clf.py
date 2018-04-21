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


start_time = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())

##preset#######################
#  input_path = '../features/*.csv'

submit = pd.read_csv('../data/20180328_submit_noip_hhmi_bin.csv') # submit base
#  input_path = '../submit/*.csv'
input_path = '../input/*.csv'

#  fn_list = ['train', 'test', 'submit']
fn_list = [
    'train'
    ,'test_@odac'
    #  ,'submit_h'
]

target = 'is_attributed'
#  test_size = 0.2
seed = 2018

# feature selection
#  feim = pd.read_csv('../output/401/20180401_12_feature_importances_auc_0.8634384371701372_feature_set.csv')
#  feim.sort_values(by='importance', ascending=False, inplace=True)
#  feature_set = list(feim['feature'].values[:30])
#  feature_set = []

##load data#######################
rawdata, fs_name = load_data(input_path, fn_list, None, None)


# model params
#  metric = 'logloss'
metric = 'auc'
categorical_feature = [ 'o', 'd', 'a', 'c', 'hm', 'h' ]
early_stopping_rounds = 10000
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


def validation(train, test, feature_set=[]):

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

    logger.info('Pre Setting \nmetric:{}'.format(metric))
    logger.info('train columns: {} \n{}'.format(len(use_cols), use_cols))
    logger.debug('train columns: {} \n{}'.format(len(use_cols), use_cols))

    logger.info('validation start {}'.format(now))
    logger.debug('validation start {}'.format(now))

    # params output file
    clf = lgb.LGBMClassifier(**fix_params)
    clf.fit(x_train, y_train,
            eval_set=[(x_test, y_test)],
            eval_metric=metric,
            early_stopping_rounds=early_stopping_rounds,
            categorical_feature=categorical_feature)

    logger.info('validation end')
    now = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())
    logger.debug('validation end {}'.format(now))

    logger.info('prediction start')

    y_pred = clf.predict_proba(x_test, num_iteration=clf.best_iteration_)[:, 1]

    logger.info('prediction end')

    logger.info('feature importance caliculation start')

    # feature importance output file
    feim_result = pd.Series(clf.feature_importances_, name='importance')
    feature_name = pd.Series(use_cols, name='feature')
    features = pd.concat([feature_name, feim_result], axis=1)
    features.sort_values(by='importance', ascending=False, inplace=True)

    sc_score = sc_metrics(y_test, y_pred)
    features.to_csv('../output/{}_feature_importances_{}_{}_{}.csv'.format(
        start_time[:11], metric, sc_score, fs_name), index=False)

    logger.info('feature importance download end')

    test['prediction'] = y_pred
    test.to_csv('../output/{}_result_viz_{}_{}.csv'.format(
        start_time[:11], metric, sc_score), index=False)

    logger.info('viz data download end')


def prediction(train, pred, feature_set=[]):
    global submit

# feature_setが決まっていたらそれのみで学習させる
# feature_setが決まっていたらそれのみで学習させる
    if len(feature_set) == 0:
        train_set = set(list(train.columns.values))
        pred_set = set(list(pred.columns.values))
        use_cols = list(train_set & pred_set)
    elif len(feature_set) > 0:
        use_cols = feature_set

    x_train, y_train = x_y_split(train[use_cols+[target]], target)
    #  x_test, y_test = x_y_split(test[use_cols+[target]], target)

    pred = pred[use_cols]

    now = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())

    clf = lgb.LGBMClassifier(**fix_params)
    clf.fit(x_train, y_train,
            eval_metric=metric,
            categorical_feature=categorical_feature)

    logger.info('submit create start')
    prediction = clf.predict_proba(pred)[:, 1]
    pred['prediction'] = prediction

    logger.info('submit set merging...')
    submit = submit.merge(pred, on=['o', 'd', 'a', 'c', 'hm'], how='inner')
    submit = submit[['click_id', 'prediction']].rename(columns={'prediction':'is_attributed'})

    logger.info('submit data downloading...')
    submit.to_csv('../submit/{}_submit.csv'.format(start_time[:11]), index=False)
    logger.info('submit create complete!!')


def cleansing(data):
    for col in data.columns:
        if col.count('p_'):continue
        if col.count('dl_') or col.count('dp_'):
            data.drop(col, axis=1, inplace=True)
        if col=='dh' or col=='dhm':
            data.drop(col, axis=1, inplace=True)

    return data


def main():

    #  train_list = [
    #      'col171_70300_trainset'
    #  ]
    #  test_list = [
    #      'col171_80300_trainset'
    #  ]
    #  for tr, te in zip(train_list, test_list):
    #      train = rawdata[tr]
    #      test  = rawdata[te]
    #      #  pred = rawdata['submit']

    #      #  cross_validation(train, test, feature_set)
    #      validation(train, test)

    train = rawdata['train']
    train.rename(columns={'dl': 'is_attributed'}, inplace=True)
    train['is_attributed'] = train['is_attributed'].map(lambda x: 1 if x > 0 else 0)

    pred = rawdata['test_@odac']

    #  train = cleansing(train)
    #  test  = cleansing(test)

    #  validation(train, test)
    #  sys.exit()
    #  for f in feature_list:
    prediction(train, pred)


if __name__ == '__main__':

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
