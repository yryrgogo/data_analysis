import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import log_loss
import sys
import glob
import datetime
from tqdm import tqdm
from logging import StreamHandler, DEBUG, Formatter, FileHandler, getLogger
from load_data import load_data


logger = getLogger(__name__)
start_time = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())


input_path = ''
fn = re.search(r'/([^/.]*).csv', inpu_path).group(1)

# feature selection
feim_result = pd.read_csv('../feature_enginering/2017_9.79468578925634_feature_importances.csv')
fe_sel = feim_result.groupby(['feature'], as_index=False)['importance'].mean()
fe_sel.sort_values(by='importance', ascending=False, inplace=True)
fe_list1 = list(fe_sel['feature'].values[:100])
fe_list2 = list(fe_sel['feature'].values[:200])

metric = 'logloss'
categorical_feature = []
early_stopping_rounds = 1000
all_params = {
    'objective': ['binary'],
    'num_leaves': [31, 63, 127, 255, 511, 1023],
    #  'learning_rate': [0.05, 0.01],
    'learning_rate': [0.05],
    'n_estimators': [100, 300, 500],
    'feature_fraction': [0.7, 0.8, 0.9],
    'random_state': [2018]
}

fix_params = {
    'objective': 'binary',
    #  'num_leaves': 1023,
    'num_leaves': 511,
    'learning_rate': 0.05,
    'n_estimators': 120,
    'feature_fraction': 0.7,
    'random_state': 2018
}



def sc_metrics(test, pred):
    if metric == 'logloss':
        return logloss(test, pred)
    elif metric == 'auc':
        return auc(test, pred)
    else:
        print('score caliculation error!')


def tuning(x_train, y_train):

    now = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())
    logger.debug('tuning start {}'.format(now))
    use_cols = x_train.columns.values

    logger.debug('train columns: {} {}'.format(use_cols.shape, use_cols))
    logger.info('data preparation end {}'.format(x_train.shape))

    list_score = []
    list_best_iterations = []
    best_score = 100
    best_params = None
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

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
        elif metric == 'auc'
            if best_score < sc_score:
                best_score = sc_score
                best_params = params

        logger.info('{}: {}'.format(metric, sc_score))
        logger.debug('   {}: {}'.format(metric, sc_score))
        logger.info('current best score: {}  best params: {}'.format(
            best_score, best_params))

    logger.info('best score : {}'.format(best_score))
    logger.info('best params: {}'.format(best_params))
    df_params = pd.DataFrame(min_params, index=['params'])
    df_params.to_csv('../output/{}_best_params_{}_{}_{}.csv'.format(start_time[:11], metric, best_score, fn))

    logger.info('tuning end')
    now = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())
    logger.debug('tuning end {}'.format(now))

    clf = lgb.LGBMClassifier(**min_params)
    clf.fit(x_train, y_train,
            eval_set=[(x_test, y_test)],
            eval_metric=metric,
            early_stopping_rounds=early_stopping_rounds,
            categorical_feature=categorical_feature)

    y_pred = clf.predict(x_test, num_iteration=clf.best_iteration_)

    feim = pd.Series(clf.feature_importances_, name='importance')
    feature_name = pd.Series(use_cols, name='feature')
    features = pd.concat([feature_name, feim], axis=1)
    features.sort_values(by='importance', ascending=False, inplace=True)

    sc_score = sc_metrics(y_test, y_pred)
    list_score.append(sc_score)
    features.to_csv('../output/{}_feature_importances_{}_{}_{}.csv'.format(start_time[:11], metric, sc_score, fn))

    mean_score = np.mean(list_score)
    logger.info('CV & TEST mean_score: {}'.format(mean_score))
    logger.debug('CV & TEST mean_score: {}'.format(mean_score))


def prediction(feature_list):
    global data

    feature_list = list(submit_feature.columns.values)
    start_time = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())
    pred_dict = {}
    feature_dict = {}
    score_list = []

    #  for season in tqdm(train_season_list):
    output = data.reset_index().copy()
    #  df = data[feature_list+['result', 'daynum', 'pred_flg']].copy()
    df = data[feature_list+['result']].copy()
    df = df.loc[train_season_list, :]
    df.reset_index(inplace=True)
    test_season = df['season'].max()

    tmp_train = df[df['season'] != test_season]
    tmp_test = df[df['season'] == test_season]
    tmp_train2 = tmp_test[tmp_test['daynum'] < 133]

    train = pd.concat([tmp_train, tmp_train2], axis=0)
    test = tmp_test[tmp_test['daynum'] >= 133]

    x_train = train.drop(['result', 'season'], axis=1).copy()
    y_train = train['result'].values
    x_test = test.drop(['result', 'season'], axis=1).copy()
    y_test = test['result'].values

    use_cols = x_train.columns.values
    use_cols = submit_feature.columns.values

    print('***TRAIN COLUMN NUMBER***')
    print(len(x_train.columns))
    print('***TRAIN ROW NUMBER***')
    print(len(x_train))
    print('***TEST ROW NUMBER***')
    print(len(x_test))

    clf = lgb.LGBMClassifier(**fix_params)
    clf.fit(x_train, y_train,
            eval_set=[(x_test, y_test)],
            eval_metric='logloss',
            early_stopping_rounds=early_stopping_rounds,
            categorical_feature=categorical_feature)

    tmp_y_pred = clf.predict_proba(
        x_test, num_iteration=clf.best_iteration_)[:, 1]
    sc_logloss = log_loss(y_test, tmp_y_pred)

    #  out_train = output[use_cols]
    #  out_test  = output['result'].values

    #  print('***PREDICTION COLUMN NUMBER***')
    #  print(len(out_train.columns))
    #  print('**************************')
    #  print('***PEDICTION ROW NUMBER***')
    #  print(len(out_train))
    #  print('**************************')

    #  y_pred = clf.predict_proba(out_train, num_iteration=clf.best_iteration_)[:,1]

    #  pred = pd.Series(y_pred, name='prediction_result').to_frame()
    #  pred['observation'] = out_test
    #  train_result = pd.concat([output[['teamid', 'teamid_2', 'season', 'daynum', 'daynum_2', 'result']], pred], axis=1)
    #  train_result.to_csv('../output/{}_{}_ncaa_pred_score_8ver_viz.csv'.format(start_time, test_season), index=False)

    #  return

    submit_data = submit_feature[submit_feature['season']
                                 == 2018][use_cols].copy()
    submit_pred = clf.predict_proba(
        submit_data[use_cols], num_iteration=clf.best_iteration_)[:, 1]

    submit_score = submit_base[submit_base['season'] == 2018].copy()
    submit_score['Pred'] = submit_pred
    submit_result = submit.merge(
        submit_score, on=['teamid', 'teamid_2', 'season'], how='inner')[['ID', 'Pred']]

    submit_result.to_csv(
        '../output/{}_2018_ncaa_submit.csv'.format(start_time), index=False)

    #  submit_result['Pred'] = submit_result['Pred'].map(lambda x: x*1.35 if x>0.7 else x*0.1 if x<0.3 else x)
    #  submit_result['Pred'] = submit_result['Pred'].map(lambda x: 0.99 if x>1.0 else 0.01 if x<0.0 else x)

    submit_result.to_csv(
        '../output/{}_2018_ncaa_submit_adjust.csv'.format(start_time), index=False)


def main():

    #  for f in [fe_list2, fe_list3]:
    prediction(fe_list2)
    #  continue
    sys.exit()


if __name__ == '__main__':

    # for log visualization
    log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s]\
    [%(funcName)s] %(message)s ')
    handler = StreamHandler()
    handler.setLevel('INFO')
    handler.setFormatter(log_fmt)
    logger.addHandler(handler)

    handler = FileHandler('../output/ncaa_train.py.log', 'a')
    handler.setLevel(DEBUG)
    handler.setFormatter(log_fmt)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)

    logger.info('start')

    main()
