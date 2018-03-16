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

logger = getLogger(__name__)
start_time = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())

#  path_list = glob.glob()

input_path = '../input/20180314_last_ncaa_feature_set.csv'
data = pd.read_csv(input_path)
data.set_index('season', inplace=True)
#  data.drop(['coachname_2_2', 'coachname_2_2.1', 'score_2_2', 'score_2_2.1'], axis=1, inplace=True)
#  data.drop(['score', 'score_2', 'coachname_2_2', 'coachname_2_2.1', 'coach_change_flg_2_2', 'coach_change_flg_2_2.1', 'score_2_2', 'score_2_2.1'], axis=1, inplace=True)
#  data.drop(['result', 'score_2'], axis=1, inplace=True)
data.drop(['result', 'score'], axis=1, inplace=True)

#  test_season_list = [2014, 2015, 2016, 2017]
test_season_list = [2017]

#  train_season_list = []
# 8year train set
#  [train_season_list.append(np.arange(i-9, i, 1)) for i in range(2019-4, 2019, 1)]
#  [train_season_list.append(np.arange(i-9, i, 1)) for i in range(2019-4, 2018, 1)]
train_season_list = np.arange(2004, 2018, 1)
# 10year train set
#  [train_season_list.append(np.arange(i-11, i, 1)) for i in range(2019-4, 2019, 1)]
categorical_feature = ['teamid_label', 'teamid_2_label', 'location', 'coachname', 'coachname_2']

# cluster

# submit
submit_path = '../input/20180315_061850_ncaa_submit_feature.csv'
submit_feature = pd.read_csv(submit_path)
submit_base = submit_feature[['teamid', 'teamid_2', 'season']].copy()

# feature selection
feim_result = pd.read_csv('../feature_enginering/2016_8.248161479074813_feature_importances.csv')
fe_sel = feim_result.groupby(['feature'], as_index=False)['importance'].mean()
fe_sel.sort_values(by='importance', ascending=False, inplace=True)
fe_list1 = list(fe_sel['feature'].values[:100])
fe_list2 = list(fe_sel['feature'].values[:200])
fe_list3 = list(fe_sel['feature'].values[:300])
fe_list4 = list(fe_sel['feature'].values[:400])

early_stopping_rounds = 10000
all_params = {
    'objective': ['binary'],
    'num_leaves': [511, 1023],
    'learning_rate': [0.01, 0.05],
    'n_estimators': [100, 300],
    'feature_fraction': [0.7, 0.8, 0.9],
    'random_state':[2018]
}

fix_params = {
    'objective': 'regression',
    'num_leaves': 1023,
    'learning_rate': 0.01,
    'n_estimators': 120,
    'feature_fraction': 0.7,
    'random_state':2018
}


def main():

    #  for f in [fe_list1, fe_list2, fe_list3, fe_list4]:
    prediction(fe_list2)
    #  continue
    sys.exit()

    season_score = []
    for season in tqdm(train_season_list):
        df = data.loc[season, :]
        df.reset_index(inplace=True)
        test_season = df['season'].max()

        tmp_train  = df[df['season'] != test_season]
        tmp_test   = df[df['season'] == test_season]
        tmp_train2 = tmp_test[tmp_test['daynum']<133]

        train = pd.concat([tmp_train, tmp_train2], axis=0)
        test = tmp_test[tmp_test['daynum']>=133]

        x_train = train.drop(['result', 'teamid', 'teamid_2', 'season'], axis=1).copy()
        y_train = train['result'].values
        x_test = test.drop(['result', 'teamid', 'teamid_2', 'season'] , axis=1).copy()
        y_test = test['result'].values

        use_cols = x_train.columns.values

        logger.debug('train columns: {} {}'.format(use_cols.shape, use_cols))
        logger.info('data preparation end {}'.format(x_train.shape))

        list_logloss = []
        list_best_iterations = []
        min_score = 100
        min_params = None

        for params in tqdm(list(ParameterGrid(all_params))):
            logger.info('params: {}'.format(params))
            clf = lgb.LGBMClassifier(**params)
            clf.fit(x_train, y_train,
                    eval_set=[(x_test, y_test)],
                    eval_metric='logloss',
                    early_stopping_rounds=early_stopping_rounds,
                    categorical_feature=categorical_feature)

            y_pred = clf.predict(x_test, num_iteration=clf.best_iteration_)
            sc_logloss = log_loss(y_test, y_pred)

            list_logloss.append(sc_logloss)
            list_best_iterations.append(clf.best_iteration_)
            logger.debug('   logloss: {}'.format(sc_logloss))

            params['n_estimators'] = int(np.mean(list_best_iterations))
            sc_logloss = np.mean(list_logloss)
            if min_score > sc_logloss:
                min_score = sc_logloss
                min_params = params
                #  break
            logger.info('logloss: {}'.format(sc_logloss))
            logger.info('current min score: {}  min params: {}'.format(
                min_score, min_params))

        logger.info('min score : {}'.format(min_score))
        logger.info('min params: {}'.format(min_params))
        best_params = pd.DataFrame(min_params, index=['params'])
        best_params.to_csv('../output/{}_{}_'.format(test_season, min_score) + 'best_params.csv')

        logger.info('train end'.format(min_score, min_params))
        logger.debug('train end'.format(min_score, min_params))

        clf = lgb.LGBMClassifier(**min_params)
        clf.fit(x_train, y_train,
                eval_set=[(x_test, y_test)],
                eval_metric='logloss',
                early_stopping_rounds=early_stopping_rounds,
                categorical_feature=categorical_feature)

        y_pred = clf.predict(x_test, num_iteration=clf.best_iteration_)
        sc_logloss = log_loss(y_test, y_pred)
        feim = pd.Series(clf.feature_importances_, name='importance')
        feature_name = pd.Series(use_cols, name='feature')
        features = pd.concat([feature_name, feim], axis=1)
        features.sort_values(by='importance', ascending=False, inplace=True)
        features.to_csv('../output/{}_{}_'.format(test_season, sc_logloss) + 'feature_importances.csv')
        season_score.append(min_score)

    mean_score = np.mean(season_score)
    logger.info('mean_score: {}'.format(mean_score))
    logger.debug('mean_score: {}'.format(mean_score))


def prediction(feature_list):
    global data

    start_time = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())
    pred_dict = {}
    feature_dict = {}
    score_list = []

    #  for season in tqdm(train_season_list):
    output = data.reset_index().copy()
    #  df = data[feature_list+['score', 'daynum']].copy()
    df = data[feature_list+['score_2', 'daynum']].copy()
    df = df.loc[train_season_list, :]
    df.reset_index(inplace=True)
    test_season = df['season'].max()

    tmp_train  = df[df['season'] != test_season]
    tmp_test   = df[df['season'] == test_season]
    tmp_train2 = tmp_test[tmp_test['daynum']<133]

    train = pd.concat([tmp_train, tmp_train2], axis=0)
    test = tmp_test[tmp_test['daynum']>=133]

    #  x_train = train.drop(['score', 'season'], axis=1).copy()
    #  y_train = train['score'].values
    #  x_test = test.drop(['score', 'season'] , axis=1).copy()
    #  y_test = test['score'].values

    x_train = train.drop(['score_2', 'season'], axis=1).copy()
    y_train = train['score_2'].values
    x_test = test.drop(['score_2', 'season'] , axis=1).copy()
    y_test = test['score_2'].values

    use_cols = x_train.columns.values

    print('***TRAIN COLUMN NUMBER***')
    print(len(x_train.columns))
    print('***TRAIN ROW NUMBER***')
    print(len(x_train))
    print('***TEST ROW NUMBER***')
    print(len(x_test))

    reg = lgb.LGBMRegressor(**fix_params)
    reg.fit(x_train, y_train,
            eval_set=[(x_test, y_test)],
            eval_metric='l2',
            early_stopping_rounds=early_stopping_rounds,
            categorical_feature=categorical_feature)

    #  out_train = output[use_cols]
    #  out_test  = output['score'].values
    #  out_test  = output['score_2'].values

    #  print('***PREDICTION COLUMN NUMBER***')
    #  print(len(out_train.columns))
    #  print('**************************')
    #  print('***PEDICTION ROW NUMBER***')
    #  print(len(out_train))
    #  print('**************************')

    #  y_pred = reg.predict(out_train, num_iteration=reg.best_iteration_)

    #  pred = pd.Series(y_pred, name='prediction_score').to_frame()
    #  pred = pd.Series(y_pred, name='prediction_score_2').to_frame()
    #  pred['observation'] = out_test
    #  train_score = pd.concat([output[['teamid', 'teamid_2', 'season', 'daynum', 'daynum_2']], pred], axis=1)
    #  train_score.to_csv('../output/{}_{}_ncaa_pred_score_8ver.csv'.format(start_time, test_season), index=False)
    #  train_score.to_csv('../output/{}_{}_ncaa_pred_score_2_8ver.csv'.format(start_time, test_season), index=False)

    #  return

    submit_data = submit_feature[submit_feature['season'] == 2018][use_cols].copy()
    submit_pred = reg.predict(submit_data, num_iteration=reg.best_iteration_)

    submit_score = submit_base[submit_base['season']==2018].copy()
    submit_score['prediction_score_2'] = submit_pred
    #  submit_score['prediction_score_2'] = submit_pred

    submit_score.to_csv('../output/{}_2018_ncaa_score_2_pred_for_submit.csv'.format(start_time), index=False)

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
