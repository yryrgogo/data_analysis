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

sample_path = '../input/2018/SampleSubmissionStage2_SampleTourney2018.csv'
sample = pd.read_csv(sample_path)
sample['season'] = sample.apply(lambda x:int(x.ID.split('_')[0]), axis=1)
sample['teamid'] = sample.apply(lambda x:int(x.ID.split('_')[1]), axis=1)
sample['teamid_2'] = sample.apply(lambda x:int(x.ID.split('_')[2]), axis=1)
sample.drop('Pred', axis=1, inplace=True)
submit = sample

#  input_path = '../input/20180314_last_ncaa_feature_set.csv'
input_path = '../input/20180315_223852_ncaa_feature_set.csv'
data = pd.read_csv(input_path)
data.drop(['score', 'score_2'], axis=1, inplace=True)

#  for col in data.columns:
#w      print(data[col].head())
#  sys.exit()
#  data = data[['teamid', 'teamid_2', 'season', 'daynum', 'daynum_2']]

#  score = pd.read_csv('../feature_enginering/20180314_2017_ncaa_pred_score_8ver.csv')
#  score.drop('observation', axis=1, inplace=True)
#  score_2 = pd.read_csv('../feature_enginering/20180314_2017_ncaa_pred_score_2_8ver.csv')
#  score_2.drop('observation', axis=1, inplace=True)
#  pred_score = score.merge(score_2, on= ['teamid', 'teamid_2', 'season', 'daynum', 'daynum_2'], how='inner')
#  pred_score['pred_score_diff'] = pred_score.apply(lambda x:x.prediction_score - x.prediction_score_2, axis=1)
#  pred_score['pred_flg'] = pred_score['pred_score_diff'].map(lambda x: 1 if x>15 else -1 if x<-15 else x)
#  pred_score.drop(['prediction_score', 'prediction_score_2', 'pred_score_diff'], axis=1, inplace=True)
#  data = data.merge(pred_score, on= ['teamid', 'teamid_2', 'season', 'daynum', 'daynum_2'], how='inner')

data.set_index('season', inplace=True)

#  test_season_list = [2014, 2015, 2016, 2017]

#  train_season_list = []
# 8year train set
train_season_list = np.arange(1993, 2018, 1)
#  train_season_list = [np.arange(1993, 2017, 1), np.arange(1993, 2018, 1) ]
#  categorical_feature = ['teamid_label', 'teamid_2_label', 'location', 'coachname', 'coachname_2']
categorical_feature = ['teamid_label', 'teamid_2_label', 'coachname', 'coachname_2']

# cluster

# submit
#  submit_path = '../input/20180315_061850_ncaa_submit_feature.csv'
submit_path = '../input/20180315_224433_ncaa_submit_feature.csv'
submit_feature = pd.read_csv(submit_path)
#  print(submit_feature.columns)
#  sys.exit()
#  score = pd.read_csv('../feature_enginering/20180315_last_2018_ncaa_score_pred_for_submit.csv')
#  score_2 = pd.read_csv('../feature_enginering/20180315_last_2018_ncaa_score_2_pred_for_submit.csv')
#  pred_score = score.merge(score_2, on= ['teamid', 'teamid_2', 'season'], how='inner')
#  pred_score['pred_score_diff'] = pred_score.apply(lambda x:x.prediction_score - x.prediction_score_2, axis=1)
#  pred_score['pred_flg'] = pred_score['pred_score_diff'].map(lambda x: 1 if x>15 else -1 if x<-15 else x)
#  pred_score.drop(['prediction_score', 'prediction_score_2', 'pred_score_diff'], axis=1, inplace=True)
#  submit_feature = submit_feature.merge(pred_score, on= ['teamid', 'teamid_2', 'season'], how='inner')

submit_base = submit_feature[['teamid', 'teamid_2', 'season']].copy()

# feature selection
feim_result = pd.read_csv('../feature_enginering/2017_9.79468578925634_feature_importances.csv')
fe_sel = feim_result.groupby(['feature'], as_index=False)['importance'].mean()
fe_sel.sort_values(by='importance', ascending=False, inplace=True)
fe_list1 = list(fe_sel['feature'].values[:100])
fe_list2 = list(fe_sel['feature'].values[:200])
#  fe_list3 = list(fe_sel['feature'].values[:300])
#  fe_list4 = list(fe_sel['feature'].values[:400])

early_stopping_rounds = 10000
all_params = {
    'objective': ['binary'],
    'num_leaves': [511, 1023],
    'learning_rate': [0.05, 0.01],
    'n_estimators': [100, 300],
    'feature_fraction': [0.7, 0.8, 0.9],
    'random_state':[2018]
}

fix_params = {
    'objective': 'binary',
    #  'num_leaves': 1023,
    'num_leaves': 511,
    'learning_rate': 0.05,
    'n_estimators': 120,
    'feature_fraction': 0.7,
    'random_state':2018
}


def main():

    #  for f in [fe_list2, fe_list3]:
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
        print(test_season)
        sys.exit()

        use_cols = x_train.columns.values

        logger.debug('train columns: {} {}'.format(use_cols.shape, use_cols))
        logger.info('data preparation end {}'.format(x_train.shape))

        list_logloss = []
        list_best_iterations = []
        min_score = 100
        min_params = None

        for params in tqdm(list(ParameterGrid(all_params))):
            logger.info('params: {}'.format(params))

            #  for col in x_train.columns:
            #      print(x_train[col].head(2))
            #  sys.exit()
            clf = lgb.LGBMClassifier(**params)
            clf.fit(x_train, y_train,
                    eval_set=[(x_test, y_test)],
                    eval_metric='logloss',
                    early_stopping_rounds=early_stopping_rounds,
                    categorical_feature=categorical_feature)

            y_pred = clf.predict_proba(x_test, num_iteration=clf.best_iteration_)[:,1]
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

    tmp_train  = df[df['season'] != test_season]
    tmp_test   = df[df['season'] == test_season]
    tmp_train2 = tmp_test[tmp_test['daynum']<133]

    train = pd.concat([tmp_train, tmp_train2], axis=0)
    test = tmp_test[tmp_test['daynum']>=133]

    x_train = train.drop(['result', 'season'], axis=1).copy()
    y_train = train['result'].values
    x_test = test.drop(['result', 'season'] , axis=1).copy()
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

    tmp_y_pred = clf.predict_proba(x_test, num_iteration=clf.best_iteration_)[:,1]
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

    submit_data = submit_feature[submit_feature['season'] == 2018][use_cols].copy()
    submit_pred = clf.predict_proba(submit_data[use_cols], num_iteration=clf.best_iteration_)[:,1]

    submit_score = submit_base[submit_base['season']==2018].copy()
    submit_score['Pred'] = submit_pred
    submit_result = submit.merge(submit_score, on= ['teamid', 'teamid_2', 'season'], how='inner')[['ID', 'Pred']]

    submit_result.to_csv('../output/{}_2018_ncaa_submit.csv'.format(start_time), index=False)

    #  submit_result['Pred'] = submit_result['Pred'].map(lambda x: x*1.35 if x>0.7 else x*0.1 if x<0.3 else x)
    #  submit_result['Pred'] = submit_result['Pred'].map(lambda x: 0.99 if x>1.0 else 0.01 if x<0.0 else x)

    submit_result.to_csv('../output/{}_2018_ncaa_submit_adjust.csv'.format(start_time), index=False)

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
