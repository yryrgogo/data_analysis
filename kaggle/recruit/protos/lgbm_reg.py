import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import ParameterGrid, StratifiedKFold, train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
import datetime
from tqdm import tqdm
import sys
from logger import logger_func
from recruit_kaggle_load import RMSLE
sys.path.append('../../../module')
from load_data import load_data, x_y_split

start_time = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())
logger = logger_func()


# model params
#  metric = 'rmsle'
metric = 'l2'
categorical_feature = []
early_stopping_rounds = 1000
seed = 2018
all_params = {
    'objective': ['regression'],
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
    'objective': 'regression',
    'num_leaves': 63,
    'learning_rate': 0.1,
    'n_estimators': 120,
    'feature_fraction': 0.7,
    'random_state': seed,

    'sub_feature': 0.7,
    'min_data': 100,
    'min_hessian': 1,
    'verbose': -1,
}


def sc_metrics(test, pred):
    if metric == 'l2':
    #      return r2_score(test, pred)
    #  if metric == 'rmsle':
        return RMSLE(test, pred)
    else:
        print('score caliculation error!')


def validation(train, test, target, categorical_feature, pts_score='0', test_viz=[]):

    x_train, y_train = x_y_split(train, target)
    x_test, y_test = x_y_split(test, target)

    use_cols = list(x_train.columns.values)

    y_train = np.log1p(y_train)
    y_test = np.log1p(y_test)

    logger.info('metric:{}'.format(metric))
    logger.info('train columns: {} \nuse columns: {}'.format(len(use_cols), use_cols))

    reg = lgb.LGBMRegressor(**fix_params)
    reg.fit(x_train, y_train,
            eval_set=[(x_test, y_test)],
            eval_metric=metric,
            early_stopping_rounds=early_stopping_rounds,
            categorical_feature=categorical_feature)

    y_pred = reg.predict(x_test, num_iteration=reg.best_iteration_)

    y_test = np.expm1(y_test)
    y_pred = np.expm1(y_pred)

    sc_score = sc_metrics(y_test, y_pred)

    logger.info('prediction score: {}'.format(sc_score))

    """feature importance output file"""
    ftim = pd.DataFrame({'feature': use_cols, 'importance': reg.feature_importances_})
    ftim['score_cv'] = sc_score

    """特徴量セットに予測結果をJOINして出力"""
    if len(test_viz) != 0:
        prediction = reg.predict(x_train, num_iteration=reg.best_iteration_)
        prediction = list(np.expm1(prediction))
        prediction = prediction + list(y_pred)
        test_viz['prediction'] = prediction

        ''' 予測結果を保管しておく '''
        result_pred = pd.Series(prediction, name='{}_col{}'.format(start_time[:11], len(use_cols)))
        result_pred.to_csv('../prediction/{}_col{}_{}.csv'.format(start_time[:11], len(use_cols), sc_score), index=False, header=True)

        """ パーティション毎にスコアを出力し、格納する """
        if pts_score!='0':
            df_pts = test_viz[[pts_score, target, 'prediction']]

            pts_list = df_pts[pts_score].drop_duplicates().values
            pts_score_list = []
            for pts in pts_list:
                tmp = df_pts[df_pts[pts_score]==pts]
                y_test_pts = tmp[target].values
                y_pred_pts = tmp['prediction'].values
                score = sc_metrics(y_test_pts, y_pred_pts)
                pts_score_list.append(score)

            result_pts = pd.DataFrame({pts_score:pts_list, 'pts_score':pts_score_list})

            test_viz = pd.merge(test_viz, result_pts, on=pts_score, how='left', copy=False)

        test_viz.to_csv('../eda/{}_test_viz_{}_{}.csv'.format(
            start_time[:11], metric, sc_score), index=False)

    return ftim


def cross_validation(data):

    df = data.copy()
    score_list = []
    list_best_iterations = []
    best_params = None
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    best_score = 100

    train, test = train_test_split(df[use_cols], test_size=test_size, random_state=seed)
    x_train, y_train = x_y_split(train, target)
    x_test, y_test = x_y_split(test, target)

    for params in tqdm(list(ParameterGrid(all_params))):
        logger.info('params: {}'.format(params))

        for train_idx, valid_idx in tqdm(cv.split(x_train, y_train)):
            trn_x = x_train.iloc[train_idx, :]
            val_x = x_train.iloc[valid_idx, :]

            trn_y = y_train[train_idx]
            val_y = y_train[valid_idx]
            trn_y = np.log1p(trn_y)
            val_y = np.log1p(val_y)

            reg = lgb.LGBMRegressor(**params)
            reg.fit(trn_x, trn_y,
                    eval_set=[(val_x, val_y)],
                    eval_metric=metric,
                    early_stopping_rounds=early_stopping_rounds,
                    categorical_feature=categorical_feature)

            y_pred = reg.predict(val_x, num_iteration=reg.best_iteration_)
            sc_score = sc_metrics(val_y, y_pred)

            score_list.append(sc_score)
            list_best_iterations.append(reg.best_iteration_)
            logger.debug('{}: {}'.format(metric, sc_score))

        logger.info('CV end')

        params['n_estimators'] = int(np.mean(list_best_iterations))
        mean_score = np.mean(score_list)

        if best_score > mean_score:
            best_score = mean_score
            best_params = params

        logger.info('{}: {}'.format(metric, mean_score))
        logger.info('current {}: {}  best params: {}'.format(
            metric, best_score, best_params))

    logger.info('CV best score : {}'.format(best_score))
    logger.info('CV best params: {}'.format(best_params))

    # params output file
    df_params = pd.DataFrame(best_params, index=['params'])
    df_params.to_csv('../output/{}_best_params_{}_{}_{}.csv'.format(
        start_time[:11], metric, best_score, fs_name), index=False)

    reg = lgb.LGBMRegressor(**best_params)
    reg.fit(x_train, y_train,
            eval_set=[(x_test, y_test)],
            eval_metric=metric,
            early_stopping_rounds=early_stopping_rounds,
            categorical_feature=categorical_feature)

    y_pred = reg.predict(x_test, num_iteration=reg.best_iteration_)

    logger.info('CV & TEST mean {}: {}'.format(metric, mean_score))


def prediction(train, pred, target):

    x, y = x_y_split(train, target)
    y = np.log1p(y)

    reg = lgb.LGBMRegressor(**fix_params)
    reg.fit(x, y,
            eval_metric=metric,
            categorical_feature=categorical_feature)

    use_cols = x.columns.values

    y_pred = reg.predict(pred[use_cols])
    y_pred = np.expm1(y_pred)

    pred['visitors'] = y_pred
    result = pred[['id', 'visitors']]

    result.to_csv('../output/{}_submission.csv'.format(start_time[:11]), index=False)


