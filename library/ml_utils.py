import os
import re
import gc
import sys
import pickle
import subprocess
import glob
from contextlib import contextmanager
from datetime import datetime
from time import time, sleep
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from itertools import permutations
from joblib import Parallel, delayed

import seaborn as sns
from matplotlib import pyplot as plt

# Model
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, log_loss, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, GroupKFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso

#========================================================================
# original library 
HOME = os.path.expanduser('~')
sys.path.append(f"{HOME}/kaggle/data_analysis/library/")
import utils
#========================================================================

#  import eli5
#  from eli5.sklearn import PermutationImportance

#  perm = PermutationImportance(rfc_model, random_state=1).fit(val_X, val_y)
#  eli5.show_weights(perm, feature_names = val_X.columns.tolist(), top=150)

# CatBoost
#  train_pool = Pool(train_X,train_y)
#  cat_model = CatBoostClassifier(
#                                 iterations=3000,# change 25 to 3000 to get best performance 
#                                 learning_rate=0.03,
#                                 objective="Logloss",
#                                 eval_metric='AUC',
#                                )
#  cat_model.fit(train_X,train_y,silent=True)
#  y_pred_cat = cat_model.predict(X_test)


@contextmanager
def timer(name):
    t0 = time()
    yield
    print(f'''
#========================================================================
# [{name}] done in {time() - t0:.0f} s
#========================================================================
          ''')

#========================================================================
# make feature set
def get_train_test(feat_path_list, base=[], target='target'):
    print(base.shape)
    feature_list = utils.parallel_load_data(path_list=feat_path_list)
    df_feat = pd.concat(feature_list, axis=1)
    df_feat = pd.concat([base, df_feat], axis=1)
    train = df_feat[~df_feat[target].isnull()].reset_index(drop=True)
    test = df_feat[df_feat[target].isnull()].reset_index(drop=True)

    return train, test
#========================================================================


def Regressor(model_type, x_train, x_val, y_train, y_val, x_test, params={}, seed=1208, get_score='rmse'):

    if model_type=='linear':
        estimator = LinearRegression(**params)
    elif model_type=='ridge':
        estimator = Ridge(**params)
    elif model_type=='lasso':
        estimator = Lasso(**params)
    elif model_type=='rmf':
        params['n_jobs'] = -1
        params['n_estimators'] = 10000
        estimator = RandomForestRegressor(**params)
    elif model_type=='lgb':
        if len(params.keys())==0:
            metric = 'auc'
            params['n_jobs'] = -1
            params['metric'] = metric
            params['num_leaves'] = 31
            params['colsample_bytree'] = 0.3
            params['lambda_l2'] = 1.0
            params['learning_rate'] = 0.01
        early_stopping_rounds = 150
        num_boost_round = 30000
        params['objective'] = 'regression'

    #========================================================================
    # Fitting
    if model_type!='lgb':
        estimator.fit(x_train.values, y_train.values)
    else:
        lgb_train = lgb.Dataset(data=x_train, label=y_train)
        lgb_val = lgb.Dataset(data=x_val, label=y_val)

        cat_cols = utils.get_categorical_features(df=x_train)

        estimator = lgb.train(
            params = params
            ,train_set = lgb_train
            ,valid_sets = lgb_val
            ,early_stopping_rounds = early_stopping_rounds
            ,num_boost_round = num_boost_round
            ,categorical_feature = cat_cols
            ,verbose_eval = 200
        )
    #========================================================================

    #========================================================================
    # Prediction
    oof_pred = estimator.predict(x_val.values)
    y_pred = estimator.predict(x_test.values)
    #========================================================================

    #========================================================================
    # Scoring
    if get_score=='auc':
        score = roc_auc_score(y_val, oof_pred)
    else:
        from sklearn.metrics import roc_auc_score, log_loss, r2_score, mean_squared_error
        score = np.sqrt(mean_squared_error(y_val, oof_pred))
        r2_score = r2_score(y_val, oof_pred)
        print(f"""
        # R2 Score: {r2_score}
        """)
    # Model   : {model_type}
    # feature : {x_train.shape, x_val.shape}
    #========================================================================

    feim = get_tree_importance(estimator=estimator, use_cols=x_train.columns)

    return score, oof_pred, y_pred, feim


def Classifier(model_type, x_train, x_val, y_train, y_val, x_test, params={}, seed=1208, get_score='auc'):

    if model_type=='lgr':
        params['n_jobs'] = -1
        estimator = LogisticRegression(**params)
    if model_type=='rmf':
        params['n_jobs'] = -1
        params['n_estimators'] = 10000
        estimator = RandomForestClassifier(**params)
    elif model_type=='lgb':
        if len(params.keys())==0:
            metric = 'auc'
            params['n_jobs'] = 32
            params['metric'] = metric
            params['num_leaves'] = 31
            params['colsample_bytree'] = 0.3
            params['lambda_l2'] = 1.0
            params['learning_rate'] = 0.01
            params['objective'] = 'binary'

        if 'early_stopping_rounds' in list(params.keys()):
            early_stopping_rounds = params['early_stopping_rounds']
        else:
            early_stopping_rounds = 100
        num_boost_round = 30000

    #========================================================================
    # Fitting
    if model_type!='lgb':
        estimator.fit(x_train.values, y_train.values)
    else:
        lgb_train = lgb.Dataset(data=x_train, label=y_train)
        lgb_val = lgb.Dataset(data=x_val, label=y_val)
        #  cat_cols = utils.get_categorical_features(df=x_train)
        cat_cols = []

        estimator = lgb.train(
            params = params
            ,train_set = lgb_train
            ,valid_sets = lgb_val
            ,early_stopping_rounds = early_stopping_rounds
            ,num_boost_round = num_boost_round
            ,categorical_feature = cat_cols
            ,verbose_eval = 200
        )
    #========================================================================

    #========================================================================
    # Prediction
    if model_type=='lgb':
        oof_pred = estimator.predict(x_val.values)
        if len(x_test):
            y_pred = estimator.predict(x_test.values)
        else:
            y_pred = []
    else:
        oof_pred = estimator.predict_proba(x_val.values)[:, 1]
        if len(x_test):
            y_pred = estimator.predict_proba(x_test.values)[:, 1]
        else:
            y_pred = []

    if get_score=='auc':
        score = roc_auc_score(y_val, oof_pred)
    elif get_score=='logloss':
        score = log_loss(y_val, oof_pred)
    #  print(f"""
    #  Model: {model_type}
    #  feature: {x_train.shape, x_val.shape}
    #  {get_score}: {score}
    #  """)

    feim = get_tree_importance(estimator=estimator, use_cols=x_train.columns)

    return score, oof_pred, y_pred, feim


def get_kfold(train, Y, fold_type='kfold', fold_n=5, seed=1208, shuffle=True):
    if fold_type=='stratified':
        folds = StratifiedKFold(n_splits=fold_n, shuffle=True, random_state=seed)
    elif fold_type=='kfold':
        folds = KFold(n_splits=fold_n, shuffle=True, random_state=seed)
#     elif fold_type=='group':
#         folds = GroupKFold(n_splits=fold_n, shuffle=True, random_state=seed)

    kfold = list(folds.split(train, Y))

    return kfold


def get_best_blend_ratio(df_pred, y_train, min_ratio=0.01):
    """
    min_ratio: blendを行う比率の最低値
    """

    # 予測値のカラム名
    pred_cols = list(df_pred.columns)

    # blendを行う予測の数
    blend_num = len(pred_cols)

    ratio_list = np.arange(min_ratio, 1.0, min_ratio)
    ratio_combi_list = list(permutations(ratio_list, blend_num))
    ratio_combi_list = [
        combi for combi in ratio_combi_list if np.sum(combi) == 1]

    with timer(f"combi num:{len(ratio_combi_list)}"):

        def get_score_ratio(ratio_combi):

            y_pred_avg = np.zeros(len(df_pred))
            for col, ratio in zip(df_pred.columns, ratio_combi):
                y_pred_avg += df_pred[col].values*ratio
            score = roc_auc_score(y_train, y_pred_avg)
            return [score] + list(ratio_combi)

        score_list = Parallel(
            n_jobs=-1)([delayed(get_score_ratio)(ratio_combi) for ratio_combi in ratio_combi_list])

        df_score = pd.DataFrame(score_list, columns=['score'] + pred_cols)
        df_score.sort_values(by='score', ascending=False, inplace=True)
        best = df_score.iloc[0, :]
        print(f"Best Blend: \n{best}")
        print(df_score.head(10))

        df_corr = pd.DataFrame(np.corrcoef(df_pred.T.values), columns=pred_cols)
        print(f"Corr: \n{df_corr}")

    return best, df_corr


def get_tree_importance(estimator, use_cols):
    feim = estimator.feature_importance(importance_type='gain')
    feim = pd.DataFrame([np.array(use_cols), feim]).T
    feim.columns = ['feature', 'importance']
    feim['importance'] = feim['importance'].astype('float32')
    return feim


def display_importances(feim, viz_num = 100):
    cols = feim[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:viz_num].index
    best_features = feim.loc[feim.feature.isin(cols)]

    fig = plt.figure(figsize=(12, 20))
    fig.patch.set_facecolor('white')
    sns.set_style("whitegrid")
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig('lgb_importances.png')
    plt.show()


def get_feat_path_list(col_list, file_key='', feat_path='../features/all_features/*.gz'):
    '''
    Explain:
	カラム名の一部が入ったリストを受け取り、featureを検索して
	featureのpathのリストを返す
    Args:
    Return:
    '''

    get_list = []
    path_list = glob.glob(feat_path)

    for path in path_list:
        filename = re.search(r'/([^/.]*).gz', path).group(1)
        for col in col_list:
            if filename.count(col) and filename[:3]==col[:3] and filename.count(file_key):
                get_list.append(path)

    get_list = list(set(get_list))
    return get_list


def split_train_test(df, target):
    train = df[~df[target].isnull()]
    test = df[df[target].isnull()]
    return train, test
