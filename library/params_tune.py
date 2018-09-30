import sys
import gc
import numpy as np
import pandas as pd
import glob
import re
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from tqdm import tqdm


def params_scoring(params):
    global df

    # Global Variable
    #  from info_R_Competition import r_cmp_key_cols
    #  key, fvid, target, val_label, ignore_list = r_cmp_key_cols()
    #  judge_flg=False
    #  metric = 'rmse'
    #  fold_type='group'
    #  fold=5
    #  group_col_name='fullVisitorId'

    if model_type=='lgb':
        params["max_bin"] = int(params["max_bin"])
        params["min_data_in_bin"] = int(params["min_data_in_bin"])
        params["min_child_samples"] = int(params["min_child_samples"])
        params["num_leaves"] = int(params["num_leaves"])
        params["min_child_weight"] = int(params["min_child_weight"])
    elif model_type=='xgb':
        params["min_child_weight"] = int(params["min_child_weight"])
        params["max_depth"] = int(params["max_depth"])
    elif model_type=='ext':
        params["max_leaf_nodes"] = int(params["max_leaf_nodes"])
        params["min_samples_leaf"] = int(params["min_samples_leaf"])
        params["min_samples_split"] = int(params["min_samples_split"])
    elif model_type=='lgr':
        params["max_iter"] = int(params["max_iter"])

    cv_feim, _, _ = first_train(
        logger=logger,
        train=df,
        key=fvid,
        target=target,
        fold_type=fold_type,
        fold=fold,
        group_col_name=group_col_name,
        val_label=val_label,
        params=params,
        metric=metric,
        judge_flg=judge_flg,
        model_type=model_type,
        dummie=dummie,
        ignore_list=ignore_list
    )

    if len(result)<=1:
        return result[0] * -1

    score = result['cv_score'].values[0] * -1

    logger.info(f'optimize current score: {score}')

    return score


def params_optimize(df):

    trials = Trials()

    if model_type=='lgb':

        search_params = {
            #  'boosting': 'dart',
            'num_threads': 35,
            'metric': 'auc',
            'objective': 'binary',
            'learning_rate': 0.1,
            #  'subsample': 0.9,
            'subsample': hp.quniform("subsample", 0.5, 0.95, 0.05),
            'num_leaves': hp.quniform("num_leaves", 31, 127, 12),
            #  'num_leaves': 14,
            'max_bin': hp.quniform("max_bin", 100, 500, 100),
            'max_depth': 5,
            'min_child_samples': hp.quniform("min_child_samples", 10, 200, 10),
            'min_child_weight': hp.quniform("min_child_weight", 10, 200, 10),
            #  'min_child_weight': 18,
            'min_data_in_bin': hp.quniform("min_data_in_bin", 10, 200, 10),
            'min_split_gain': 0.01,
            'bagging_freq': 1,
            #  'colsample_bytree': 0.01,
            'colsample_bytree': hp.quniform("colsample_bytree", 0.3, 0.8, 0.1),
            #  'sigmoid': hp.quniform("sigmoid", 0.8, 1.2, 0.1),
            'random_seed': seed,
            'bagging_seed':seed,
            'feature_fraction_seed':seed,
            'data_random_seed':seed,
            #  'random_seed': 605,
            #  'bagging_seed':605,
            #  'feature_fraction_seed':605,
            #  'data_random_seed':605,
            #  'lambda_l1': 0.1,
            'lambda_l1': hp.quniform('lambda_l1', 0, 0.5, 0.1),
            'lambda_l2': hp.quniform("lambda_l2", 0.0, 0.7, 0.14),
        }

    elif model_type=='xgb':
        search_params = {
            'objective': "binary:logistic",
            'booster': "gbtree",
            'eval_metric': 'auc',
            'eta': 0.02,
            'max_depth': hp.quniform("max_depth", 4, 5, 1),
            #  'max_depth': 5,
            'gamma': hp.quniform("gamma", 0.01, 0.51, 0.05),
            'min_child_weight': hp.quniform("min_child_weight", 10, 30, 2),
            'subsample': 0.9,
            #  'colsample_bytree': hp.quniform("colsample_bytree", 0.2, 0.85, 0.05),
            #  'colsample_bytree': hp.quniform("colsample_bytree", 0.01, 0.04, 0.01),
            'colsample_bytree': 0.01,
            #  'alpha': hp.quniform("alpha", 0.1, 0.5, 0.1),
            #  'lambda': hp.quniform("lambda", 70, 90, 10),
            'alpha': 0.1,
            'lambda': 70,
            #  'lambda': 3,
            #  'alpha': 5,
            'seed': 1208
        }

    elif model_type=='ext':
        search_params = {
            'criterion': 'gini',
            'max_depth': None,
            'max_features': hp.quniform("max_features", 0.2, 0.4, 0.1),
            'max_leaf_nodes': hp.quniform("max_leaf_nodes", 10, 400, 30), # 10~5000
            'min_impurity_decrease': hp.quniform("min_impurity_decrease", 0.1, 1, 0.1),
            'min_samples_leaf': hp.quniform("min_samples_leaf", 5, 35, 3),
            'min_samples_split': hp.quniform("min_samples_split", 10, 50, 5),
            'min_weight_fraction_leaf': hp.quniform("min_weight_fraction_leaf", 0.01, 0.33, 0.04),
            'n_jobs': -1,
            'random_state': 1208
        }

    elif model_type=='lgr':
        search_params = {
            'penalty': 'l2',
            'C': hp.quniform("C", 0.1, 0.9, 0.1),
            'max_iter': hp.quniform("max_iter", 100, 1000, 100),
            'n_jobs': -1,
            'random_state': 1208
        }

    best = fmin(params_scoring, search_params, algo=tpe.suggest, trials=trials, max_evals=250)
    logger.info(f"best parameters: {best}")


def parameter_grid():

    best_score = first_train(base, feature_set_path, dummie=0, val_col='valid_no')

    for params in tqdm(list(ParameterGrid(all_params))):
        logger.info(f'params: {params}')

        sc_score = tmp_result['cv_score'].values[0]
        score_list.append(sc_score)

        if metric == 'auc':
            if best_score < sc_score:
                best_score = sc_score
                logger.info(f'\nBest Score Update!!!!!')
                logger.info(f'\nBest Params: {params}')
                tmp_result.to_csv(f'../output/params/LGBM_params_auc{sc_score}.csv', index=False)

        elif metric == 'logloss':
            if best_score > sc_score:
                best_score = sc_score

        logger.info(f'\nCurrent Best_Score: {best_score}')

