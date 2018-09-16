import gc
import sys
import numpy as np
import pandas as pd
import datetime
import os
from select_feature import move_feature
HOME = os.path.expanduser('~')

sys.path.append(f'{HOME}/kaggle/github/model/')
from classifier import prediction, cross_prediction
from regression import time_prediction

sys.path.append(f"{HOME}/kaggle/github/library/")
import utils
from preprocessing import factorize_categoricals, get_dummies
from utils import get_categorical_features, get_numeric_features

start_time = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())


def make_submission(logger, data, key, target, fold, fold_type, params, model_type, dummie=1, seed_num=1, ignore_list=[], pred_type=1, stack_name='', exclude_category=True):

    #========================================================================
    # Make Train Test Dataset
    #========================================================================
    train = data[~data[target].isnull()]
    test = data[data[target].isnull()]

    #========================================================================
    # For Seed Averaging
    #========================================================================
    seed_list = [
        1208,
        308,
        1012,
        1128,
        605,
        1222,
        405,
        503,
        2018,
        1212
    ][:seed_num]
    logger.info(f'SEED AVERAGING LIST: {seed_list}')

    tmp_result = np.zeros(len(test)) # For Prediction Array
    score_list = []
    result_stack = []

    #========================================================================
    # Arrange Seed
    #========================================================================
    for i, seed in enumerate(seed_list):
        if model_type=='lgb':
            params['bagging_seed'] = seed
            params['data_random_seed'] = seed
            params['feature_fraction_seed'] = seed
            params['random_seed'] = seed
        elif model_type=='xgb':
            params['seed'] = seed
        elif model_type=='extra':
            params['seed'] = seed

        #========================================================================
        # 1 validation
        #========================================================================
        if pred_type==0:
            ' 予測 '
            result = prediction(
                logger=logger,
                train=train,
                test=test,
                target=target,
                categorical_feature=categorical_feature,
                params = params,
                model_type=model_type
            )
            score = '?'
        #========================================================================
        # Cross Validation 
        #========================================================================
        elif pred_type==1:
            ' 予測 '
            y_pred, score, stack = cross_prediction(
                logger=logger,
                train=train,
                test=test,
                key=key,
                target=target,
                fold=fold,
                fold_type=fold_type,
                categorical_feature=categorical_feature,
                params = params,
                model_type=model_type,
                ignore_list=ignore_list,
                oof_flg=len(stack_name)
            )

            #========================================================================
            # Make Oof For Stacking
            #========================================================================
            if len(stack)>0:
                if i==0:
                    result_stack = stack.copy()
                else:
                    pred_stack = stack[target].values
                    result_stack[target] = result_stack[target].values + pred_stack

                del stack
                gc.collect()

            tmp_result += y_pred
            score_list.append(score)
            score_avg = np.mean(score_list)
            logger.info(f'''
#==============================================================================
# CURRENT AUC AVERAGE: {score_avg}
#==============================================================================''')

        #========================================================================
        # Time Series
        #========================================================================
        elif pred_type==2:
            ' 予測 '
            y_pred, score, stack = time_prediction(
                logger=logger,
                train=train,
                test=test,
                key=key,
                target=target,
                fold=fold,
                fold_type=fold_type,
                categorical_feature=categorical_feature,
                params = params,
                model_type=model_type,
                ignore_list=ignore_list,
                oof_flg=len(stack_name)
            )

    result = tmp_result / len(seed_list)

    if len(result_stack)>0:
        result_stack[target] = result_stack[target].values / len(seed_list)

    return score_avg, result, result_stack


