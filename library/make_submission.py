import gc
import sys
import numpy as np
import pandas as pd
import datetime
import os
from select_feature import move_feature
HOME = os.path.expanduser('~')

sys.path.append(f'{HOME}/kaggle/github/model/')
from Estimator import prediction, cross_prediction, TimeSeriesPrediction

sys.path.append(f"{HOME}/kaggle/github/library/")
import utils
from preprocessing import factorize_categoricals, get_dummies


def make_submission(submit_params):

    #========================================================================
    # For Seed Averaging
    #========================================================================
    seed_num = submit_params['seed_num']
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

    test = submit_params['test']
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
        if pred_type=='v':
            ' 予測 '
            result = prediction(
                **submit_params
            )
            score = '?'
        #========================================================================
        # Cross Validation 
        #========================================================================
        elif pred_type=='cv':
            ' 予測 '
            y_pred, tmp_score, stack = cross_prediction(
                **submit_params
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
            score_list.append(tmp_score)
            score = np.mean(score_list)
            logger.info(f'''
#==============================================================================
# CURRENT SCORE AVERAGE: {score}
#==============================================================================''')

        #========================================================================
        # Time Series Prediction
        #========================================================================
        elif pred_type=='time':
            ' 予測 '
            y_pred, score = TimeSeriesPrediction(
                logger=logger,
                train=train,
                test=test,
                key=key,
                target=target,
                val_label=val_label,
                params = params,
                model_type=model_type,
                ignore_list=ignore_list
            )
            result = y_pred

            logger.info(f'''
#==============================================================================
# CURRENT SCORE : {score}
#==============================================================================''')

    if pred_type=='cv':
        result = tmp_result / len(seed_list)

    if len(result_stack)>0:
        result_stack[target] = result_stack[target].values / len(seed_list)

    return result, score, result_stack
