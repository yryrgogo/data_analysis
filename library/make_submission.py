import gc
import sys
import numpy as np
import pandas as pd
import datetime
import os
HOME = os.path.expanduser('~')

sys.path.append(f'{HOME}/kaggle/github/model/')
from lgbm_clf import prediction, cross_prediction

sys.path.append(f"{HOME}/kaggle/github/library/")
import utils
from preprocessing import factorize_categoricals, get_dummies
from convinience_function import get_categorical_features, get_numeric_features

start_time = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())


def make_submission(logger, data, key, target, submit, params, model_type, dummie=0, seed_num=1):

    logger.info(f'''
    #==============================================================================
    # DATA CHECK START
    #==============================================================================''')
    categorical_feature = get_categorical_features(data, [])
    logger.info(f'''
                CATEGORICAL FEATURE: {categorical_feature}
                LENGTH: {len(categorical_feature)}
                DUMMIE: {dummie}
                ''')

    if dummie==0:
        data = factorize_categoricals(data, categorical_feature)
        categorical_feature=[]
    elif dummie==1:
        data = get_dummies(data, categorical_feature)
        categorical_feature=[]

    train = data[data[target]>=0]
    test = data[data[target].isnull()]
    logger.info(f'TRAIN SHAPE: {train.shape}')
    logger.info(f'TEST SHAPE: {test.shape}')

    for col in test.columns:
        length = len(test[col].drop_duplicates())
        if length <=1:
            logger.info(f'***********WARNING************* LENGTH {length} COLUMN: {col}')

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
    logger.info(f'SEED LIST: {seed_list}')

    logger.info(f'''
    #==============================================================================
    # DATA CHECK END
    #==============================================================================''')

    tmp_result = np.zeros(len(test))
    score_list = []

    for i, seed in enumerate(seed_list):
        if model_type=='lgb':
            submit_params['bagging_seed'] = seed
            submit_params['data_random_seed'] = seed
            submit_params['feature_fraction_seed'] = seed
            submit_params['random_seed'] = seed
        elif model_type=='xgb':
            submit_params['seed'] = seed
        elif model_type=='extra':
            submit_params['seed'] = seed

        if pred_type==0:
            ' 予測 '
            result,feature_num = prediction(
                logger=logger,
                train=train,
                test=test,
                target=target,
                categorical_feature=categorical_feature,
                params = params,
                num_iterations=num_iterations,
                learning_rate=learning_rate,
                model_type=model_type
            )
            score = '?'
        elif pred_type==1:
            ' 予測 '
            y_pred, score, feature_num, stack = cross_prediction(
                logger=logger,
                train=train,
                test=test,
                target=target,
                categorical_feature=categorical,
                val_col=val_col,
                params = params,
                num_iterations=num_iterations,
                learning_rate=learning_rate,
                early_stopping_rounds=early_stopping_rounds,
                model_type=model_type
            )

            ' for stacking pred_value '
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

    result = tmp_result / len(seed_list)

    submit[target] = result
    if pred_type==0:
        submit.to_csv(f'../submit/{start_time[:12]}_submit_{model_type}_rate{learning_rate}_{feature_num}features_CV{score}_LB_early{early_stopping_rounds}_iter{num_iterations}.csv', index=False)
    elif pred_type==1:
        submit.to_csv(f'../submit/{start_time[:12]}_submit_{model_type}_rate{learning_rate}_{feature_num}features_CV{score_avg}_LB_early{early_stopping_rounds}_iter{num_iterations}.csv', index=False)

    if len(stack)>0:
        result_stack[target] = result_stack[target].values / len(seed_list)
        result_stack = base[key].to_frame().merge(result_stack, on=key, how='inner')
        result_stack.to_csv(f'../output/{start_time[:12]}_stack_{model_type}_rate{learning_rate}_{feature_num}features_CV{score_avg}_LB_early{early_stopping_rounds}_iter{num_iterations}.csv', index=False)
        logger.info(f'result_stack shape: {result_stack.shape}')

