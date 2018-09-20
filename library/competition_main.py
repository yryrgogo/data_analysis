import gc
import sys

try:
    model_type=sys.argv[2]
except IndexError:
    model_type='lgb'
try:
    learning_rate = float(sys.argv[3])
except IndexError:
    learning_rate = 0.02
try:
    early_stopping_rounds = int(sys.argv[4])
except IndexError:
    early_stopping_rounds = 150
num_iterations = 20000

try:
    pred_type = int(sys.argv[5])
except IndexError:
    pred_type = 1

import numpy as np
import pandas as pd
import datetime
import os
HOME = os.path.expanduser('~')

sys.path.append(f'{HOME}/kaggle/github/model/')
from params_lgbm import xgb_params_0814, train_params_dima

sys.path.append(f"{HOME}/kaggle/github/library/")
import utils
from make_file import make_feature_set
from experiment import first_train, much_feature_validation
from params_tune import params_scoring, params_optimize
from make_submission import make_submission

if model_type=='lgb':
    train_params = train_params_dima()
    train_params['learning_rate'] = learning_rate
    train_params['random_seed'] = seed
    train_params['bagging_seed'] = seed
    train_params['feature_fraction_seed'] = seed
    train_params['data_random_seed'] = seed
elif model_type=='xgb':
    train_params = xgb_params_0814()
    train_params['learning_rate'] = learning_rate
elif model_type=='ext':
    train_params = extra_params()
elif model_type=='lgr':
    train_params = lgr_params()

start_time = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())

#========================================================================
# Global Variables
#========================================================================
metric='auc'
dummie=0
seed_num=1


def main():

    do_code = [
        ,'submit'
        ,''
        ,'much'
        ,'params'
    ][0]

    if do_code=='much':
        much_feature_validation(base, path='../features/1_first_valid/', move_path='../features/1_second_valid/', dummie=0, val_col=val_col)

    elif do_code=='':
        ' Once Cross Validation '
        first_train(base, path, dummie=0, val_col=val_col)
        sys.exit()

    elif do_code=='params':
        ' パラメータのベイズ最適化 '
        params_optimize(base=base, path=path, metric=metric)
        sys.exit()

    elif do_code=='submit':

        submit = pd.read_csv('../data/sample_submission.csv')

        make_submission(
            logger=logger
            ,data=data
            ,key=key
            ,target=target
            ,submit=submit
            ,params=params
            ,model_type=model_type
            ,dummie=dummie
            ,seed_num=seed_num
        )

    elif do_code=='increase':
        ' STEP WISE INCREASE ONE '
        incremental_increase(base, input_path='../features/1_first_valid/*.npy', move_path='../features/1_second_valid/', dummie=0)

    elif do_code=='decrease':
        ' STEP WISE DECREASE '
        incremental_decrease(base, path=path, decrease_path=decrease_path, decrease_word=decrease_word, dummie=0, val_col=val_col, iter_no=iter_no)

if __name__ == '__main__':
    main()
