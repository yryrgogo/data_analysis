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
from make_submission import make_submission

if model_type=='lgb':
    params = train_params_dima()
    params['learning_rate'] = learning_rate
    params['num_iterations'] = num_iterations
    params['early_stopping_rounds'] = early_stopping_rounds
elif model_type=='xgb':
    params = xgb_params_0814()
    params['eta'] = learning_rate
elif model_type=='extra':
    params = extra_params()

start_time = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())

key = 'SK_ID_CURR'
target = 'TARGET'


def main():
    base = pd.read_csv('../input/base.csv')
    base[target] = base[target].where(base[target]>=0, np.nan)
    print(base[target].drop_duplicates())
    sys.exit()
    base.to_csv('../input/base.csv', index=False)
    sys.exit()

    data = make_feature_set(data, path)

    #  keras_1 = pd.read_csv('../output/20180829_105454_442features_auc0.71133_keras_prediction.csv')
    #  keras_1.fillna(0, inplace=True)
    #  t_value_1 = keras_1[target].values
    #  p_value_1 = keras_1['prediction'].values
    #  keras_1['prediction'] = t_value_1 + p_value_1
    #  data['emb_buro_prev'] = keras_1['prediction']

    data.set_index(key, inplace=True)

    submit = pd.read_csv('../data/sample_submission.csv')

    dummie=0
    seed_num=1
    path = f'../features/3_winner/*.npy'

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


if __name__ == '__main__':
    main()
