
def params_lgb():
    params = {
        'objective':'binary',
        "boosting": "gbdt",
        'num_threads': 36,
        #  'num_leaves': 2**8-1,
        'num_leaves': 2**8-1,
        'max_depth': 8,
        'learning_rate': 0.05,
        #  "min_child_samples": 20,
        #  "bagging_freq": 1,
        "subsample": 0.9 ,
        "colsample_bytree": 0.25,
        "lambda_l1": 0.1,
        "lambda_l1": 1.5,
        "verbosity": -1,
        'random_seed': 1208,
        'bagging_seed':1208,
        'feature_fraction_seed':1208,
        'data_random_seed':1208
    }
    return params
