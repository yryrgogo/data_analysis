
def params_MS():
    params = {
        "metric": 'logloss',
        'objective':'binary',
        "boosting": "gbdt",
        'num_threads': 64,
        'num_leaves': 2**8-1,
        'max_depth': -1,
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


def params_quara():
    train_params = {
        'num_threads': -1,
        'metric': 'binary_logloss',
        'objective': 'binary',
        'boosting_type':'gbdt',
        'bagging_freq': 1,
        'sigmoid': 1.1,
        'subsample': 0.9,
        'colsample_bytree': 0.2,
        'lambda_l1': 1,
        'lambda_l2': 5,
        'learning_rate': 0.1,
        #  'max_bin': 500,
        #  'min_child_samples': 90,
        #  'min_child_weight': 160,
        #  'min_data_in_bin': 40,
        #  'min_split_gain': 0,
        'num_leaves': 100,
        'max_depth': 9,
        'bagging_seed': 1208,
        'data_random_seed': 1208,
        'feature_fraction_seed': 1208,
        'random_seed': 1208,
        'verbose': 1
    }
    return train_params

def params_home_credit():

    # 81354, 8040 AUC AVG 0.809672
    train_params = {'num_threads':35, 'bagging_freq': 1, 'bagging_seed': 1208, 'colsample_bytree': 0.01, 'data_random_seed': 1208, 'feature_fraction_seed': 1208, 'lambda_l1': 0.1, 'lambda_l2': 48.0, 'learning_rate': 0.02, 'max_bin': 400, 'max_depth': 5, 'metric': 'auc', 'min_child_samples': 44, 'min_child_weight': 12, 'min_data_in_bin': 72, 'min_split_gain': 0.01, 'num_leaves': 13, 'num_threads': 35, 'objective': 'binary', 'random_seed': 1208, 'subsample': 1.0} 

    train_params = {
    'num_threads': -1,
    #  'colsample_bytree':0.01,
    'colsample_bytree':0.2,
    'subsample':0.9,
    'min_split_gain':0,
    'objective':'binary',
    #  'objective':'regression',
    'boosting_type':'gbdt',
    'num_leaves':63,
    'max_depth':7,
    #  'min_child_weight':36,
    #  'max_bin':250,
    #  'min_child_samples':96,
    #  'min_data_in_bin':96,

    'lambda_l1':0.5,
    'lambda_l2':100.,
    'random_seed': 1208,
    'bagging_seed':1208,
    'feature_fraction_seed':1208,
    'data_random_seed':1208
    }
    #  train_params = {'bagging_freq': 1, 'bagging_seed': 605, 'colsample_bytree': 0.01, 'data_random_seed': 605, 'feature_fraction_seed': 605, 'lambda_l1': 0.1, 'lambda_l2': 48.0, 'learning_rate': 0.02, 'max_bin': 400, 'max_depth': 5, 'metric': 'auc', 'min_child_samples': 100, 'min_child_weight': 76, 'min_data_in_bin': 76, 'min_split_gain': 0.01, 'num_leaves': 11, 'num_threads': 35, 'objective': 'binary', 'random_seed': 605, 'subsample': 1.0}
    return train_params


def train_params_nlp():
    train_params = {
    'boosting_type':'gbdt',
    'num_threads': -1,
    'learning_rate':0.1,
    'subsample':0.9,
    'colsample_bytree':0.80,
    'objective':'binary',
    'metric':'auc',
    'max_depth':4,
    'lambda_l1':0.1,
    'lambda_l2':0.5,
    'num_leaves':8,
    'random_seed': 1208,
    'bagging_seed':1208,
    'feature_fraction_seed':1208,
    'data_random_seed':1208
    }
    return train_params


#  0.80028
def train_params_0816():
    train_params = {
        'bagging_freq': 1,
        'bagging_seed': 1208,
        'colsample_bytree': 0.22,
        'data_random_seed': 1208,
        'feature_fraction_seed': 1208,
        'lambda_l1': 1.2000000000000002,
        'lambda_l2': 0.5,
        'learning_rate': 0.02,
        'max_bin': 1400,
        'max_depth': -1,
        'metric': 'auc',
        'min_child_samples': 170,
        'min_child_weight': 160,
        'min_data_in_bin': 120,
        'min_split_gain': 0,
        'num_leaves': 12,
        'objective': 'binary',
        'random_seed': 1208,
        'sigmoid': 1.1,
        'subsample': 0.9,
        'verbose': 1
     #  }
     #     'bagging_freq': 1,
     #     'bagging_seed': 1208,
     #     'colsample_bytree': 0.22,
     #     'data_random_seed': 1208,
     #     'feature_fraction_seed': 1208,
     #     'lambda_l1': 4.4,
     #     'lambda_l2': 3.1,
     #     'learning_rate': 0.02,
     #     'max_bin': 1600,
     #     'max_depth': -1,
     #     'metric': 'auc',
     #     'min_child_samples': 160,
     #     'min_child_weight': 130,
     #     'min_data_in_bin': 50,
     #     'min_split_gain': 0,
     #     'num_leaves': 5,
     #     'objective': 'binary',
     #     'random_seed': 1208,
     #     'sigmoid': 0.9,
     #     'subsample': 0.9,
     #     'verbose': 1
    }
    return train_params

def xgb_params_0814():
    ' 0.80003 '
    train_params = {
        'nthread': 35,
        'booster': 'gbtree',
        'colsample_bytree': 0.01,
        'eta': 0.02,
        'eval_metric': 'auc',
        'gamma': 0.1,
        'alpha': 0.1,
        'lambda': 70.0,
        'max_depth': 5,
        'min_child_weight': 12,
        'objective': 'binary:logistic',
        'seed': 1208,
        'subsample': 0.9
    }
    #  ' 0.80004 '
    #  train_params = {
    #      'alpha': 1.4000000000000001,
    #      'booster': 'gbtree',
    #      'colsample_bytree': 0.25,
    #      'eta': 0.02,
    #      'eval_metric': 'auc',
    #      'gamma': 0.5,
    #      'lambda': 3.2,
    #      'max_depth': 3,
    #      'min_child_weight': 90,
    #      'objective': 'binary:logistic',
    #      'seed': 1208,
    #      'subsample': 0.9
    #  }
    return train_params


def extra_params():
    train_params = {
    'criterion': 'gini',
    'max_features': 0.3, # 0.2 ~ 0.8くらい？
    'max_depth': None,
    'min_samples_split': 10,
    'min_samples_leaf': 5,
    'min_weight_fraction_leaf': 0.01,
    'max_leaf_nodes': 20, # 10~5000
    #  'min_impurity_split'
    'min_impurity_decrease': 0.1,
    'n_jobs': -1,
    'subsample': 0.9,
    'random_state': 1208
    }
    return train_params


def lgr_params():
    train_params = {
    'penalty': 'l2',
    'C': 0.8,
    'max_iter': 100,
    'n_jobs': -1,
    'random_state': 1208
    }
    return train_params


def train_params_0812():
    train_params = {
    'metric': 'auc',
    'bagging_freq': 1,
    'bagging_seed': 1208,
    'colsample_bytree': 0.2,
    'data_random_seed': 1208,
    'feature_fraction_seed': 1208,
    'lambda_l1': 6.0,
    'lambda_l2': 8.6,
    'learning_rate': 0.04,
    'max_bin': 1800,
    'max_depth': -1,
    'min_child_samples': 110,
    'min_child_weight': 160,
    'min_data_in_bin': 120,
    'min_split_gain': 0,
    'num_leaves': 5,
    'objective': 'binary',
    'random_seed': 1208,
    'sigmoid': 1.1,
    'subsample': 0.9,
    'verbose': 1}
    return train_params

def train_params_0811():
    train_params = {
    'bagging_freq': 1,
    'bagging_seed': 1208,
    'colsample_bytree': 0.2,
    'data_random_seed': 1208,
    'feature_fraction_seed': 1208,
    'lambda_l1': 3.3000000000000003,
    'lambda_l2': 4.3,
    'learning_rate': 0.04,
    'max_bin': 1100,
    'max_depth': -1,
    'min_child_samples': 140,
    'min_child_weight': 130,
    'min_data_in_bin': 140,
    'min_split_gain': 0,
    'num_leaves': 9,
    'objective': 'binary',
    'random_seed': 1208,
    'sigmoid': 1.0,
    'subsample': 0.9,
    'verbose': 1}
    return train_params

' EC2 '
def train_params():

    train_params = {
    'objective': 'binary',
    'bagging_freq': 1,
    'colsample_bytree': 0.5,
    'lambda_l1': 0.55,
    'lambda_l2': 0.35,
    #'learning_rate': 0.05,
    'learning_rate': 0.1,
    'max_bin': 750,
    'max_depth': -1,
    'min_child_samples': 33,
    'min_child_weight': 51,
    'min_data_in_bin': 6,
    'min_split_gain': 0,
    'num_leaves': 8,
    'sigmoid': 0.7,
    'subsample': 0.85,
    'random_seed': 1208,
    'bagging_seed':1208,
    'feature_fraction_seed':1208,
    'data_random_seed':1208
    }

    return train_params


' GCP '
def train_params_gcp():

    train_params = {
    'objective': 'binary',
    'bagging_freq': 1,
    'colsample_bytree': 0.45,
    'lambda_l1': 0.45,
    'lambda_l2': 0.05,
    'learning_rate': 0.1,
    'max_bin': 800,
    'max_depth': -1,
    'min_child_samples': 12,
    'min_child_weight': 27,
    'min_data_in_bin': 27,
    'min_split_gain': 0,
    'num_leaves': 8,
    'sigmoid': 0.6,
    'subsample': 0.9,
    'random_seed': 1208,
    'bagging_seed':1208,
    'feature_fraction_seed':1208,
    'data_random_seed':1208
    }

    return train_params


def valid_params():

    train_params = {
    'objective': 'binary',
    'bagging_freq': 1,
    'colsample_bytree': 1,
    'lambda_l1': 0.5,
    'lambda_l2': 0.25,
    'learning_rate': 0.1,
    'max_bin': 550,
    'max_depth': -1,
    'min_child_samples': 42,
    'min_child_weight': 45,
    'min_data_in_bin': 15,
    'min_split_gain': 0,
    'num_leaves': 24,
    'sigmoid': 0.7,
    'subsample': 1,
    'random_seed': 1208,
    'bagging_seed':1208,
    'feature_fraction_seed':1208,
    'data_random_seed':1208
    }

    return train_params
