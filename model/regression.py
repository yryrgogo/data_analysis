import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.metrics import r2_score, mean_squared_error
import datetime
from tqdm import tqdm
import sys
from params_lgbm import train_params_0729
sys.path.append('../../../github/module')
from load_data import load_data
from preprocessing import set_validation, split_dataset


def sc_metrics(test, pred):
    if metric=='l2':
        return r2_score(test, pred)
    elif metric=='rmse':
        return np.sqrt(mean_squared_error(test, pred))


def TimeSeriesPrediction(logger, train, test, key, target, val_label, categorical_feature=[], metric='rmse', params={}, num_iterations=3000, learning_rate=0.1, early_stopping_rounds=150, model_type='lgb', ignore_list=[]):
    '''
    Explain:
    Args:
    Return:
    '''

    ' Data Check '
    train = data_check(logger, df=train, target=target)
    test = data_check(logger, df=test, target=target, test=True)

    ' Make Train Set & Validation Set  '
    x_train = train.query("val_label==0")
    x_val = train.query("val_label==1")
    y_train = x_train[target]
    y_val = x_val[target]
    logger.info(f'''
#========================================================================
# X_Train Set : {x_train.shape}
# X_Valid Set : {x_val.shape}
#========================================================================''')

    ' Logarithmic transformation '
    y_train = np.log1p(y_train)
    y_val = np.log1p(y_val)

    use_cols = [f for f in train.columns if f not in ignore_list]
    x_train = x_train[use_cols]
    x_val = x_val[use_cols]

    if model_type=='xgb':
        " XGBはcolumn nameで'[]'と','と'<>'がNGなのでreplace "
        if i==0:
            test = test[use_cols]
        use_cols = []
        for col in x_train.columns:
            use_cols.append(col.replace("[", "-q-").replace("]", "-p-").replace(",", "-o-"))
        x_train.columns = use_cols
        x_val.columns = use_cols
        test.columns = use_cols

    Model, y_pred = regression(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        params=params,
        categorical_feature=categorical_feature,
        num_iterations=num_iterations,
        early_stopping_rounds=early_stopping_rounds,
        learning_rate=learning_rate,
        model_type=model_type
    )

    ' 対数変換を元に戻す '
    y_train = np.expm1(y_train)
    y_pred = np.expm1(y_pred)
    sc_score = sc_metrics(y_train, y_pred)

    if model_type != 'xgb':
        test_pred = Model.predict(test[use_cols])
    elif model_type == 'xgb':
        test_pred = Model.predict(xgb.DMatrix(test))

    ' Feature Importance '
    feim_name = f'importance'
    feim = df_feature_importance(model=Model, model_type=model_type, use_cols=use_cols, feim_name=feim_name)
    feim.sort_values(by=f'avg_importance', ascending=False, inplace=True)
    feim['rank'] = np.arange(len(feim))+1

    feim.to_csv(f'../output/feature{len(feim)}_importances_{metric}_{score}.csv', index=False)

    return y_pred, feim


def regression(x_train, y_train, x_val=[], y_val=[], test=[], params={}, categorical_feature=[], num_iterations=3000, learning_rate=0.1, early_stopping_rounds=150, model_type='lgb'):

    ' LightGBM / XGBoost / ExtraTrees / LogisticRegression'
    if model_type=='lgb':
        lgb_train = lgb.Dataset(data=x_train, label=y_train)
        if len(x_val)>0:
            lgb_eval = lgb.Dataset(data=x_val, label=y_val)
            reg = lgb.train(params=params,
                            train_set=lgb_train,
                            valid_sets=lgb_eval,
                            num_boost_round=num_iterations,
                            early_stopping_rounds=early_stopping_rounds,
                            verbose_eval=200,
                            categorical_feature = categorical_feature
                            )
            y_pred = reg.predict(x_val)
        else:
            reg = lgb.train(params,
                            train_set=lgb_train,
                            categorical_feature = categorical_feature
                            )
            y_pred = reg.predict(test)

    elif model_type=='xgb':

        d_train = xgb.DMatrix(x_train, label=y_train)
        if len(x_val)>0:
            d_valid = xgb.DMatrix(x_val, label=y_val)
            watch_list = [(d_train, 'train'), (d_valid, 'eval')]

            reg = xgb.train(params,
                            dtrain=d_train,
                            evals=watch_list,
                            num_boost_round=num_iterations,
                            early_stopping_rounds=early_stopping_rounds,
                            verbose_eval=200
                            )
            y_pred = reg.predict(d_valid)
        else:
            d_test = xgb.DMatrix(test)
            reg = xgb.train(params,
                            dtrain=d_train,
                            num_boost_round=num_iterations
                            )
            y_pred = reg.predict(d_test)

    elif model_type=='ext':

        reg = ExtraTreesClassifier(
            **params,
            n_estimators = num_iterations
        )

        reg.fit(x_train, y_train)
        y_pred = reg.predict_proba(x_val)[:,1]

    elif model_type=='lgr':

        reg = LogisticRegression(
            **params,
            n_estimators = num_iterations
        )

        reg.fit(x_train, y_train)
        y_pred = reg.predict_proba(x_val)[:,1]

    else:
        logger.info(f'{model_type} is not supported.')

    return reg, y_pred

