import gc
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import ParameterGrid, StratifiedKFold, GroupKFold
from sklearn.metrics import log_loss, roc_auc_score, mean_squared_error, r2_score
import datetime
from tqdm import tqdm
import sys
sys.path.append('../library')
from select_feature import move_feature
import utils
from utils import get_categorical_features, get_datetime_features
from preprocessing import factorize_categoricals
import xgboost as xgb
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, ridge
import pickle
from sklearn.ensemble.partial_dependence import partial_dependence


def cross_validation(logger, train, key, target, metric, fold_type='stratified', fold=5, group_col_name='', val_label='val_label', seed=1208, params={}, categorical_list=[], judge_flg=False, model_type='lgb', ignore_list=[], xray=False):


    if params['objective']=='regression':
        y = train[target].astype('float64')
        y = np.log1p(y)
    else:
        y = train[target]

    list_score = []
    cv_feim = pd.DataFrame([])

    ' KFold '
    if fold_type=='stratified':
        folds = StratifiedKFold(n_splits=fold, shuffle=True, random_state=seed) #1
        kfold = folds.split(train,y)
    elif fold_type=='group':
        if group_col_name=='':raise ValueError(f'Not exist group_col_name.')
        folds = GroupKFold(n_splits=fold)
        kfold = folds.split(train, y, groups=train[group_col_name].values)


    use_cols = [f for f in train.columns if f not in ignore_list]
    model_list = []

    for n_fold, (trn_idx, val_idx) in enumerate(kfold):

        x_train, y_train = train[use_cols].iloc[trn_idx, :], y.iloc[trn_idx]

        R_com = False
        if R_com:
            # fullIdを後で使うので、validationを切ってからset_indexしておく
            x_val, y_val = train.iloc[val_idx, :], y.iloc[val_idx]
            x_val = x_val.set_index(key)[use_cols]
        else:
            x_val, y_val = train[use_cols].iloc[val_idx, :], y.iloc[val_idx]

        clf, y_pred = Estimator(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            params=params,
            categorical_list=categorical_list,
            model_type=model_type
        )

        if R_com:
            x_val[target] = np.expm1(y_val)
            x_val.fillna(0, inplace=True)
            x_val['prediction'] = np.expm1(y_pred)
            x_val.reset_index(inplace=True)
            x_val = x_val.groupby(key)[target, 'prediction'].sum().reset_index()
            x_val['prediction'] = np.clip(x_val['prediction'].values, a_min=0, a_max=None)
            sc_score = sc_metrics(np.log1p(x_val[target].values), np.log1p(x_val['prediction'].values), metric)
        else:
            sc_score = sc_metrics(y_val, y_pred, metric)

        if n_fold==0:
            first_score = sc_score

        list_score.append(sc_score)
        logger.info(f'Valid No: {n_fold} | {metric}: {sc_score} | {x_train.shape}')

        #========================================================================
        # 学習結果に閾値を設けて、それを超えなかったら中止する
        if judge_flg:
            logger.info(f'''
#========================================================================
# Train Result Judgement Start.
#========================================================================''')
            return_score, judge = judgement(score=sum(list_score), iter_no=n_fold, return_score=first_score)
            if judge: return return_score, 0
        #========================================================================

        #========================================================================
        # Get Feature Importance
        feim_name = f'{n_fold}_importance'
        feim = df_feature_importance(model=clf, model_type=model_type, use_cols=use_cols, feim_name=feim_name)
        if len(cv_feim)==0:
            cv_feim = feim.copy()
        else:
            cv_feim = cv_feim.merge(feim, on='feature', how='inner')
        #========================================================================

        # For X-RAY
        if xray: model_list.append(clf)

    cv_score = np.mean(list_score)
    logger.info(f'''
#========================================================================
# Train End.''')
    [logger.info(f'''
# Validation No: {i} | {metric}: {score}''') for i, score in enumerate(list_score)]
    logger.info(f'''
# Cross Validation End. CV score : {cv_score}
#======================================================================== ''')

    cv_feim['cv_score'] = cv_score

    #========================================================================
    # Make Feature Importance Summary
    importance = []
    for n_fold in range(n_fold):
        if len(importance)==0:
            importance = cv_feim[f'{n_fold}_importance'].values.copy()
        else:
            importance += cv_feim[f'{n_fold}_importance'].values

    cv_feim['avg_importance'] = importance / fold
    cv_feim.sort_values(by=f'avg_importance', ascending=False, inplace=True)
    cv_feim['rank'] = np.arange(len(cv_feim))+1
    #========================================================================

    if xray:
        return cv_feim, len(use_cols), model_list
    else:
        return cv_feim, len(use_cols)


def prediction(logger, train, test, target, categorical_list=[], metric='auc', params={}, num_iterations=20000, learning_rate=0.02, model_type='lgb', oof_flg=False):

    x_train, y_train = train, train[target]
    use_cols = x_train.columns.values
    use_cols = [col for col in use_cols if col not in ignore_list and not(col.count('valid_no'))]
    test = test[use_cols]

    ' 学習 '
    clf, y_pred = Estimator(x_train=x_train[use_cols], y_train=y_train, params=params, test=test, categorical_list=categorical_list, num_iterations=num_iterations, learning_rate=learning_rate, model_type=model_type)

    return y_pred, len(use_cols)


def cross_prediction(logger, train, test, key, target, metric, fold_type='stratified', fold=5, group_col_name='', val_label='val_label', seed=1208, categorical_list=[], params={}, model_type='lgb', oof_flg=True, ignore_list=[]):

    ' Data Check '
    test, drop_list = data_check(logger, df=test, target=target, test=True, ignore_list=ignore_list)
    test.drop(drop_list, axis=1, inplace=True)
    train.drop(drop_list, axis=1, inplace=True)
    train, _ = data_check(logger, df=train, target=target, ignore_list=ignore_list)
    y = train[target]

    list_score = []
    list_pred = []

    prediction = np.array([])
    cv_feim = pd.DataFrame([])

    ' KFold '
    if fold_type=='stratified':
        folds = StratifiedKFold(n_splits=fold, shuffle=True, random_state=seed) #1
        kfold = folds.split(train,y)
    elif fold_type=='group':
        folds = GroupKFold(n_splits=fold)
        kfold = folds.split(train,y, groups=train.index.values)

    use_cols = [f for f in train.columns if f not in ignore_list]

    for n_fold, (trn_idx, val_idx) in enumerate(kfold):

        x_train, y_train = train[use_cols].iloc[trn_idx, :], y.iloc[trn_idx]
        x_val, y_val = train[use_cols].iloc[val_idx, :], y.iloc[val_idx]

        use_cols = list(x_train.columns)
        if n_fold==0:
            logger.info(f'\nTrainset Col Number: {len(use_cols)}')

        if model_type=='xgb':
            " XGBは'[]'と','と'<>'がNGなのでreplace "
            if i==0:
                test = test[use_cols]
            use_cols = []
            for col in x_train.columns:
                use_cols.append(col.replace("[", "-q-").replace("]", "-p-").replace(",", "-o-"))
            x_train.columns = use_cols
            x_val.columns = use_cols
            test.columns = use_cols

        clf, y_pred = Estimator(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val, params=params, categorical_list=categorical_list, num_iterations=num_iterations, early_stopping_rounds=early_stopping_rounds, learning_rate=learning_rate, model_type=model_type)

        sc_score = sc_metrics(y_val, y_pred)

        list_score.append(sc_score)
        logger.info(f'Fold No: {n_fold} | {metric}: {sc_score}')

        ' OOF for Stackng '
        if oof_flg:
            val_pred = y_pred
            if n_fold==0:
                val_stack = x_val.reset_index()[key].to_frame()
                val_stack[target] = val_pred
            else:
                tmp = x_val.reset_index()[key].to_frame()
                tmp[target] = val_pred
                val_stack = pd.concat([val_stack, tmp], axis=0)
            logger.info(f'Fold No: {n_fold} | valid_stack shape: {val_stack.shape} | cnt_id: {len(val_stack[key].drop_duplicates())}')

        if model_type != 'xgb':
            test_pred = clf.predict(test[use_cols])
        elif model_type == 'xgb':
            test_pred = clf.predict(xgb.DMatrix(test))

        if len(prediction)==0:
            prediction = test_pred / fold
        else:
            prediction += test_pred / fold

        ' Feature Importance '
        feim_name = f'{n_fold}_importance'
        feim = df_feature_importance(model=clf, model_type=model_type, use_cols=use_cols, feim_name=feim_name)

        if len(cv_feim)==0:
            cv_feim = feim.copy()
        else:
            cv_feim = cv_feim.merge(feim, on='feature', how='inner')

    cv_score = np.mean(list_score)
    logger.info(f'use_features: {use_cols}')
    logger.info(f'params: {params}')
    logger.info(f'\nCross Validation End\nCV score : {cv_score}')

    ' fold数で平均をとる '
    prediction = prediction / fold

    ' OOF for Stackng '
    if oof_flg:
        pred_stack = test.reset_index()[key].to_frame()
        pred_stack[target] = prediction
        result_stack = pd.concat([val_stack, pred_stack], axis=0)
        logger.info(f'result_stack shape: {result_stack.shape} | cnt_id: {len(result_stack[key].drop_duplicates())}')
    else:
        result_stack=[]

    importance = []
    for fold_no in range(fold):
        if len(importance)==0:
            importance = cv_feim[f'{fold_no}_importance'].values.copy()
        else:
            importance += cv_feim[f'{fold_no}_importance'].values

    cv_feim['avg_importance'] = importance / fold
    cv_feim.sort_values(by=f'avg_importance', ascending=False, inplace=True)
    cv_feim['rank'] = np.arange(len(cv_feim))+1

    cv_feim.to_csv(f'../output/cv_feature{len(cv_feim)}_importances_{metric}_{cv_score}.csv', index=False)

    return prediction, cv_score, result_stack


' Regression '
def TimeSeriesPrediction(logger, train, test, key, target, val_label='val_label', categorical_list=[], metric='rmse', params={}, model_type='lgb', ignore_list=[]):
    '''
    Explain:
    Args:
    Return:
    '''

    ' Data Check '
    test, drop_list = data_check(logger, df=test, target=target, test=True, ignore_list=ignore_list)
    test.drop(drop_list, axis=1, inplace=True)
    train.drop(drop_list, axis=1, inplace=True)
    train, _ = data_check(logger, df=train, target=target, ignore_list=ignore_list)

    ' Make Train Set & Validation Set  '
    x_train = train.query("val_label==0")
    x_val = train.query("val_label==1")
    y_train = x_train[target].values.astype('float64')
    y_val = x_val[target].values.astype('float64')
    logger.info(f'''
#========================================================================
# X_Train Set : {x_train.shape}
# X_Valid Set : {x_val.shape}
#========================================================================''')

    #========================================================================
    # 対数変換
    #========================================================================
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

    Model, y_pred = Estimator(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        params=params,
        categorical_list=categorical_list,
        num_iterations=num_iterations,
        early_stopping_rounds=early_stopping_rounds,
        learning_rate=learning_rate,
        model_type=model_type
    )

    ' 対数変換を元に戻す '
    sc_score = sc_metrics(y_val, y_pred)
    y_train = np.expm1(y_val)
    y_pred = np.expm1(y_pred)

    if model_type != 'xgb':
        test_pred = Model.predict(test[use_cols])
    elif model_type == 'xgb':
        test_pred = Model.predict(xgb.DMatrix(test))

    ' Feature Importance '
    feim_name = f'importance'
    feim = df_feature_importance(model=Model, model_type=model_type, use_cols=use_cols, feim_name=feim_name)
    feim.sort_values(by=f'importance', ascending=False, inplace=True)
    feim['rank'] = np.arange(len(feim))+1

    feim.to_csv(f'../output/feature{len(feim)}_importances_{metric}_{sc_score}.csv', index=False)

    return y_pred, sc_score


def Estimator(x_train, y_train, x_val=[], y_val=[], test=[], params={}, categorical_list=[], num_iterations=3000, learning_rate=0.1, early_stopping_rounds=150, model_type='lgb'):

    ' LightGBM / XGBoost / ExtraTrees / LogisticRegression'
    if model_type=='lgb':
        lgb_train = lgb.Dataset(data=x_train, label=y_train)
        if len(x_val)>0:
            lgb_eval = lgb.Dataset(data=x_val, label=y_val)
            clf = lgb.train(params=params,
                            train_set=lgb_train,
                            valid_sets=lgb_eval,
                            verbose_eval=200,
                            categorical_feature = categorical_list
                            )
            y_pred = clf.predict(x_val)
        else:
            clf = lgb.train(params,
                            train_set=lgb_train,
                            categorical_feature = categorical_list
                            )
            y_pred = clf.predict(test)

    elif model_type=='xgb':

        d_train = xgb.DMatrix(x_train, label=y_train)
        if len(x_val)>0:
            d_valid = xgb.DMatrix(x_val, label=y_val)
            watch_list = [(d_train, 'train'), (d_valid, 'eval')]

            clf = xgb.train(params,
                            dtrain=d_train,
                            evals=watch_list,
                            num_boost_round=num_iterations,
                            early_stopping_rounds=early_stopping_rounds,
                            verbose_eval=200
                            )
            y_pred = clf.predict(d_valid)
        else:
            d_test = xgb.DMatrix(test)
            clf = xgb.train(params,
                            dtrain=d_train,
                            num_boost_round=num_iterations
                            )
            y_pred = clf.predict(d_test)

    elif model_type=='ext':

        clf = ExtraTreesClassifier(
            **params,
            n_estimators = num_iterations
        )

        clf.fit(x_train, y_train)
        y_pred = clf.predict_proba(x_val)[:,1]

    elif model_type=='lgr':

        clf = LogisticRegression(
            **params,
            n_estimators = num_iterations
        )

        clf.fit(x_train, y_train)
        y_pred = clf.predict_proba(x_val)[:,1]

    else:
        logger.info(f'{model_type} is not supported.')

    return clf, y_pred


def sc_metrics(test, pred, metric='rmse'):
    if metric == 'logloss':
        return log_loss(test, pred)
    elif metric == 'auc':
        return roc_auc_score(test, pred)
    elif metric=='l2':
        return r2_score(test, pred)
    elif metric=='rmse':
        return np.sqrt(mean_squared_error(test, pred))
    else:
        print('score caliculation error!')

def judgement(score, iter_no, return_score):
    ' 一定基準を満たさなかったら打ち止め '
    if iter_no==0 and score<0.813:
        return [return_score], True
    elif iter_no==1 and score<1.6155:
        return [return_score], True
    elif iter_no==2 and score<2.3:
        return [return_score], True
    elif iter_no==3 and score<3.1:
        return [return_score], True
    else:
        return [return_score], False

def df_feature_importance(model, model_type, use_cols, feim_name='importance'):
    ' Feature Importance '
    if model_type=='lgb':
        tmp_feim = pd.Series(model.feature_importance(), name=feim_name)
        feature_name = pd.Series(use_cols, name='feature')
        feim = pd.concat([feature_name, tmp_feim], axis=1)
    elif model_type=='xgb':
        tmp_feim = model.get_fscore()
        feim = pd.Series(tmp_feim,  name=feim_name).to_frame().reset_index().rename(columns={'index':'feature'})
    elif model_type=='ext':
        tmp_feim = model.feature_importance_()
        feim = pd.Series(tmp_feim,  name=feim_name).to_frame().reset_index().rename(columns={'index':'feature'})

    return feim


def data_check(logger, df, target, test=False, dummie=0, exclude_category=False, ignore_list=[]):
    logger.info(f'''
#==============================================================================
# DATA CHECK START
#==============================================================================''')
    categorical_list = get_categorical_features(df, ignore_list=ignore_list)
    dt_list = get_datetime_features(df, ignore_list=ignore_list)
    logger.info(f'''
#==============================================================================
# CATEGORICAL FEATURE: {categorical_list}
# LENGTH: {len(categorical_list)}
# DUMMIE: {dummie}
#==============================================================================
    ''')

    #========================================================================
    # 連続値として扱うべきカラムがobjectになっていることがあるので
    #========================================================================
    #  for cat in categorical_list:
    #      try:
    #          df[cat] = df[cat].astype('int64')
    #          categorical_list.remove(cat)
    #      except ValueError:
    #          pass
    #========================================================================
    # datetime系のカラムはdrop
    #========================================================================
    for dt in dt_list:
        df.drop(dt, axis=1, inplace=True)

    ' 対象カラムのユニーク数が100より大きかったら、ラベルエンコーディングにする '
    label_list = []
    for cat in categorical_list:
        if len(df[cat].drop_duplicates())>100:
            label_list.append(cat)
            categorical_list.remove(cat)
        df = factorize_categoricals(df, label_list)

    if exclude_category:
        for cat in categorical_list:
            df.drop(cat, axis=1, inplace=True)
            move_feature(feature_name=cat)
        categorical_list = []
    elif dummie==1:
        df = get_dummies(df, categorical_list)
        categorical_list=[]
    elif dummie==0:
        df = factorize_categoricals(df, categorical_list)
        categorical_list=[]

    logger.info(f'df SHAPE: {df.shape}')

    drop_list = []
    if test:
        for col in df.columns:
            length = len(df[col].drop_duplicates())
            if length <=1 and col not in ignore_list:
                logger.info(f'''
    ***********WARNING************* LENGTH {length} COLUMN: {col}''')
                move_feature(feature_name=col)
                if col!=target:
                    drop_list.append(col)

    ' カラム名をソートし、カラム順による学習への影響をなくす '
    df.sort_index(axis=1, inplace=True)

    logger.info(f'''
#==============================================================================
# DATA CHECK END
#==============================================================================''')

    return df, drop_list
