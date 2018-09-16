import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import ParameterGrid, StratifiedKFold, GroupKFold
from sklearn.metrics import log_loss, roc_auc_score
import datetime
from tqdm import tqdm
import sys
from select_feature import move_feature
sys.path.append('../library')
from utils import get_categorical_features
from preprocessing import factorize_categoricals
from preprocessing import set_validation, split_dataset
import xgboost as xgb
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, ridge
import pickle
from sklearn.ensemble.partial_dependence import partial_dependence


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
        return [re_value], True
    elif iter_no==1 and score<1.6155:
        return [re_value], True
    elif iter_no==2 and score<2.3:
        return [re_value], True
    elif iter_no==3 and score<3.1:
        return [re_value], True
    else:
        return [re_value], False


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
    categorical_feature = get_categorical_features(df, ignore=ignore_list)
    logger.info(f'''
CATEGORICAL FEATURE: {categorical_feature}
LENGTH: {len(categorical_feature)}
DUMMIE: {dummie}
                ''')

    if exclude_category:
        for cat in categorical_feature:
            df.drop(cat, axis=1, inplace=True)
            move_feature(feature_name=cat)
        categorical_feature = []
    elif dummie==0:
        df = factorize_categoricals(df, categorical_feature)
        categorical_feature=[]
    elif dummie==1:
        df = get_dummies(df, categorical_feature)
        categorical_feature=[]

    logger.info(f'df SHAPE: {df.shape}')

    if test:
        drop_list = []
        for col in df.columns:
            length = len(df[col].drop_duplicates())
            if length <=1:
                logger.info(f'''
    ***********WARNING************* LENGTH {length} COLUMN: {col}''')
                move_feature(feature_name=col)
                if col!=target:
                    drop_list.append(col)
        df.drop(drop_list, axis=1, inplace=True)

    ' カラム名をソートし、カラム順による学習への影響をなくす '
    df.sort_index(axis=1, inplace=True)

    logger.info(f'''
#==============================================================================
# DATA CHECK END
#==============================================================================''')

    return df


def cross_validation(logger, train, target, fold_type='stratified', fold=5, seed=1208, params={}, metric='auc', categorical_feature=[], truncate_flg=0, num_iterations=3500, learning_rate=0.1, early_stopping_rounds=150, model_type='lgb', ignore_list=[]):

    train = data_check(logger, train, target)

    list_score = []
    y = train[target]
    cv_feim = pd.DataFrame([])

    ' カラム名をソートし、カラム順による影響をなくす '
    train.sort_index(axis=1, inplace=True)

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

        if n_fold==0:
            logger.info(f'\nTrainset Col Number: {len(use_cols)}')

        clf, y_pred = Estimator(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val, params=params, categorical_feature=categorical_feature, num_iterations=num_iterations, early_stopping_rounds=early_stopping_rounds, learning_rate=learning_rate, model_type=model_type)

        ' 自作x-rayのテスト '
        #  result = x_ray(model=clf, valid=x_val)
        #  return result
        #  import pickle
        #  sys.exit()
        with open('../output/clf.pickle', 'wb') as f:
            pickle.dump(obj=clf, file=f)
        sys.exit()

        sc_score = sc_metrics(y_val, y_pred, metric)
        if n_fold==0:
            first_score = sc_score

        list_score.append(sc_score)
        logger.info(f'validation: {n_fold} | {metric}: {sc_score}')

        if truncate_flg==1:
            re_value, judge = judgement(score=sum(list_score), iter_no=n_fold, re_value=first_score)
            if judge: return re_value, 0

        feim_name = f'{n_fold}_importance'
        feim = df_feature_importance(model=clf, model_type=model_type, use_cols=use_cols, feim_name=feim_name)

    if len(cv_feim)==0:
        cv_feim = feim.copy()
    else:
        cv_feim = cv_feim.merge(feim, on='feature', how='inner')

    cv_score = np.mean(list_score)
    logger.info(f'train shape: {x_train.shape}')
    logger.info(f'\nCross Validation End\nCV score : {cv_score}')

    cv_feim['cv_score'] = cv_score

    importance = []
    for n_fold in range(n_fold+1):
        if len(importance)==0:
            importance = cv_feim[f'{n_fold}_importance'].values.copy()
        else:
            importance += cv_feim[f'{n_fold}_importance'].values

    cv_feim['avg_importance'] = importance / n_fold+1
    cv_feim.sort_values(by=f'avg_importance', ascending=False, inplace=True)
    cv_feim['rank'] = np.arange(len(cv_feim))+1

    return cv_feim, len(use_cols)


def prediction(logger, train, test, target, categorical_feature=[], metric='auc', params={}, num_iterations=20000, learning_rate=0.02, model_type='lgb', oof_flg=False):

    x_train, y_train = train, train[target]
    use_cols = x_train.columns.values
    use_cols = [col for col in use_cols if col not in ignore_list and not(col.count('valid_no'))]
    test = test[use_cols]

    ' 学習 '
    clf, y_pred = Estimator(x_train=x_train[use_cols], y_train=y_train, params=params, test=test, categorical_feature=categorical_feature, num_iterations=num_iterations, learning_rate=learning_rate, model_type=model_type)

    return y_pred, len(use_cols)


def cross_prediction(logger, train, test, key, target, fold_type='stratified', fold=5, seed=605, categorical_feature=[], metric='auc', params={}, num_iterations=20000, learning_rate=0.02, early_stopping_rounds=150, model_type='lgb', oof_flg=True, ignore_list=[]):

    train = data_check(logger, df=train, target=target)
    test = data_check(logger, df=test, target=target, test=True)
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

        clf, y_pred = Estimator(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val, params=params, categorical_feature=categorical_feature, num_iterations=num_iterations, early_stopping_rounds=early_stopping_rounds, learning_rate=learning_rate, model_type=model_type)

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

    Model, y_pred = Estimator(
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


def Estimator(x_train, y_train, x_val=[], y_val=[], test=[], params={}, categorical_feature=[], num_iterations=3900, learning_rate=0.1, early_stopping_rounds=150, model_type='lgb'):

    ' LightGBM / XGBoost / ExtraTrees / LogisticRegression'
    if model_type=='lgb':
        lgb_train = lgb.Dataset(data=x_train, label=y_train)
        if len(x_val)>0:
            lgb_eval = lgb.Dataset(data=x_val, label=y_val)
            clf = lgb.train(params=params,
                            train_set=lgb_train,
                            valid_sets=lgb_eval,
                            num_boost_round=num_iterations,
                            early_stopping_rounds=early_stopping_rounds,
                            verbose_eval=200,
                            categorical_feature = categorical_feature
                            )
            y_pred = clf.predict(x_val)
        else:
            clf = lgb.train(params,
                            train_set=lgb_train,
                            categorical_feature = categorical_feature
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
