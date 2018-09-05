import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import ParameterGrid, StratifiedKFold, GroupKFold
from sklearn.metrics import log_loss, roc_auc_score
import datetime
from tqdm import tqdm
import sys
from x_ray import x_ray
sys.path.append('../library')
from utils import x_y_split
from preprocessing import set_validation, split_dataset
from params_lgbm import train_params, train_params_0729, train_params_0815, xgb_params_0814, extra_params, lgr_params
import xgboost as xgb
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, ridge
from sklearn.preprocessing import StandardScaler


start_time = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())
sc = StandardScaler()

' データセットからそのまま使用する特徴量 '
unique_id = 'SK_ID_CURR'
target = 'TARGET'
ignore_features = [unique_id, target, 'valid_no', 'is_train', 'is_test', 'valid_no_2', 'valid_no_3', 'valid_no_4']

train_params = {
    #  'boosting':'dart',
    'num_threads': 35,
    'learning_rate':0.02,
    #  'colsample_bytree':0.01,
    'colsample_bytree':0.02,
    #  'subsample':0.9,
    'min_split_gain':0.01,
    'objective':'binary',
    'boosting_type':'gbdt',
    'metric':'auc',
    'max_depth':6,
    'min_child_weight':18,
    #  'min_child_weight':36,
    #  'max_bin':250,
    #  'min_child_samples':96,
    #  'min_data_in_bin':96,

    'lambda_l1':0.1,
    'lambda_l2':90,
    'num_leaves':20,
    #  'num_leaves':11,
    'random_seed': 1208,
    'bagging_seed':1208,
    'feature_fraction_seed':1208,
    'data_random_seed':1208
    }


def sc_metrics(test, pred, metric='auc'):
    if metric == 'logloss':
        return log_loss(test, pred)
    elif metric == 'auc':
        return roc_auc_score(test, pred)
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


def df_feature_importance(model, model_type, feim_name='importance'):
    ' Feature Importance '
    if model_type=='lgb':
        tmp_feim = pd.Series(clf.feature_importance(), name=feim_name)
        feature_name = pd.Series(use_cols, name='feature')
        feim = pd.concat([feature_name, tmp_feim], axis=1)
    elif model_type=='xgb':
        tmp_feim = clf.get_fscore()
        feim = pd.Series(tmp_feim,  name=feim_name).to_frame().reset_index().rename(columns={'index':'feature'})
    elif model_type=='ext':
        tmp_feim = clf.feature_importance_()
        feim = pd.Series(tmp_feim,  name=feim_name).to_frame().reset_index().rename(columns={'index':'feature'})

    return feim


def cross_validation(logger, train, target, fold_type='stratified', fold=5, seed=1208, params=train_params, metric='auc', categorical_feature=[], truncate_flg=0, num_iterations=3500, learning_rate=0.1, early_stopping_rounds=150, model_type='lgb'):

    list_score = []
    y = train[target]
    cv_feim = pd.trainFrame([])

    ' カラム名をソートし、カラム順による影響をなくす '
    train.sort_index(axis=1, inplace=True)

    ' KFold '
    if fold_type=='stratified':
        folds = StratifiedKFold(n_splits=fold, shuffle=True, random_state=seed) #1
        kfold = folds.split(train,y)
    elif fold_type=='group':
        folds = GroupKFold(n_splits=fold)
        kfold = folds.split(train,y, groups=train.index.values)

    use_cols = [f for f in train.columns if f not in ignore_features]

    for n_fold, (trn_idx, val_idx) in enumerate(kfold):

        x_train, y_train = train[use_cols].iloc[trn_idx, :], y.iloc[trn_idx]
        x_val, y_val = train[use_cols].iloc[val_idx, :], y.iloc[val_idx]

        if n_fold==0:
            logger.info(f'\nTrainset Col Number: {len(use_cols)}')

        clf, y_pred = classifier(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val, params=params, categorical_feature=categorical_feature, num_iterations=num_iterations, early_stopping_rounds=early_stopping_rounds, learning_rate=learning_rate, model_type=model_type)

        ' 自作x-rayのテスト '
        #  result = x_ray(model=clf, valid=x_val)
        #  return result
        #  import pickle
        #  with open('clf.pickle', 'wb') as f:
        #      pickle.dump(obj=clf, file=f)
        #  sys.exit()

        sc_score = sc_metrics(y_val, y_pred, metric)
        if n_fold==0:
            first_score = sc_score

        list_score.append(sc_score)
        logger.info(f'validation: {n_fold} | {metric}: {sc_score}')

        if truncate_flg==1:
            re_value, judge = judgement(score=sum(list_score), iter_no=n_fold, re_value=first_score)
            if judge: return re_value, 0

        feim_name = f'{n_fold}_importance'
        feim = df_feature_importance(model=clf, model_type=model_type, feim_name=feim_name)

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
    use_cols = [col for col in use_cols if col not in ignore_features and not(col.count('valid_no'))]
    test = test[use_cols]

    ' 学習 '
    clf, y_pred = classifier(x_train=x_train[use_cols], y_train=y_train, params=params, test=test, categorical_feature=categorical_feature, num_iterations=num_iterations, learning_rate=learning_rate, model_type=model_type)

    return y_pred, len(use_cols)


def cross_prediction(logger, train, test, target, categorical_feature=[], val_col='valid_no', metric='auc', params={}, num_iterations=20000, learning_rate=0.02, early_stopping_rounds=150, model_type='lgb'):

    list_score = []
    list_pred = []

    valid_list = train[val_col].drop_duplicates().values
    prediction = np.array([])
    cv_feim = pd.DataFrame([])

    ' カラム名をソートし、カラム順による精度への影響をなくす '
    train.sort_index(axis=1, inplace=True)
    y = train[target]
    test.sort_index(axis=1, inplace=True)
    use_cols = [f for f in train.columns if f not in ignore_features]

    folds =StratifiedKFold(n_splits=fold, shuffle=True, random_state=seed) #1
    for n_fold, (trn_idx, val_idx) in enumerate(folds.split(train,y)):

        x_train, y_train = train.iloc[use_cols].iloc[trn_idx], y.iloc[trn_idx]
        x_val, y_val = train[use_cols].iloc[val_idx], y.iloc[val_idx]

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

        clf, y_pred = classifier(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val, params=params, categorical_feature=categorical_feature, num_iterations=num_iterations, early_stopping_rounds=early_stopping_rounds, learning_rate=learning_rate, model_type=model_type)

        sc_score = sc_metrics(y_val, y_pred)

        list_score.append(sc_score)
        logger.info(f'validation: {val_no} | {metric}: {sc_score}')

        ' OOF for Stackng '
        if oof_flg:
            val_pred = y_pred
            if n_fold==0:
                val_stack = x_val.reset_index()[unique_id].to_frame()
                val_stack[target] = val_pred
            else:
                tmp = x_val.reset_index()[unique_id].to_frame()
                tmp[target] = val_pred
                val_stack = pd.concat([val_stack, tmp], axis=0)
            logger.info(f'valid_no: {val_no} | valid_stack shape: {val_stack.shape} | cnt_id: {len(val_stack[unique_id].drop_duplicates())}')

        if model_type != 'xgb':
            test_pred = clf.predict(test[use_cols])
        elif model_type == 'xgb':
            test_pred = clf.predict(xgb.DMatrix(test))

        if len(prediction)==0:
            prediction = test_pred / len(valid_list)
        else:
            prediction += test_pred / len(valid_list)

        ' Feature Importance '
        feim_name = f'{n_fold}_importance'
        feim = df_feature_importance(model=clf, model_type=model_type, feim_name=feim_name)

        if len(cv_feim)==0:
            cv_feim = feim.copy()
        else:
            cv_feim = cv_feim.merge(feim, on='feature', how='inner')

    cv_score = np.mean(list_score)
    logger.info(f'use_features: {use_cols}')
    logger.info(f'params: {params}')
    logger.info(f'\nCross Validation End\nCV score : {cv_score}')

    ' fold数で平均をとる '
    prediction = prediction / len(valid_list)

    ' OOF for Stackng '
    if oof_flg:
        pred_stack = test.reset_index()[unique_id].to_frame()
        pred_stack[target] = prediction
        result_stack = pd.concat([val_stack, pred_stack], axis=0)
        logger.info(f'result_stack shape: {result_stack.shape} | cnt_id: {len(result_stack[unique_id].drop_duplicates())}')
    else:
        result_stack=[]

    importance = []
    for val_no in valid_list:
        if len(importance)==0:
            importance = cv_feim[f'{val_no}_importance'].values.copy()
        else:
            importance += cv_feim[f'{val_no}_importance'].values

    cv_feim['avg_importance'] = importance / len(valid_list)
    cv_feim.sort_values(by=f'avg_importance', ascending=False, inplace=True)
    cv_feim['rank'] = np.arange(len(cv_feim))+1

    cv_feim.to_csv(f'../output/cv_feature{len(cv_feim)}_importances_{metric}_{cv_score}.csv', index=False)

    return prediction, cv_score, len(use_cols), result_stack


def classifier(x_train, y_train, x_val=[], y_val=[], test=[], params={}, categorical_feature=[], num_iterations=3500, learning_rate=0.1, early_stopping_rounds=150, model_type='lgb'):

    ' LightGBM / XGBoost / ExtraTrees / LogisticRegression'
    if model_type=='lgb':
    classifiers)==0:
            params = train_params
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

