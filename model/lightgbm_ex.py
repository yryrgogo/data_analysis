"""
LightGBM
"""
from .base_model import Model
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import ParameterGrid, StratifiedKFold, GroupKFold
import sys


class LightGBM_EX(Model):

    def __init__(self, model_type='lgb', logger=False, ignore_list=[]):
        self.__model = lgb
        self.model_type==model_type
        self.logger = logger
        self.ignore_list = []
        self.fold_model_list = []
        self.cv_feim = []
        self.prediction = np.array()
        self.cv_score = None
        self.result_stack = []

    def train(self, x_train, y_train, x_val, y_val, params, verbose_eval=100, early_stopping_rounds=100, num_boost_round=2000, categorical_list=[]):
        lgb_train = self.__model.Dataset(data=x_train, label=y_train)
        lgb_eval = self.__model.Dataset(data=x_val, label=y_val)
        return self.__model.train(
            train_set=lgb_train,
            valid_sets=lgb_eval,
            params=params,
            verbose_eval=verbose_eval,
            early_stopping_rounds=early_stopping_rounds,
            num_boost_round=num_boost_round,
            categorical_feature= [col for col in list(x_train.columns) if (x_train[col].dtype == 'object') and col not in self.ignore_list]
        )

    def predict(self, X, Y):
        y_pred = self.__model.predict(X)
        return y_pred

    def feature_importance(self):
        return self.__model.feature_importance()

    def cross_prediction(self, train, test, key, target, metric, fold_type='stratified', fold=5, group_col_name='', params={}, num_boost_round=2000, early_stopping_rounds=100, seed=1208, oof_flg=True, ignore_list=[]):

        if params['objective'] == 'regression':
            y = train[target].astype('float64')
            y = np.log1p(y)
        else:
            y = train[target]

        list_score = []
        cv_feim = pd.DataFrame([])
        prediction = np.array([])

        ' KFold '
        if fold_type == 'stratified':
            folds = StratifiedKFold(n_splits=fold, shuffle=True, random_state=seed)  # 1
            kfold = folds.split(train, y)
        elif fold_type == 'group':
            if group_col_name == '':
                raise ValueError(f'Not exist group_col_name.')
            folds = GroupKFold(n_splits=fold)
            kfold = folds.split(train, y, groups=train[group_col_name].values)

        use_cols = [f for f in train.columns if f not in self.ignore_list]
        self.use_cols = sorted(use_cols)  # カラム名をソートし、カラム順による学習への影響をなくす

        if kaggle == 'ga':
            if 'unique_id' in list(train.columns):
                train.set_index(['unique_id', key], inplace=True)
                test.set_index(['unique_id', key], inplace=True)
            else:
                train.set_index(key, inplace=True)
                test.set_index(key, inplace=True)

        for n_fold, (trn_idx, val_idx) in enumerate(kfold):

            x_train, y_train = train[use_cols].iloc[trn_idx, :], y.iloc[trn_idx]
            x_val, y_val = train[use_cols].iloc[val_idx, :], y.iloc[val_idx]

            if self.model_type.count('xgb'):
                " XGBは'[]'と','と'<>'がNGなのでreplace "
                if i == 0:
                    test = test[use_cols]
                use_cols = []
                for col in x_train.columns:
                    use_cols.append(col.replace(
                        "[", "-q-").replace("]", "-p-").replace(",", "-o-"))
                x_train.columns = use_cols
                x_val.columns = use_cols
                test.columns = use_cols

            if n_fold == 0:
                test = test[use_cols]

            estimator = self.train(
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val,
                params=params,
                categorical_list=categorical_list,
                num_boost_round=num_boost_round,
                early_stopping_rounds=early_stopping_rounds
            )
            y_pred = estimator.predict(x_val)

            self.fold_model_list.append(estimator)

            if kaggle == 'ga':
                hits = x_val['totals-hits'].map(lambda x: 0 if x ==
                                                1 else 1).values
                bounces = x_val['totals-bounces'].map(
                    lambda x: 0 if x == 1 else 1).values
                y_pred = y_pred * hits * bounces
                if metric == 'rmse':
                    y_pred[y_pred < 0.1] = 0

            sc_score = self.sc_metrics(y_val, y_pred, metric)

            list_score.append(sc_score)
            self.logger.info(f'Fold No: {n_fold} | {metric}: {sc_score}')

            ' OOF for Stackng '
            if oof_flg:
                val_pred = y_pred
                if n_fold == 0:
                    if kaggle == 'ga':
                        val_stack = x_val.reset_index()[['unique_id', key]]
                    else:
                        val_stack = x_val.reset_index()[key].to_frame()
                    val_stack[target] = val_pred
                else:
                    if kaggle == 'ga':
                        tmp = x_val.reset_index()[['unique_id', key]]
                    else:
                        tmp = x_val.reset_index()[key].to_frame()
                    tmp[target] = val_pred
                    val_stack = pd.concat([val_stack, tmp], axis=0)

            if not(self.model_type.count('xgb')):
                test_pred = estimator.predict(test)
            elif self.model_type.count('xgb'):
                test_pred = estimator.predict(xgb.DMatrix(test))

            #  if params['objective']=='regression':
            #      test_pred = np.expm1(test_pred)

            if len(prediction) == 0:
                prediction = test_pred
            else:
                prediction += test_pred

            ' Feature Importance '
            feim_name = f'{n_fold}_importance'
            feim = df_feature_importance(
                model=estimator, use_cols=use_cols, feim_name=feim_name)

            if len(cv_feim) == 0:
                cv_feim = feim.copy()
            else:
                cv_feim = cv_feim.merge(feim, on='feature', how='inner')

        cv_score = np.mean(list_score)
        self.logger.info(f'''
    #========================================================================
    # Train End.''')
        [self.logger.info(f'''
    # Validation No: {i} | {metric}: {score}''') for i, score in enumerate(list_score)]
        self.logger.info(f'''
    # Params   : {params}
    # CV score : {cv_score}
    #======================================================================== ''')

        ' fold数で平均をとる '
        prediction = prediction / fold
        if params['objective'] == 'regression':
            prediction = np.expm1(prediction)

        ' OOF for Stackng '
        if oof_flg:
            if kaggle == 'ga':
                pred_stack = test.reset_index()[['unique_id', key]]
            else:
                pred_stack = test.reset_index()[key].to_frame()
            pred_stack[target] = prediction
            result_stack = pd.concat([val_stack, pred_stack], axis=0)
            logger.info(
                f'result_stack shape: {result_stack.shape} | cnt_id: {len(result_stack[key].drop_duplicates())}')
        else:
            result_stack = []

        importance = []
        for fold_no in range(fold):
            if len(importance) == 0:
                importance = cv_feim[f'{fold_no}_importance'].values.copy()
            else:
                importance += cv_feim[f'{fold_no}_importance'].values

        cv_feim['avg_importance'] = importance / fold
        cv_feim.sort_values(by=f'avg_importance',
                            ascending=False, inplace=True)
        cv_feim['rank'] = np.arange(len(cv_feim))+1

        if kaggle == 'ga':
            if 'unique_id' in train.reset_index().columns:
                cv_feim.to_csv(
                    f'../valid/{self.model_type}_feat{len(cv_feim)}_{metric}{str(cv_score)[:8]}.csv', index=False)
            else:
                cv_feim.to_csv(
                    f'../valid/two_{self.model_type}_feat{len(cv_feim)}_{metric}{str(cv_score)[:8]}.csv', index=False)
        else:
            cv_feim.to_csv(
                f'../valid/{self.model_type}_feat{len(cv_feim)}_{metric}{str(cv_score)[:8]}.csv', index=False)

        self.cv_feim = cv_feim
        self.prediction = prediction
        self.cv_score = cv_score
        self.result_stack = result_stack

        return self

    def data_check(self, df, test=False, cat_encode=False, dummie=0, exclude_category=False):
        '''
        Explain:
            学習を行う前にデータに問題がないかチェックする
            カテゴリカルなデータが入っていたらエンコーディング or Dropする
        Args:
        Return:
        '''
        categorical_list = get_categorical_features(df, ignore_list=ignore_list)
        dt_list = get_datetime_features(df, ignore_list=ignore_list)
        self.logger.info(f'''
#==============================================================================
# DATA CHECK START
# CATEGORICAL FEATURE: {categorical_list}
# DATETIME FEATURE   : {dt_list}
# CAT ENCODE         : {cat_encode}
# DUMMIE             : {dummie}
# ignore_list        : {self.ignore_list}
#==============================================================================
        ''')

        if cat_encode:
            ' 対象カラムのユニーク数が100より大きかったら、ラベルエンコーディングにする '
            label_list = []
            for cat in categorical_list:
                if df[cat].nunique()>100:
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

        ' Testsetで値のユニーク数が1のカラムを除外する '
        drop_list = []
        if test:
            for col in df.columns:
                length = df[col].nunique()
                if length <=1 and col not in ignore_list:
                    self.logger.info(f'''
***********WARNING************* LENGTH {length} COLUMN: {col}''')
                    move_feature(feature_name=col)
                    if col not in self.ignore_list:
                        drop_list.append(col)

        self.logger.info(f'''
#==============================================================================
# DATA CHECK END
# SHAPE: {df.shape}
#==============================================================================''')

        return df, drop_list
