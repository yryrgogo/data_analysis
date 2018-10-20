"""
機械学習モデル基底クラス
"""
from abc import ABCMeta, abstractmethod
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score, mean_squared_error, r2_score


class Model(metaclass=ABCMeta):
    """
    機械学習モデル基底クラス
    """
    @abstractmethod
    def train(self, train_features, train_target):
        """
        学習を実行します。

        Parameters
        ----------
        train_features : DataFrame
            学習用説明変数
        train_target : DataFrame
            学習用目的変数
        """
        pass

    @abstractmethod
    def predict(self, test_features):
        """
        予測を実行します。

        Parameters
        ----------
        test_features : DataFrame
            予測用説明変数
        """
        pass

    #  @abstractmethod
    def predict_proba(self, test_features):
        """
        予測を実行し、予測結果の確立を返却します。

        Parameters
        ----------
        test_features : DataFrame
            予測用説明変数
        """
        pass

    @abstractmethod
    def accuracy(self, test_features, test_target):
        """
        正解率を返却します。

        Parameters
        ----------
        test_features : DataFrame
            予測用説明変数
        test_target : DataFrame
            予測用目的変数
        """
        pass

    @abstractmethod
    def cross_validation(self, features, target):
        """
        交差検定を実行します。

        Parameters
        ----------
        features : DataFrame
            説明変数
        target : DataFrame
            目的変数
        """
        pass

    #  @abstractmethod
    def cross_val_score(self, features, target):
        """
        交差検定を実行します。

        Parameters
        ----------
        features : DataFrame
            説明変数
        target : DataFrame
            目的変数
        """
        pass

    def sc_metrics(y_test, y_pred, metric='logloss'):
        if metric == 'logloss':
            return log_loss(y_test, y_pred)
        elif metric == 'auc':
            return roc_auc_score(y_test, y_pred)
        elif metric=='l2':
            return r2_score(y_test, y_pred)
        elif metric=='rmse':
            return np.sqrt(mean_squared_error(y_test, y_pred))
        else:
            print('score caliculation error!')

    def auc(self, test_features, test_target):
        """
        AUCを返却します。

        Parameters
        ----------
        test_features : DataFrame
            説明変数
        test_target : DataFrame
            目的変数

        Returns
        -------
        auc : float
            AUC
        """
        return roc_auc_score(test_target, self.predict(test_features))

    def feature_impact(self, train_features, train_target):
        """
        特徴量インパクトを算出します。

        Parameters
        ----------
        train_features : DataFrame
            説明変数
        train_target : DataFrame
            目的変数

        Returns
        -------
        feature_impact : DataFrame
            特徴量のインパクト
        """
        result = []
        model_name = self.__class__.__name__.replace("Model", "")
        # 全特徴量結果
        all_auc = self.cross_val_score(train_features, train_target)
        print("ALL\t{}".format(all_auc))

        result.append({
            "model": model_name,
            "feature": "ALL",
            "auc": all_auc,
            "diff": None,
            "impact": None
        })

        auc_dict = {}
        diff_dict = {}
        for column in train_features.columns:
            dropped_features = train_features.drop(column, axis=1)
            dropped_auc = self.cross_val_score(dropped_features, train_target)
            auc_dict[column] = dropped_auc
            diff_dict[column] = all_auc - dropped_auc
            print("{}\t{}".format(column, dropped_auc))

        max_diff = np.max(list(diff_dict.values()))
        for column, value in diff_dict.items():
            result.append({
                "model": model_name,
                "feature": column,
                "auc": auc_dict[column],
                "diff": value,
                "impact": (value / max_diff) * 100
            })

        return pd.DataFrame(result)

    def xray(self, test_features, target_feature, max_size):
        """
        モデルX-Rayを算出します。

        Parameters
        ----------
        test_features : DataFrame
            説明変数
        target_feature : str
            算出対象の変数名
        max_size : int
            算出するデータ数上限

        Returns
        -------
        xray : DataFrame
            モデルX-Ray
        """
        feature_data = test_features.copy()
        result = []
        model_name = self.__class__.__name__.replace("Model", "")

        # 一意な値を代入した場合の処理
        # Nullに該当しない値に絞ってる?
        unique_data = feature_data[feature_data[target_feature +
                                                "_isnull"] == False].copy()
        unique_values = feature_data[target_feature].unique()

        if len(unique_values) > max_size:
            unique_values = np.random.choice(unique_values, max_size)

        for unique_value in unique_values:
            unique_data[target_feature] = unique_value
            average_proba = self.predict_proba(unique_data)[:, 1].mean()
            result.append({
                "model": model_name,
                "feature": target_feature,
                "value": unique_value,
                "proba": average_proba
            })

        # 欠損値の処理
        completion = test_features[target_feature].median()
        feature_data[target_feature] = completion
        feature_data[target_feature + "_isnull"] = True
        missing_value = self.predict_proba(feature_data)[:, 1].mean()

        result.append({
            "model": model_name,
            "feature": target_feature,
            "value": "_isnull",
            "proba": missing_value
        })

        return pd.DataFrame(result)


    def df_feature_importance(self):
        feim_name='importance'
        ' Feature Importance '
        if self.model_type.count('lgb'):
            tmp_feim = pd.Series(self.__model.feature_importance(), name=feim_name)
            feature_name = pd.Series(use_cols, name='feature')
            feim = pd.concat([feature_name, tmp_feim], axis=1)
        elif self.model_type.count('xgb'):
            tmp_feim = self.__model.get_fscore()
            feim = pd.Series(tmp_feim,  name=feim_name).to_frame().reset_index().rename(columns={'index':'feature'})
        elif self.model_type.count('ext'):
            tmp_feim = self.__model.feature_importance_()
            feim = pd.Series(tmp_feim,  name=feim_name).to_frame().reset_index().rename(columns={'index':'feature'})
        return feim
