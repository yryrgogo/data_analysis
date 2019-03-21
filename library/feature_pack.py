from tqdm import tqdm
from feature_engineering import diff_feature, division_feature, product_feature, cnt_encoding, select_category_value_agg, exclude_feature, target_encoding
from utils import get_categorical_features, get_numeric_features, parallel_process
import utils
import gc
import numpy as np
import pandas as pd
import sys
import re
from glob import glob
import os
from itertools import combinations, chain
from joblib import Parallel, delayed
HOME = os.path.expanduser('~')
sys.path.append(f"{HOME}/kaggle/data_analysis/library/")
#  sys.path.append(f"/mnt/c/Git/go/kaggle/github/library/")


class FeaturePack():

    def __init__(self, data, key, target, ignore_list=[], base=[]):
        self.ignore_list = ignore_list
        self.data = data
        self.key = key
        self.target = target
        self.base = base


    def combi_calculate(self, df, num_list):
        used_list = []
        for f1 in num_list:
            for f2 in num_list:
                if f1 == f2:
                    continue
                if sorted([f1, f2]) in used_list:
                    continue
                used_list.append(sorted([f1, f2]))
                df = division_feature(df=df, first=f1, second=f2)
                df = diff_feature(df=df, first=f1, second=f2)
        return df


    def name_clean(self, name):
        return name.replace(r'/', '_').replace(r'(', '_').replace(r')', '_').replace(r'.', '-').replace(r'.', '_')


    def go_agg(self, df_agg, col_key='', method_list=['sum', 'mean', 'max', 'min', 'std']):
        """
        Explain:
            集計はattributeのkeyを使って行われる
        Args:
            df_agg : 集計を行いたいデータフレーム
            col_key: 集計を行いたいカラムを絞るキー

        """
        aggs = {}
        for col in df_agg.columns:
            if not(col.count(col_key)):
                continue
            aggs[col] = method_list

        df_agg = df_agg.groupby(self.key).agg(aggs)

        new_col_list = [self.name_clean(
            col) + f"-{method}" for col in aggs.keys() for method in aggs[col]]
        df_agg.columns = new_col_list

        return df_agg

    def go_ohe(self, cat_list, method_list=[], prefix=''):
        """
        Explain:
            One Hot Encodingを行う.
        Args:
            cat_list: Encodingするカテゴリカルのカラムリスト
            prefix: stringが入ってる時は、作成したfeatureのsaveも行う
        """
        with utils.timer("One Hot Encoding"):
            tmp_list = []
            for col in tqdm(cat_list):
                df_ohe = pd.get_dummies(
                    data=self.data[col], columns=col, prefix=f'ohe_{col}')
                df_ohe[self.key] = self.data[self.key].values

                # 集計が必要な場合のみ
                if len(method_list):
                    df_agg = self.go_agg(
                        df_agg=df_ohe, col_key='ohe_', method_list=method_list)
                tmp_list.append(df_agg.copy())
                del df_ohe
                gc.collect()

            df_ohe = pd.concat(tmp_list, axis=1)

            df_ohe = self.base.join(df_ohe)

            if len(prefix):
                self.save_feature(df_feat=df_ohe, prefix=prefix, is_train=2)
                return df_ohe
            else:
                return df_ohe

    def go_label(self, cat_list, method_list=[], prefix=''):
        """
        Explain:
            Label Encodingを行う.
        Args:
            cat_list: Encodingするカテゴリカルのカラムリスト
            prefix: stringが入ってる時は、作成したfeatureのsaveも行う
        """
        with utils.timer("Factorize Encoding"):
            for col in tqdm(cat_list):
                self.data[f"label_{col}"], _ = pd.factorize(self.data[col])

        with utils.timer("Aggregation"):
            label_cols = [
                col for col in self.data.columns if col.count('label_')]

            if len(method_list):
                df_agg_label = self.go_agg(
                    df_agg=self.data[[self.key] + label_cols], col_key='label_', method_list=method_list)
                df_agg_label = self.base.join(df_agg_label)
            else:
                df_agg_label = self.data[[self.key]+label_cols].set_index(self.key)

        if len(prefix):
            self.save_feature(df_feat=df_agg_label, prefix=prefix, is_train=2)
            return df_agg_label
        else:
            return df_agg_label

    def go_count(self, cat_list, method_list=[], prefix=''):
        """
        Explain:
            Count Encodingを行う.
        Args:
            cat_list: Encodingするカテゴリカルのカラムリスト
            prefix: stringが入ってる時は、作成したfeatureのsaveも行う
        """
        tmp_list = []
        for col in tqdm(cat_list):
            df_cnt = cnt_encoding(
                self.data[[self.key, col]], col, ignore_list=self.ignore_list)
            tmp_list.append(df_cnt.set_index(self.key))
        df_cnt = pd.concat(tmp_list, axis=1)

        if len(method_list):
            df_agg_cnt = self.go_agg(
                df_agg=df_cnt, col_key='cnt_', method_list=method_list)
            df_agg_cnt = self.base.join(df_agg_cnt)
        else:
            df_agg_cnt = df_cnt.set_index(self.key)

        del df_cnt
        gc.collect()

        if len(prefix):
            self.save_feature(df_feat=df_agg_cnt, prefix=prefix, is_train=2)
            return df_agg_cnt
        else:
            return df_agg_cnt


    def go_interact(self, combi_list=[], num_list=[], is_diff=False, is_div=False, is_pro=False, prefix='', n_jobs=0):

        with utils.timer(f"Calcurate Intaract"):
            if len(combi_list)==0:
                combi_list = list(combinations(num_list, 2))

            feature_list = []

            def calculate(f1, f2):
                df = self.data[[f1, f2]]
                if is_diff:
                    feat = diff_feature(df=df, first=f1, second=f2, only_feat=True)
                if is_div:
                    feat = division_feature(df=df , first=f1, second=f2, only_feat=True)
                if is_pro:
                    feat = product_feature(df=df, first=f1, second=f2, only_feat=True)

                return feat

            if n_jobs==0:
                for (f1, f2) in combi_list:
                    feature_list.append(diff_feature(f1, f2))
            else:

                length = len(combi_list)
                p_num = int(length / n_jobs)
                rem = length % n_jobs

                for i in range(p_num):
                    tmp_list = combi_list[i*n_jobs : (i+1)*n_jobs]

                    p_list = Parallel(n_jobs=n_jobs)(
                        [delayed(calculate)(f1, f2)
                         for (f1, f2) in tmp_list
                         ]
                    )

                    #  feature_list += list(chain(*p_list))
                    if len(prefix):
                        df_feat = pd.concat(p_list, axis=1)

                        df_feat[self.key] = self.base.index.tolist()
                        df_feat[self.target] = self.base[self.target].values

                        self.save_feature(df_feat=df_feat, prefix=prefix, is_train=2)
                        del p_list, df_feat
                        gc.collect()
                    else:
                        feature_list += p_list

                if rem>0:
                    tmp_list = combi_list[p_num*n_jobs:]

                    p_list = Parallel(n_jobs=-1)(
                        [delayed(calculate)(f1, f2)
                         for (f1, f2) in tmp_list
                         ]
                    )

                    if len(prefix):
                        df_feat = pd.concat(p_list, axis=1)

                        df_feat[self.key] = self.base.index.tolist()
                        df_feat[self.target] = self.base[self.target].values

                        self.save_feature(df_feat=df_feat, prefix=prefix, is_train=2)
                        del p_list, df_feat
                        gc.collect()
                    else:
                        feature_list += p_list


        if len(prefix):
            return

        df_feat = pd.concat(feature_list, axis=1)
        df_feat[self.key] = self.base.index.tolist()
        df_feat[self.target] = self.base[self.target].values

        if n_jobs==0:
            self.data.drop(df_feat.columns, axis=1, inplace=True)

        return df_feat


    def save_feature(self, df_feat, prefix, is_train=2):
        """
        is_train: 0->train only, 1->test only, 2->train & test
        """
        utils.save_feature(df_feat=df_feat, is_train=is_train,
                           target=self.target, ignore_list=self.ignore_list, prefix=prefix)
