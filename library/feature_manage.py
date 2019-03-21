import glob
from tqdm import tqdm
import utils
import gc
import numpy as np
import pandas as pd
import sys
import re
import os
HOME = os.path.expanduser('~')
sys.path.append(f"{HOME}/kaggle/data_analysis/library/")


class FeatureManage():

    def __init__(self, key, target):
        self.key = key
        self.target = target
        self.feature_path_list = []


    def get_feature_path_list(self, feat_key_list=[], feature_path='../features/*.gz'):
        '''
        Explain:
            featureのpath_listを作成し、attributeにセットする.
            feat_key_list: 取得するfeatureを絞るkeyの2D list.
                Example...
                [
                    [feat_key1, feat_key2, feat_key3]
                    ,[feat_key1, feat_key2, feat_key3]
                    ,[feat_key1, feat_key2, feat_key3]
                ]
        Args:
        Return:
        '''
        path_list = glob.glob(feature_path)
        feat_path_list = []

        if len(feat_key_list):
            for path in path_list:
                for feat_key in feat_key_list:
                    cnt_list = [1 if path.count(
                        fkey) else 0 for fkey in feat_key]
                    if len(feat_key) == np.sum(cnt_list):
                        feat_path_list.append(path)
        else:
            feat_path_list = path_list

        self.feature_path_list = feat_path_list


    def select_feature(self, feim_path, rank=50000, gain=0):
        '''
        Explain:
            feature importance fileを元にfeatureをselectする（作成中）
        '''
        feim = pd.read_csv(feim_path)
        if gain:
            feim = feim[feim['importance']>=gain]
        else:
            feim = feim[feim['rank']<=rank]
        select_list = feim['feature'].values.tolist()

        return select_list


    def set_base(self, base):
        '''
        Explain:
            baseをtrain, testに分ける
        '''
        self.base_train = base[~base[self.target].isnull()].reset_index()
        self.base_test = base[base[self.target].isnull() ].reset_index()
        if 'index' in self.base_train.columns:
            self.base_train.drop('index', axis=1, inplace=True)
        if 'index' in self.base_test.columns:
            self.base_test.drop('index', axis=1, inplace=True)

    def feature_matrix(self, feat_key_list=[], is_reduce=False, feim_path='', rank=50000, gain=0, limit=3000):
        '''
        Explain:
            feature_path_listからtrain, testを作成する。
            feature_path_listの作成がされてない時は、featuresディレクトリ配下の
            feature全てを読み込む
        Args:
            is_reduce: メモリ削減したい時
        Return:
            train, test(DF)
        '''

        if len(self.feature_path_list) == 0:
            self.get_feature_path_list(feat_key_list=feat_key_list)[:limit]

        train_path_list = []
        test_path_list = []
        for path in self.feature_path_list:
            filename = utils.get_filename(path)

            if filename[:3] == 'tra':
                if len(feim_path):
                    if gain:
                        select_list = self.select_feature(feim_path, gain=gain)
                    else:
                        select_list = self.select_feature(feim_path, rank=rank)
                    trn_name = filename[6:]
                    if trn_name in select_list:
                        train_path_list.append(path)
                else:
                    train_path_list.append(path)

            elif filename[:3] == 'tes':
                if len(feim_path):
                    select_list = self.select_feature(feim_path, rank)
                    tes_name = filename[5:]
                    if tes_name in select_list:
                        test_path_list.append(path)
                else:
                    test_path_list.append(path)

        #========================================================================
        # Valid Feature
        valid_list = glob.glob('../features/valid_features/*.gz')

        for path in valid_list:
            filename = utils.get_filename(path)

            if filename[:3] == 'tra':
                train_path_list.append(path)

            elif filename[:3] == 'tes':
                test_path_list.append(path)
        #========================================================================

        train_list = utils.parallel_load_data(path_list=train_path_list)
        test_list = utils.parallel_load_data(path_list=test_path_list)
        train = pd.concat(train_list, axis=1)
        test = pd.concat(test_list, axis=1)

        train = pd.concat([self.base_train, train], axis=1)
        test = pd.concat([self.base_test,  test], axis=1)

        if is_reduce:
            train = utils.reduce_mem_usage(train)
            test = utils.reduce_mem_usage(test)

        return train, test
