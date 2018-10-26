import numpy as np
import pandas as pd
import multiprocessing
import gc
from scipy.stats.mstats import mquantiles
from tqdm import tqdm
import sys
import os
HOME = os.path.expanduser('~')
sys.path.append(f"{HOME}/kaggle/data_analysis/library/")
from pararell_utils import pararell_process
from caliculate_utils import round_size

kaggle = 'home-credit-default-risk'

class Xray_Cal:

    def __init__(self, logger, model, ignore_list=[]):
        self.logger = logger
        self.model = model
        self.ignore_list = ignore_list
        self.df_xray=[]
        self.point_dict = {}
        self.N_dict = {}

    def pararell_xray_caliculation(self, col, val, N):
        # TODO 並列プロセス内での更新はプロセス内でのみ適用されるはず
        dataset = self.df_xray
        dataset[col] = val
        pred = self.model.predict(dataset)
        p_avg = np.mean(pred)

        self.logger.info(f'''
#========================================================================
# CALICULATION PROGRESS... COLUMN: {col} | VALUE: {val} | X-RAY: {p_avg}
#========================================================================''')
        return {
            'feature':col,
            'value'  :val,
            'xray'   :p_avg,
            'N'      :N
        }


    def pararell_xray_wrapper(self, args):
        return self.pararell_xray_caliculation(*args)


    def single_xray_caliculation(self, col, val, N):

        dataset = self.df_xray.copy()
        dataset[col] = val
        pred = self.model.predict(dataset)
        gc.collect()
        p_avg = np.mean(pred)

        self.logger.info(f'''
#========================================================================
# CALICULATION PROGRESS... COLUMN: {col} | VALUE: {val} | X-RAY: {p_avg}
#========================================================================''')
        return {
            'feature':col,
            'value'  :val,
            'xray'   :p_avg,
            'N'      :N
        }


    def get_xray(self, base_xray, fold_num, col_list=[], max_point=30, N_sample=300000, ex_feature_list=[], Pararell=False, cpu_cnt=multiprocessing.cpu_count()):
        '''
        Explain:
        Args:
            model  : 何番目のfoldsのモデルから出力するか
            col_list  : x-rayを出力したいカラムリスト.引数なしの場合はデータセットの全カラム
            max_point : x-rayを可視化するデータポイント数
            ex_feature: データポイントの取得方法が特殊なfeature_list
        Return:
        '''
        result = pd.DataFrame([])

        if len(col_list)==0:
            col_list = base_xray.columns
        for i, col in enumerate(col_list):

            # ignore_listにあるfeatureはcontinue
            if col in self.ignore_list: continue

            if fold_num==0:
                null_values = base_xray[col][base_xray[col].isnull()].values
                if len(null_values)>0:
                    null_value = null_values[0]
                    null_cnt = len(null_values)

                df_not_null = base_xray[~base_xray[col].isnull()]

                #========================================================================
                # Get X-RAY Data Point
                # 1. 対象カラムの各値のサンプル数をカウントし、割合を算出。
                # 2. 全体においてサンプル数の少ない値は閾値で切ってX-RAYを算出しない
                #========================================================================
                val_cnt = df_not_null[col].value_counts().reset_index().rename(columns={'index':col, col:'cnt'})

                # 初回ループで全量データを使いデータポイント(bin)と各binのサンプル数を取得する
                # max_pointよりnuniqueが大きい場合、データポイントはmax_pointにする
                # binによる中央値と10パーセンタイルをとり, 分布全体のポイントを取得できるようにする
                if len(val_cnt)>max_point:
                    # 1. binにして中央値をとりデータポイントとする
                    bins = max_point-10
                    tmp_points = pd.qcut(x=df_not_null[col], q=bins, duplicates='drop')
                    tmp_points.name = f'bin_{col}'
                    tmp_points = pd.concat([tmp_points, df_not_null[col]], axis=1)
                    # 各binの中央値をデータポイントとする
                    mode_points = tmp_points[[f'bin_{col}', col]].groupby(f'bin_{col}')[col].median().to_frame()
                    # 各binのサンプル数を計算
                    data_N = tmp_points[[f'bin_{col}', col]].groupby(f'bin_{col}')[col].size().rename(columns={col:'N'})
                    mode_points['N'] = data_N

                    # 2. binの中央値と合わせ、percentileで10データポイントとる
                    percentiles = np.linspace(0.05, 0.95, num=10)
                    percentiles_points = mquantiles(val_cnt.index.values, prob=percentiles, axis=0)
                    max_val = df_not_null[col].max()
                    min_val = df_not_null[col].min()
                    # 小数点以下が大きい場合、第何位までを計算するか取得して丸める
                    r = round_size(max_val, max_val, min_val)
                    percentiles_points = np.round(percentiles_points, r)
                    # data point
                    data_points = list(np.hstack((mode_points[col], percentiles_points)))
                    # data N
                    data_N = list(np.hstack((mode_points['N'], np.zeros(len(percentiles_points))+np.nan )))
                else:
                    length = len(val_cnt)
                    data_points = list(val_cnt.head(length).index.values) # indexにデータポイント, cntにサンプル数が入ってる
                    data_N = list(val_cnt['cnt'].head(length).values) # indexにデータポイント, cntにサンプル数が入ってる

                if len(null_values)>0:
                    data_points.append(null_value)
                    data_N.append(null_cnt)

                # X-RAYの計算する際は300,000行くらいでないと重くて時間がかかりすぎるのでサンプリング
                # データポイント、N数は各fold_modelで共通の為、初期データでのみ取得する
                self.point_dict[col] = data_points
                self.N_dict[col] = data_N

            self.df_xray = base_xray.sample(N_sample, random_state=fold_num)
            #========================================================================
            # 一番計算が重くなる部分
            # multi_processにするとprocess数分のデータセットをメモリに乗せる必要が
            # あり, Overheadがめちゃでかく大量のメモリを食う。また、各データポイントに
            # ついてpredictを行うので、毎回全CPUを使っての予測が行われる。
            # また、modelオブジェクトも重いらしく引数にmodelを載せて並列すると死ぬ
            # データセットを逐一更新して予測を行うので、データセットの非同期並列必須
            # TODO DataFrameでなくnumpyでデータセットをattributeに持たせてみる?
            #========================================================================
            if Pararell:
                arg_list = []
                for point, N in zip(self.point_dict[col], self.N_dict[col]):
                    arg_list.append([col, point, N])
                xray_list = pararell_process(self.pararell_xray_wrapper, arg_list, cpu_cnt=cpu_cnt)
            else:
                xray_list = []
                for point, N in zip(self.point_dict[col], self.N_dict[col]):
                    one_xray = self.single_xray_caliculation(col=col, val=point, N=N)
                    xray_list.append(one_xray)
            #========================================================================
            # 各featureのX-RAYの結果を統合
            #========================================================================
            tmp_result = pd.DataFrame(data=xray_list)

            xray_list = []
            gc.collect()

            if len(result):
                result = pd.concat([result, tmp_result], axis=0)
            else:
                result = tmp_result.copy()

        self.logger.info(f"FOLD: {fold_num}")
        # 全てのfeatureのNとdata_pointを取得したら、全量データは必要なし
        try:
            del df_not_null
            del base_xray
        except UnboundLocalError:
            del base_xray
        gc.collect()

        return self, result
