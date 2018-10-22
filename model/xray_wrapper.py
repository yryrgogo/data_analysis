import numpy as np
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

    def __init__(self, model, ignore_list=[]):
        self.df_xray=[]
        self.ignore_list = ignore_list
        self.model = model

    def pararell_xray_caliculation(self, col, val):
        # TODO 並列プロセス内での更新はプロセス内でのみ適用されるはず
        dataset = self.df_xray
        dataset[col] = val
        pred = self.model.predict(dataset)
        p_avg = np.mean(pred)

        logger.info(f'''
#========================================================================
# CALICULATION PROGRESS... COLUMN: {col} | VALUE: {val} | X-RAY: {p_avg}
#========================================================================''')
        return {
            'feature':col,
            'value':val,
            'xray' :p_avg
        }


    def pararell_xray_wrapper(self, args):
        return self.pararell_xray_caliculation(*args)


    def single_xray_caliculation(self, col, val, model, df_xray):

        df_xray[col] = val
        pred = model.predict(df_xray)
        gc.collect()
        p_avg = np.mean(pred)

        self.logger.info(f'''
#========================================================================
# CALICULATION PROGRESS... COLUMN: {col} | VALUE: {val} | X-RAY: {p_avg}
#========================================================================''')
        self.xray_list.append({
            'feature':col,
            'value':val,
            'xray' :p_avg
        })


    def get_xray(self, base_xray, col_list=[], max_point=30, N=250000, ex_feature_list=[], Pararell=True, cpu_cnt=multiprocessing.cpu_count()):
        '''
        Explain:
        Args:
            model  : 何番目のfoldsのモデルから出力するか
            col_list  : x-rayを出力したいカラムリスト.引数なしの場合はデータセットの全カラム
            max_point : x-rayを可視化するデータポイント数
            ex_feature: データポイントの取得方法が特殊なfeature_list
        Return:
        '''
        base_xray = base_xray.sample(N)
        result = pd.DataFrame([])

        if len(col_list)==0:
            col_list = base_xray.columns
        for i, col in enumerate(col_list):
            if col in self.ignore_list:
                continue
            xray_list = []

            null_values = base_xray[col][base_xray[col].isnull()].values
            if len(null_values)>0:
                null_value = null_values[0]

            #========================================================================
            # Get X-RAY Data Point
            # 1. 対象カラムの各値のサンプル数をカウントし、割合を算出。
            # 2. 全体においてサンプル数の少ない値は閾値で切ってX-RAYを算出しない
            #========================================================================
            val_cnt = base_xray[col].value_counts().reset_index().rename(columns={'index':col, col:'cnt'})

            # max_pointよりnuniqueが大きい場合、max_pointに取得ポイント数を絞る.
            # 合わせて10パーセンタイルをとり, 分布全体のポイントを最低限取得できるようにする
            if len(val_cnt)>max_point:
                # 1. binにして中央値をとりデータポイントとする
                bins = max_point-10
                tmp_bin = base_xray[col].to_frame()
                tmp_points = pd.qcut(x=base_xray[col], q=bins)
                tmp_bin[f'bin_{col}'] = tmp_points
                data_points = tmp_bin[[f'bin_{col}', col]].groupby(f'bin_{col}')[col].median().values

                # 2. percentileで10データポイントとる
                percentiles = np.linspace(0.05, 0.95, num=10)
                percentiles_points = mquantiles(val_cnt.index.values, prob=percentiles, axis=0)
                max_val = base_xray[col].max()
                min_val = base_xray[col].min()
                # 小数点以下が大きい場合、第何位までを計算するか取得して丸める
                r = round_size(max_val, max_val, min_val)
                percentiles_points = np.round(percentiles_points, r)
                data_points = list(np.hstack((data_points, percentiles_points)))
            else:
                length = len(val_cnt)
                data_points = list(val_cnt.head(length).index.values) # indexにデータポイント, valueにサンプル数が入ってる

            if len(null_values)>0:
                data_points.append(null_value)

            data_points = sorted(data_points)

            self.df_xray = base_xray.copy()
            del base_xray
            gc.collect()
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
                for point in data_points:
                    arg_list.append([col, point])
                xray_list = pararell_process(self.pararell_xray_wrapper, arg_list, cpu_cnt=cpu_cnt)
            else:
                for point in data_points:
                    one_xray = self.single_xray_caliculation(col=col, val=point)
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

        return result
