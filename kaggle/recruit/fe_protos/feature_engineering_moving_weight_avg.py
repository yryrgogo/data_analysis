import numpy as np
import pandas as pd
import datetime
from time import time
import sys
import glob
import re
from tqdm import tqdm
from multiprocessing import Pool
import multiprocessing

' 自作モジュール '
from feature_engineering import exp_weight_avg, moving_agg
from recruit_make_input_data import consective_holiday

sys.path.append('../protos/')
from recruit_kaggle_load import recruit_load_data, set_validation, load_submit

sys.path.append('../../../module/')
from preprocessing import outlier, impute_avg, date_range, date_diff, lag_feature

start_time = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())

""" データセット関連 """
input_path = '../input/*.csv'
submit_path = '../input/sample_submission.csv'
path_list = glob.glob(input_path)
key_air_vi = 'air_vi_ex'
key_air_re = 'air_re'
key_air_st = 'air_st'
key_air_cal = 'air_cal_june_ex'
key_raw_cal = 'air_cal_june【'
key_air_area = 'air_area'
key_air_genre = 'air_genre'
key_air_genre = 'air_genre'
key_list = [
    key_air_st,
    #  key_air_vi,
    key_air_area,
    key_air_genre,
    key_air_cal,
    key_raw_cal]

""" データロード """
#  base = pd.read_csv('../input/20180424_air_base.csv')
data_dict = recruit_load_data(key_list, path_list)
#  air_vi = data_dict[key_air_vi]
air_st = data_dict[key_air_st]
air_cal = data_dict[key_air_cal]
raw_cal = data_dict[key_raw_cal]
#  air_re = data_dict[key_air_re]
air_area = data_dict[key_air_area]
air_genre = data_dict[key_air_genre]

first_date = air_cal['visit_date'].min()
last_date = air_cal['visit_date'].max()

''' カレンダーの日付を絞る '''
air_cal = air_cal[air_cal['visit_date'] <= last_date]


def let_wg_agg(data, level, window_list):
    """ 重み付き平均 """
    #  weight_list = [0.9, 0.95, 0.98]
    #  ''' 学習データの日程（重み付き平均は1日ずつ計算してあげる必要があるので全日付リストが必要） '''
    #  date_list = air_vi['visit_date'].drop_duplicates().sort_values().values
    #  for weight in weight_list:
    #      result = pd.DataFrame([])
    #      for end_date in date_list:
    #          tmp = date_range(air_vi, first_date, end_date)
    #          wg_avg = exp_weight_avg(tmp, 'air_store_id', 'last_visit', weight).to_frame().reset_index()
    #          wg_avg['visit_date'] = end_date

    #          if len(result)==0:
    #              result = wg_avg
    #          else:
    #              result = pd.concat([result, wg_avg], axis=0)

    #      result = air_vi[['air_store_id', 'visit_date']].merge(result, on=['air_store_id', 'visit_date'], how='inner')
    #      result.sort_values(by=['air_store_id', 'visit_date'], inplace=True)
    #      result = result.iloc[:,2]
    #      result.to_csv('../feature/{}.csv'.format(result.name), header=True, index=False)
    #      print(result.shape)

    #  sys.exit()


def let_mv_agg(data, level, number, value, method_list, window_list, period, lag, segment=None, segment_list=['nothing']):
    """
    Explain:
        移動平均の前処理が必要な場合の処理をまとめた関数。

    Args:
        data(DF)          :
        level(list)       : 移動平均結果のvalueカラムとは別に残すカラムリスト
                            頭(0)から移動平均の集計粒度となり、次のindex変数に
                            集計粒度カラム数が格納される。
                            例：idの粒度で集計し、id * dateの粒度で結果DFを作る
                                場合は、リストに[id, date]、index=1となる
                            ※リストの要素順は注意
        index(int)        : 集計に使うlevel内の要素の数。頭(0)から使う
        value(float)      : 集計する値
        method(sum|avg)   :
        window_list(list) : 移動平均をとる行数（自分の行を含む前〇行）
        period(int)       : 移動平均をとる行内にNullなどあった場合、最低何要素
                            あったら値を算出するか？（足りない時はNull）
        lag(int)          : shiftによってずらす行の数
                           （正：前lag行のデータ、 負：後lag行のデータ）
        segment(column)   : セグメントの入ったカラム名
        segment_list(list): セグメントに分けて集計を実施する場合、その
                            セグメントに切り分ける要素を格納したリスト
                            セグメントがない場合は引数なしで'nothing'が入る
    Return:
        なし。csvをoutput

    """

    ' 連休、非連休の祝日データセット作成 '
    consective, discrete = consective_holiday('air_cal_next')
    consective.sort_values(by=level, inplace=True)
    discrete.sort_values(by=level, inplace=True)

    """ 移動平均 """
    for method in method_list:
        for window in window_list:
            seg_result = pd.DataFrame([])
            for seg in segment_list:
                if seg != 'nothing':
                    ' 対象セグメントに絞る '
                    df_seg = data[data[segment] == seg]

                    '''
                    祝日はその曜日の移動平均に含めないが、週カウントは行う。
                    その為、祝日は値をNullにして行のみ残す
                    NULLにしたい値をもつカラムをdropしてtmpDFに格納（その行は
                    集計しないだけでカウントしたいので、インデックスは残す為）
                    '''
                    ' Nullにしたいレコードを落とす '
                    tmp = df_seg.drop(value, axis=1)
                    key_columns = list(tmp.columns.values)
                    df_seg = df_seg[df_seg['day_of_week'] != 'Special']
                    '''
                    全てのインデックスを持ったDFに、NULLにしたいレコードを落とした
                    DFをLeft Joinすることで、対象レコードのvalueのみNullにする
                    '''
                    dataset = tmp.merge(
                        df_seg, on=key_columns, how='left', copy=False)

                else:
                    dataset = data

                ' 目的変数をエンコーディングする為、リークしない時系列でもたせる '
                dataset = lag_feature(
                    data=dataset, value=value, lag=lag, level=level)

                ' コア部分：移動平均 '
                seg_mv = moving_agg(method, dataset, level,
                                    number, value, window, 1, 'visit_date')
                ' 特徴量は@を持ったカラムとしているので、これで取得できる '
                col_name = [col for col in seg_mv.columns if col.count('@')][0]

                if seg != 'nothing':
                    '''
                    集計後はNULLでカウントさせてたセグメントのレコードは必要ないので
                    inner joinで除外する
                    '''
                    seg_mv = seg_mv.merge(df_seg[level], on=level, how='inner')

                ' 欠損値を平均で補完 '
                tmp_result = impute_avg(seg_mv, level, number, col_name)

                if seg == 'nothing':
                    tmp_result.to_csv(
                        f'../feature/valid_feature/{col_name}.csv', header=True, index=False)
                    return
                    ' セグメント別に移動平均を集計しない場合、ここで完了 '

                if len(seg_result) == 0:
                    seg_result = tmp_result
                else:
                    seg_result = pd.concat([seg_result, tmp_result], axis=0)

            '''
            ********************************
            ********************************
            **連休セグをもっと綺麗に分け、**
            **上記のループに含めることはで**
            **きないか？
            ********************************
            ********************************
            '''
            consective = lag_feature(
                data=consective, value=value, lag=lag, level=level)
            discrete = lag_feature(
                data=discrete, value=value, lag=lag, level=level)

            ''' 連休の特徴量 '''
            consective_mv = moving_agg(
                method, consective, level, number, value, window, 1, 'visit_date')
            ' feature nameを取得 '
            col_name = [
                col for col in consective_mv.columns if col.count('@')][0]
            ' NULL埋め '
            result_cont = impute_avg(consective_mv, level, number, col_name)

            ''' 非連休の特徴量 '''
            discrete_mv = moving_agg(
                method, discrete, level, number, value, window, 1, 'visit_date')
            ' feature nameを取得 '
            col_name = [
                col for col in discrete_mv.columns if col.count('@')][0]
            ' NULL埋め '
            result_disc = impute_avg(discrete_mv, level, number, col_name)

            ''' 結合 '''
            result = pd.concat([seg_result, result_cont, result_disc], axis=0)

            ' 日程を元のデータセットと同様にする '
            result = air_cal[level].merge(result, on=level, how='inner')

            result.sort_values(by=level, inplace=True)
            #  result = result[col_name]
            result.to_csv(f'../feature/{col_name}.csv',
                          index=False, header=True)
            print(result.shape)
            print(result.head())


def let_wg_agg(data, level, number, value, method_list, weight_list, index, lag, segment=None, segment_list=['nothing']):
    """
    Explain:
        移動平均の前処理が必要な場合の処理をまとめた関数。

    Args:
        data(DF)          :
        level(list)       : 移動平均結果のvalueカラムとは別に残すカラムリスト.
                            頭(0)から移動平均の集計粒度となり、次のindex変数に
                            集計粒度カラム数が格納される。
                            例：idの粒度で集計し、id * dateの粒度で結果DFを作る
                                場合は、リストに[id, date]、index=1となる
                            ※リストの要素順は注意
        index(int)        : 集計に使うlevel内の要素の数。頭(0)から使う
        value(float)      : 集計する値
        method(sum|avg)   :
        weight_list(list) : 重み付き平均をの減衰率
        index             : 時系列などで重みの調整をする場合、そのインデックスとなるカラム名
        lag(int)          : リーク防止などで集計する値をshiftによってずらす行の数
                           （正：前lag行のデータ、 負：後lag行のデータ）
        segment(column)   : セグメントの入ったカラム名
        segment_list(list): セグメントに分けて集計を実施する場合、その
                            セグメントに切り分ける要素を格納したリスト
                            セグメントがない場合は引数なしで'nothing'が入る
    Return:
        なし。csvをoutput

    """

    ' 連休、非連休の祝日データセット作成 '
    consective, discrete = consective_holiday('air_cal_next')
    consective.sort_values(by=level, inplace=True)
    discrete.sort_values(by=level, inplace=True)

    """ 重み付き平均 """
    for method in method_list:
        for weight in weight_list:
            seg_result = pd.DataFrame([])
            ' recruit competisionの場合は、曜日毎の特徴量を作りたいので、segment=dow '
            for seg in segment_list:
                if seg != 'nothing':
                    ' 対象セグメントに絞る '
                    df_seg = data[data[segment] == seg]

                    '''
                    祝日はその曜日の重み付き平均に含めないが、週カウントは行う。
                    その為、祝日は値をNullにして行のみ残す
                    NULLにしたい値をもつカラムをdropしてtmpDFに格納（その行は
                    集計しないだけでカウントしたいので、インデックスは残す為）
                    '''
                    ' Nullにしたいレコードを落とす '
                    tmp = df_seg.drop(value, axis=1)
                    key_columns = list(tmp.columns.values)
                    df_seg = df_seg[df_seg['day_of_week'] != 'Special']
                    '''
                    全てのインデックスを持ったDFに、NULLにしたいレコードを落とした
                    DFをLeft Joinすることで、対象レコードのvalueのみNullにする
                    '''
                    dataset = tmp.merge(
                        df_seg, on=key_columns, how='left', copy=False)

                else:
                    dataset = data

                ' 目的変数を特徴量化する為、リークしない時系列でもたせる '
                dataset = lag_feature(
                    data=dataset, value=value, lag=lag, level=level)

                ''''''''''''''''''''''''''''''''''''''
                '''''  コア部分：重み付き平均    '''''
                ''''''''''''''''''''''''''''''''''''''

                if seg == 'nothing':
                    seg_wg = exp_weight_avg(
                        dataset, level[0], value, weight, index)
                    col_name = [
                        col for col in seg_wg.columns if col.count('@')][0]
                    tmp_result.to_csv(
                        f'../feature/valid_feature/{col_name}.csv', header=True, index=False)
                    return
                    ' セグメント別に集計しない場合、ここで完了 '

                index_list= dataset[index].drop_duplicates().values

                idx_stack = pd.DataFrame([])
                for idx in index_list:
                    seg_wg = exp_weight_avg(dataset, level[0], value, weight, index)
                    seg_wg = seg_wg.reset_index()
                    seg_wg[index] = idx

                    if len(idx_stack)==0:
                        idx_stack = seg_wg
                    else:
                        idx_stack = pd.concat([idx_stack, seg_wg], axis=0)
                ' 特徴量は@を持ったカラムとしているので、これで取得できる '
                col_name = [col for col in idx_stack.columns if col.count('@')][0]

                '''
                集計後はNULLでカウントさせてたセグメントのレコードは必要ないので
                inner joinで除外する
                '''
                tmp_result = idx_stack.merge(df_seg[level], on=level, how='inner')

                ' 欠損値を平均で補完 '
                tmp_result = impute_avg(tmp_result, level, 1, col_name)

                if len(seg_result) == 0:
                    seg_result = tmp_result
                else:
                    seg_result = pd.concat([seg_result, tmp_result], axis=0)

            print(seg_result.shape)
            '''
            ********************************
            **連休セグをもっと綺麗に分け、**
            **上記のループに含めることはで**
            **きないか？
            ********************************
            '''
            consective = lag_feature( data=consective, value=value, lag=lag, level=level)
            discrete = lag_feature(data=discrete, value=value, lag=lag, level=level)

            index_list= consective[index].drop_duplicates().values
            idx_stack = pd.DataFrame([])

            for idx in index_list:
                seg_wg = exp_weight_avg(consective, level[0], value, weight, index)
                seg_wg = seg_wg.reset_index()
                seg_wg[index] = idx

                if len(idx_stack)==0:
                    idx_stack = seg_wg
                else:
                    idx_stack = pd.concat([idx_stack, seg_wg], axis=0)

            ''' 連休の特徴量 '''
            consective_mv = moving_agg(
                method, consective, level, 1, value, window, 1, 'visit_date')
            ' feature nameを取得 '
            col_name = [
                col for col in consective_mv.columns if col.count('@')][0]
            ' NULL埋め '
            result_cont = impute_avg(consective_mv, level, 1, col_name)

            ''' 非連休の特徴量 '''
            discrete_mv = moving_agg(
                method, discrete, level, 1, value, window, 1, 'visit_date')
            ' feature nameを取得 '
            col_name = [
                col for col in discrete_mv.columns if col.count('@')][0]
            ' NULL埋め '
            result_disc = impute_avg(discrete_mv, level, 1, col_name)

            ''' 結合 '''
            result = pd.concat([seg_result, result_cont, result_disc], axis=0)

            ' 日程を元のデータセットと同様にする '
            result = air_cal[level].merge(result, on=level, how='inner')

            result.sort_values(by=level, inplace=True)
            #  result = result[col_name]
            result.to_csv(f'../feature/{col_name}.csv',
                          index=False, header=True)
            print(result.shape)
            print(result.head())


def main():

    #  let_mv_agg(
    #      data=air_cal,
    #      level=['air_store_id', 'visit_date'],
    #      number=1,
    #      value='visitors',
    #      method_list=['avg', 'std'],
    #      window_list=[3],
    #      period=1,
    #      lag=1,
    #      segment='dow',
    #      segment_list=list(air_cal['dow'].drop_duplicates().values)
    #  )

    let_wg_agg(
        data=air_cal,
        level=['air_store_id', 'visit_date'],
        number=1,
        value='visitors',
        method_list=['avg', 'std'],
        weight_list=[0.9, 0.95, 0.98],
        index='visit_date',
        lag=1,
        segment='dow',
        segment_list=list(air_cal['dow'].drop_duplicates().values)
    )

if __name__ == '__main__':

    main()
