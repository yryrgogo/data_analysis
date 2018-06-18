import pandas as pd
import numpy as np
import pandas as pd
import glob
import re
import shutil
import sys
sys.path.append('../engineering/')
from select_feature import select_feature


unique_id = 'SK_ID_CURR'
target = 'TARGET'


def move_feature():

    #  path_list = glob.glob('../features/3_winner/*.csv')

    #  winner_list = []
    #  for path in path_list:
    #      filename = re.search(r'/([^/.]*).csv', path).group(1)
    #      winner_list.append(filename[2:])

    #  winner_list = pd.read_csv('../output/use_feature/auc_0.7761917332858363_feature_importance.csv')['feature'].values
    path_list = glob.glob('../features/3_winner/*.npy')
    #  winner_list = pd.read_csv('../prediction/use_features/20180531_13_valid2_use_169col_auc_0_7857869389179928.csv')['feature'].values

    for path in path_list:
        for win in lose_list:
            #  win = win.replace(' ', '_')
            print(win)
            #  win = f'a_{win}.npy'
            if path.count(win):
                print(path)
                shutil.move(path, '../features/1_third_valid/')


def move_diff_file(path_1, path_2):

    path_list1 = glob.glob(path_1)
    path_list2 = glob.glob(path_2)

    for p_1 in path_list1:
        filename_1 = re.search(r'/([^/.]*).npy', p_1).group(1)
        duplicate_flg = 0
        for p_2 in path_list2:
            filename_2 = re.search(r'/([^/.]*).npy', p_2).group(1)
            print(filename_1)
            print(filename_2)
            if filename_1.count(filename_2):
                duplicate_flg = 1
        if duplicate_flg == 0:
            shutil.move(p_1, '../features/tmp/')


def move_target_feature():

    path_list = glob.glob('../features/1_first_valid/*.npy')

    for path in path_list:
        if not(path.count('TARGET')):
            shutil.move(path, '../features/1_third_valid/')


def shape_check_move():
    #  path_list = glob.glob('../features/1_first_valid/*.npy')
    path_list = glob.glob('../features/1_second_valid/*.npy')
    for path in path_list:
        feature = np.load(path)
        print(len(feature))
        #  if len(feature[~np.isnan(feature)])<100:
        #      shutil.move(path, '../features/1_third_valid/')


def col_part_shape_cnt_check(data, col=0, part=0, shape=0, count=0, nunique=0):

    if col == 1:
        for col in data.columns:
            print(col)

    if part == 1:
        print(data.head())
        print(data.tail())

    if shape == 1:
        print(data.shape)

    if count == 1:
        print(data.count())

    if nunique == 1:
        print(data.nunique())


def list_check(elems):
    for ele in elems:
        print(ele)
    sys.exit()


def check_sort(data=[]):
    '''
    作成したnpyファイルがちゃんと意図したソート順になっているかを
    チェックする.
    10IDほど、trainとtestでそれぞれピックアップしておき、値を
    比較できるようにしておく。
    imputeはNullでない値を保持しておけば、正しい値がすぐわかる.
    engineeringした値は、npyファイルとIDの入ったcsvを比較する
    しかない
    '''

    if len(data) == 0:
        base = pd.read_csv(
            '../data/application_train_test.csv')[[unique_id, 'is_test']]
        path_list = glob.glob('../features/5_check_feature/*.npy')

    ' チェック用のID。このIDについては元データの値をすぐ参照できるようにエクセルで保存しておく '
    train_id = [
        100002,
        100003,
        100004,
        100006,
        100007,
        100008,
        100009,
        100010,
        100011,
        100012
    ]
    test_id = [
        100001,
        100005,
        100013,
        100028,
        100038,
        100042,
        100057,
        100065,
        100066,
        100067
    ]

    if len(data) > 0:
        print(data.loc[train_id, :])
        print(data.loc[test_id, :])
        return

    for path in path_list:

        filename = re.search(r'/([^/.]*).npy', path).group(1)
        feature = pd.Series(np.load(path), name=filename)
        data = pd.concat([base, feature], axis=1).set_index(unique_id)

        print(data.loc[train_id, :])
        print(data.loc[test_id, :])


def main():
    print()
    #  move_feature()


if __name__ == '__main__':
    main()
