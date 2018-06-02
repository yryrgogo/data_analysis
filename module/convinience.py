import pandas as pd
import numpy as np
import pandas as pd
import glob
import re
import shutil
import sys
sys.path.append('../engineering/')
from select_feature import select_feature


def move_feature():

    #  path_list = glob.glob('../features/3_winner/*.csv')

    #  winner_list = []
    #  for path in path_list:
    #      filename = re.search(r'/([^/.]*).csv', path).group(1)
    #      winner_list.append(filename[2:])

    #  winner_list = pd.read_csv('../output/use_feature/auc_0.7761917332858363_feature_importance.csv')['feature'].values
    path_list = glob.glob('../features/3_winner/*.npy')
    winner_list = pd.read_csv('../prediction/use_features/20180531_13_valid2_use_169col_auc_0_7857869389179928.csv')['feature'].values

    for path in path_list:
        for win in winner_list:
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
        if duplicate_flg==0:
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


def head_tail(data):
    sys.exit()


def col_part_shape_cnt_check(data, col=0, part=0, shape=0, count=0, nunique=0):

    if col==1:
        for col in data.columns:
            print(col)

    if part==1:
        print(data.head())
        print(data.tail())

    if shape==1:
        print(data.shape)

    if count==1:
        print(data.count())

    if nunique==1:
        print(data.nunique())


def list_check(elems):
    for ele in elems:
        print(ele)
    sys.exit()


def main():
    print()
    #  move_feature()


if __name__=='__main__':
    main()
