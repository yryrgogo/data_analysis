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

    winner_list = pd.read_csv('../output/use_feature/auc_0.7761917332858363_feature_importance.csv')['feature'].values
    path_list = glob.glob('../features/1_first_valid/*.npy')

    for path in path_list:
        for win in winner_list:
            win = win.replace(' ', '_')
            win = f'{win}.npy'
            if path.count(win):
                print(path)
                shutil.move(path, '../features/3_winner/')


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
    move_feature()


if __name__=='__main__':
    main()
