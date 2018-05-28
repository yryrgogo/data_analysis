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
    winner_list = select_feature()

    #  winner_list = []
    #  for path in path_list:
    #      filename = re.search(r'/([^/.]*).csv', path).group(1)
    #      winner_list.append(filename[2:])

    path_list = glob.glob('../features/1_second_valid/*.npy')

    for path in path_list:
        for win in winner_list:
            win = win.replace(' ', '_')
            if path.count(win):
                print(path)
                shutil.move(path, '../features/4_loser/')



def shape_check_move():
    #  path_list = glob.glob('../features/1_first_valid/*.npy')
    path_list = glob.glob('../features/1_second_valid/*.npy')
    for path in path_list:
        feature = np.load(path)
        print(len(feature))
        #  if len(feature[~np.isnan(feature)])<100:
        #      shutil.move(path, '../features/1_third_valid/')


def head_tail(data):
    print(data.head())
    print(data.tail())
    sys.exit()


def col_check(data):
    for col in data.columns:
        print(col)
    sys.exit()


def list_check(elems):
    for ele in elems:
        print(ele)
    sys.exit()


def main():
    move_feature()


if __name__=='__main__':
    main()
