import pandas as pd
import numpy as np
import glob
import re
import shutil
import sys


def move_diff_file(path_1, path_2, move_path):
    '''
    path_1とpath_2にあるファイル名を比較し、
    path_2のファイル名を含むpath_1のファイルをmove_pathへ移す
    '''

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
            shutil.move(p_1, move_path)


def move_select_feature(select, input_path, move_path):
    '''
    input_pathにあるselectの文字を含むデータをmove_pathへ移す
    '''

    path_list = glob.glob(input_path)

    for path in path_list:
        if path.count(select):
            shutil.move(path, move_path)


