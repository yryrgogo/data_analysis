import pandas as pd
import glob
import sys
import re
from multiprocessing import Pool
import multiprocessing


def load_data(input_path, key_list=[], delimiter=None, index_col=None):

    data_dict = {}
    fs_name = 'feature_set'
    path_list = glob.glob(input_path)

    if len(key_list) > 0:
        for path in path_list:
            filename = re.search(r'/([^/.]*).csv', path).group(1)
            for key in key_list:
                if path.count(key):
                    data_dict[key] = pd.read_csv(path, delimiter=delimiter, index_col=index_col)
                    print('********************************')
                    print('filename      : {}'.format(filename))
                    print('row number    : {}'.format(len(data_dict[fn])))
                    print('column number : {}'.format(len(data_dict[fn].columns)))
                    print('columns       : \n{}'.format(data_dict[fn].columns.values))
                    print('********************************\n')
        print('{} file load end.\nreturn {}'.format(len(key_list), data_dict.keys()))
        print('********************************\n')
        return data_dict, fs_name

    elif len(key_list) == 0:
        fs_name = re.search(r'/([^/.]*).csv', path_list[0]).group(1)
        return pd.read_csv(path_list[0]), fs_name


def x_y_split(data, target):
    x = data.drop(target, axis=1)
    y = data[target].values
    return x, y


def pararell_load_data(key_list, path_list):
    """
    Explain:
    ファイル名をkeyとしてデータを格納したdictを並列処理で作成し、
    そのdictを格納したリストを返す

    Args:
        key_list(list)  : pathからファイル名を特定する為のkey
        path_list(list) : loadしたいファイルが入ったパスリスト

    Return:
        p_list(list) : ファイル名のkeyをキーとしたデータセットの辞書リスト

    """

    arg_list = []

    for path in path_list:
        for key in key_list:
            if path.count(key):
                arg_list.append([key, path])

    p_list = pararell_process(load_wrapper, arg_list)

    return p_list


def load_wrapper(args):
    return pararell_read_csv(*args)


def pararell_process(func, arg_list):
    p = Pool(multiprocessing.cpu_count())
    p_list = p.map(func, arg_list)
    p.close

    return p_list


def pararell_read_csv(key, path):
    data_dict = {}
    data_dict[key] = pd.read_csv(path)

    filename = re.search(r'/([^/.]*).csv', path).group(1)
    #  print('****************')
    #  print('filename   : {}'.format(filename))
    #  print('data shape : {}'.format(data_dict[key].shape))
    #  print('columns    : {}'.format(data_dict[key].columns.values))
    #  print('****************\n')
    return data_dict


