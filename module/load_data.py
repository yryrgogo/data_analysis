import pandas as pd
import glob
import sys
import re
from multiprocessing import Pool
import multiprocessing


def load_data(path):
    return pd.read_csv(path)


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
    return csv_to_dict(*args)


def load_csv(path):
    data = pd.read_csv(path).set_index('SK_ID_CURR')
    return data


def pararell_read_csv(path_list):
    p = Pool(multiprocessing.cpu_count())
    p_list = p.map(load_csv, path_list)
    p.close

    return p_list


def csv_to_dict(key, path):
    data_dict = {}
    data_dict[key] = pd.read_csv(path)

    #  filename = re.search(r'/([^/.]*).csv', path).group(1)
    #  print('****************')
    #  print('filename   : {}'.format(filename))
    #  print('data shape : {}'.format(data_dict[key].shape))
    #  print('columns    : {}'.format(data_dict[key].columns.values))
    #  print('****************\n')
    return data_dict


def path_info(path):

    path_dict = {}
    path_dict['filename'] = re.search(r'/([^/.]*).csv', path).group(1)  # Linux
    path_dict['particle'] = re.search(r'feature_([0-9]+)@', path).group(1)
    path_dict['time'] = re.search(r'@([^.]*)@', path).group(1)
    path_dict['elem'] = re.search(r'\D@([^.]*).csv', path).group(1)

    return path_dict

