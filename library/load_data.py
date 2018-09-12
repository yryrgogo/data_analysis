import numpy as np
import pandas as pd
import glob
import sys
import re
from multiprocessing import Pool
import multiprocessing
import os
HOME = os.path.expanduser('~')
sys.path.append(f"{HOME}/kaggle/github/library/")
import utils
from utils import logger_func, get_categorical_features, get_numeric_features, pararell_process
pd.set_option('max_columns', 200)
pd.set_option('max_rows', 200)


def x_y_split(data, target):
    x = data.drop(target, axis=1)
    y = data[target].values
    return x, y

def load_file(path):
    if path.count('.csv'):
        return pd.read_csv(path)
    elif path.count('.npy'):
        filename = re.search(r'/([^/.]*).npy', path).group(1)
        tmp = pd.Series(np.load(path), name=filename)
        return tmp
    elif path.count('.fp'):
        filename = re.search(r'/([^/.]*).fp', path).group(1)
        tmp = pd.Series(utils.read_pkl_gzip(path), name=filename)
        return tmp


def pararell_load_data(path_list):
    p = Pool(multiprocessing.cpu_count())
    p_list = p.map(load_file, path_list)
    p.close

    return p_list


def path_info(path):

    path_dict = {}
    path_dict['filename'] = re.search(r'/([^/.]*).csv', path).group(1)  # Linux
    path_dict['particle'] = re.search(r'feature_([0-9]+)@', path).group(1)
    path_dict['time'] = re.search(r'@([^.]*)@', path).group(1)
    path_dict['elem'] = re.search(r'\D@([^.]*).csv', path).group(1)

    return path_dict

