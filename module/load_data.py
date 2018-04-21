import pandas as pd
import glob
import tqdm
import sys
import re


def load_data(input_path, key_list=[], delimiter=None, index_col=None):

    data_dict = {}
    fs_name = 'feature_set'
    path_list = glob.glob(input_path)

    if len(key_list) > 0:
        for path in path_list:
            filename = re.search(r'/([^/.]*).csv', path).group(1)
            for fn in key_list:
                if path.count(fn):
                    data_dict[fn] = pd.read_csv(path, delimiter=delimiter, index_col=index_col)
                    print('********************************')
                    print('filename      : {}'.format(filename))
                    print('row number    : {}'.format(len(data_dict[fn])))
                    print('column number : {}'.format(
                        len(data_dict[fn].columns)))
                    print('columns       : \n{}'.format(
                        data_dict[fn].columns.values))
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


# rowはリストでインデックスの範囲を指定する
def extract_set(data, index, row):
    tmp = data.set_index(index)
    return tmp.loc[row, :].reset_index()


def get_df(path):
    return pd.read_csv(path)


def pararell_load():
    p = Pool()
    tmp = pd.concat(p.map(get_df, glob.glob('tmp/*csv')), ignore_index=True)
    p.close()
    p.join()

