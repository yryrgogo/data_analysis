import pandas as pd
import glob
import tqdm
import sys


def load_data(input_path, fn_list=[]):

    cnt = 0
    data_dict = {}
    path_list = glob.glob(input_path)

    if len(fn_list) == len(path_list):
        while cnt != len(path_list):
            for path in path_list:
                for fn in fn_list:
                    if path.count(fn):
                        data_dict[fn] = pd.read_csv(path)
                        fn_list.remove(fn)
                        path_list.remove(path)
                        print('filename : {}'.format(fn))
                        print(data_dict[fn].count())
                        print(data_dict[fn].head())
                        cnt += 1
        print('{} file load end.'.format(cnt))
        return data_dict

    elif len(fn_list) == 0 and len(path_list) == 1:
        return pd.read_csv(path_list[0])
    else:
        print('filiname number :{}'.format(len(fn_list)))
        print('path number     :{}'.format(len(path_list)))
        print('filename is shortage.')
        sys.exit()


def make_dataset():
    df = data.loc[season, :]
    df.reset_index(inplace=True)
    test_season = df['season'].max()

    tmp_train = df[df['season'] != test_season]
    tmp_test = df[df['season'] == test_season]
    tmp_train2 = tmp_test[tmp_test['daynum'] < 133]

    train = pd.concat([tmp_train, tmp_train2], axis=0)
    test = tmp_test[tmp_test['daynum'] >= 133]

    x_train = train.drop(
        ['result', 'teamid', 'teamid_2', 'season'], axis=1).copy()
    y_train = train['result'].values
    x_test = test.drop(
        ['result', 'teamid', 'teamid_2', 'season'], axis=1).copy()
    y_test = test['result'].values
