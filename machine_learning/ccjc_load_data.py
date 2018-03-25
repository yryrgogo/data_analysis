import pandas as pd
import numpy as np
from tqdm import tqdm
from logging import getLogger
# import pdb # 実行中の変数の中身を除く
# pdb.set_trace()

TRAIN_DATA = '../../input/20171215_ccjc_ltv_day15_m3_7_cluster.csv'
# TEST_DATA = '../input/test.csv'

logger = getLogger(__name__)

def read_csv(path):
    logger.debug('enter')
    df = pd.read_csv(path)
    df = df.drop('users_id', axis=1)

    # categorical trans to dummy variable 
    for col in tqdm(df.columns.values):
        # categorical -> roop
        if 'str' in str(type(df[col][0])):
            logger.info('categorical: {}'.format(col))
            tmp  =pd.get_dummies(df[col], col)
            for col2 in tmp.columns.values:
                #as much as possible numpy
                df[col2] = tmp[col2].values
            df.drop(col, axis=1, inplace=True) # inplace=True don't create copy
    logger.debug('exit')
    return df


def load_train_data():
    logger.debug('enter')
    df = read_csv(TRAIN_DATA)
    logger.debug('exit')
    return df


def load_test_data():
    logger.debug('enter')
    df = read_csv(TEST_DATA)
    logger.debug('exit')
    return df


def create_train():
    logger.debug('enter')
    df = read_csv(TRAIN_DATA)
    logger.debug('exit')
    N = df.shape[0]
    N_train = int(df.shape[0] * 0.8)
    N_test  = N - N_train
    print('N:{}, N_train:{}, N_test:{}'.format(N, N_train, N_test))

    df_sample = df.sample(N)
    train = df_sample[:N_train]
    test  = df_sample[N_train:]

    return train, test


if __name__ == '__main__':
    print(create_train())
