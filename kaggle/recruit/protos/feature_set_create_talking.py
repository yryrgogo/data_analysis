import gc
import numpy as np
import pandas as pd
from itertools import combinations
import datetime
from time import time
import sys, glob, re
from tqdm import tqdm
from multiprocessing import Pool
import multiprocessing
sys.path.append('../module')
from load_data import load_data, x_y_split, extract_set
from feature_base_agg_talking import one_particle_COUNTD, two_particle_COUNTD, three_particle_COUNTD, four_particle_COUNTD, \
    one_particle_time_COUNTD, two_particle_time_COUNTD, three_particle_time_COUNTD, four_particle_time_COUNTD,\
    one_particle_3_value_agg, two_particle_3_value_agg, three_particle_3_value_agg, four_particle_3_value_agg, \
    one_particle_3_value_time_agg, two_particle_3_value_time_agg, three_particle_3_value_time_agg, four_particle_3_value_time_agg

start_time = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())
#  input_path = '../input/20180420_sql_talk_5Mrow_noid_no_duplicates.csv'
input_path = '../input/speed.csv'
feature_path = '../features/*.csv'

data = pd.read_csv(input_path)

#  key_list = [col for col in data.columns if col.count('key')]
#  key_list.remove('key_dhm')
#  data.drop(key_list, axis=1, inplace=True)
#  print(data.columns)
#  data.to_csv('../input/20180404_sql_talk_5Mrow_noid_no_duplicates.csv', index=False)
#  sys.exit()

# main value*******************************
one_particle = ['o', 'd', 'a', 'c']
two_particle = list(combinations(one_particle, 2))
three_particle = list(combinations(one_particle, 3))
four_particle = list(combinations(one_particle, 4))

feature_elems = ['dl', 'dp', 'rc', 'V']
time_list = [
    'pts'
    ,'dhm'
    ,'dh'
    ,'hm'
    ,'h'
]

leak_time_list = ['pts', 'hm', 'h']


##multi processing###########################
def one_agg_wrapper(args): return one_particle_3_value_agg(*args)


def one_time_agg_wrapper(args): return one_particle_3_value_time_agg(*args)


def one_time_COUNTD_wrapper(args): return one_particle_time_COUNTD(*args)


def one_COUNTD_wrapper(args): return one_particle_COUNTD(*args)


def two_agg_wrapper(args): return two_particle_3_value_agg(*args)


def two_time_agg_wrapper(args): return two_particle_3_value_time_agg(*args)


def two_time_COUNTD_wrapper(args): return two_particle_time_COUNTD(*args)


def two_COUNTD_wrapper(args): return two_particle_COUNTD(*args)


def three_agg_wrapper(args): return three_particle_3_value_agg(*args)


def three_time_agg_wrapper(args): return three_particle_3_value_time_agg(*args)


def three_time_COUNTD_wrapper(args): return three_particle_time_COUNTD(*args)


def three_COUNTD_wrapper(args): return three_particle_COUNTD(*args)


def four_agg_wrapper(args): return four_particle_3_value_agg(*args)


def four_time_agg_wrapper(args): return four_particle_3_value_time_agg(*args)


def four_time_COUNTD_wrapper(args): return four_particle_time_COUNTD(*args)


def four_COUNTD_wrapper(args): return four_particle_COUNTD(*args)


def pararell_process(func, arg_list, t='notime', part_flg=1):
    global features

    p = Pool(multiprocessing.cpu_count())
    p_list = p.map(func, arg_list)
    p.close

    for d in p_list:
        key_list = [col for col in d.columns if not(col.count('@'))]
        feature_set = [col for col in d.columns if col.count('@')]

        if part_flg==0:
            tmp_result = features.merge(d, on=key_list, how='left')
            for col in feature_set:
                result = tmp_result[['key_dhm', col]].drop_duplicates()
                result = result[col]
                print(len(result))
                result.to_frame().to_csv('../features/{}_feature_{}.csv'.format(start_time[:11], result.name), index=False, header=True, compression='gzip')

        elif part_flg==1:
            features = features.merge(d, on=key_list, how='left')


def make_arg_list(particle_list, t=0, cnt_flg=0):
    global df, part_flg

    if part_flg == 0:
        val_1 = 0
    elif part_flg == 1:
        val_1 = 'dl'

    val_2 = 'rc'

    arg_list = []
    for ele in particle_list:
        if cnt_flg == 0 and t != 0:
            arg_list.append([df, ele, t, val_1, val_2, 'sum'])
            arg_list.append([df, ele, t, val_1, val_2, 'std'])

        elif cnt_flg == 0 and t == 0:
            arg_list.append([df, ele, val_1, val_2, 'sum'])
            arg_list.append([df, ele, val_1, val_2, 'std'])

        elif cnt_flg == 1 and t != 0:
            tmp_particle = one_particle.copy()
            if str(type(ele)).count('tuple'):
                for cat in ele:
                    tmp_particle.remove(cat)
            else:
                tmp_particle.remove(ele)
            for val in tmp_particle:
                arg_list.append([df, ele, t, val])

        elif cnt_flg == 1 and t == 0:
            tmp_particle = one_particle.copy()
            if str(type(ele)).count('tuple'):
                for cat in ele:
                    tmp_particle.remove(cat)
            else:
                tmp_particle.remove(ele)
            for val in tmp_particle:
                arg_list.append([df, ele, val])

    return arg_list


def base_time_agg(t, num, part_flg):
    global df, features

# one particle
    if num == 1:
        arg_list = make_arg_list(one_particle, t)
        pararell_process(one_time_agg_wrapper, arg_list, t, part_flg)

        ##count distinct##################
        arg_list = make_arg_list(one_particle, t, 1)
        pararell_process(one_time_COUNTD_wrapper, arg_list, t, part_flg)

# two particle
    elif num == 2:

        arg_list = make_arg_list(two_particle, t)
        pararell_process(two_time_agg_wrapper, arg_list, t, part_flg)

        ##count distinct##################
        arg_list = make_arg_list(two_particle, t, 1)
        pararell_process(two_time_COUNTD_wrapper, arg_list, t, part_flg)

# three particle
    elif num == 3:
        arg_list = make_arg_list(three_particle, t)
        pararell_process(three_time_agg_wrapper, arg_list, t, part_flg)

        ##count distinct##################
        arg_list = make_arg_list(three_particle, t, 1)
        pararell_process(three_time_COUNTD_wrapper, arg_list, t, part_flg)

# four particle
    elif num == 4:
        arg_list = make_arg_list(four_particle, t)
        pararell_process(four_time_agg_wrapper, arg_list, t, part_flg)


##no time features##########################################

def base_agg(num):
    global df, features

# one particle
    if num == 1:
        arg_list = make_arg_list(one_particle)
        pararell_process(one_agg_wrapper, arg_list)

        ##count distinct##################
        arg_list = make_arg_list(one_particle, 0, 1)
        pararell_process(one_COUNTD_wrapper, arg_list)

# two particle
    elif num == 2:

        arg_list = make_arg_list(two_particle)
        pararell_process(two_agg_wrapper, arg_list)
        #  for arg in arg_list:
        #      simple_process(two_particle_3_value_agg, arg)

        ##count distinct##################
        arg_list = make_arg_list(two_particle, 0, 1)
        pararell_process(two_COUNTD_wrapper, arg_list)
        #  for arg in arg_list:
        #      simple_process(two_particle_COUNTD, arg)

# three particle
    elif num == 3:
        arg_list = make_arg_list(three_particle)
        pararell_process(three_agg_wrapper, arg_list)

        ##count distinct##################
        arg_list = make_arg_list(three_particle, 0, 1)
        pararell_process(three_COUNTD_wrapper, arg_list)

# four particle
    elif num == 4:
        arg_list = make_arg_list(four_particle)
        pararell_process(four_agg_wrapper, arg_list)


# no partisionの場合、ptsにそのpartision noが入る
def split_features(tmp, ptc, time=0, pts=0):
    global nopts_dict

    df = tmp.drop(one_particle + ['dhm', 'dh', 'hm', 'h', 'val'] + ['dl', 'i_v', 'rc'], axis=1)
    df['key_dhm'] = df['key_dhm'].astype('category')
    data['key_dhm'] = data['key_dhm'].astype('category')

    for col in df.columns:
        if col.count('key') or col=='pts':continue

        elif time=='pts' and pts!=0:
            # noptsptsは、ptsあたりの集計になってるので、ptsあたり平均にまとめる
            tmp_agg = df.groupby('key', as_index=False)[col].mean()
            # noptsの場合、含まれていないpartisionに結果をjoinする
            tmp_df = data[data['pts']==pts][['key_dhm', 'key']]

            tmp_result = tmp_df.merge(tmp_agg, on='key', how='left')
            result = tmp_result[['key_dhm', col]]

        elif time!='pts' and pts!=0:
            # noptsの場合、含まれていないpartisionに結果をjoinする
            if time==0:
                key = 'key'
            else:
                key = 'key_{}'.format(time)

            tmp_df = data[data['pts']==pts][['key_dhm', key]]
            tmp_agg = df[[key, col]].drop_duplicates()

            tmp_result = tmp_df.merge(tmp_agg, on=key, how='left')
            result = tmp_result[['key_dhm', col]]

        else:
            print('ERROR!')
            print(tmp.head())
            print(time)
            print(pts)
            sys.exit()

        if pts==0:
            print(len(result))
            print('ERROR!')
            #  result.to_frame().to_csv('../features/{}_feature_{}.csv'.format(start_time[:11], result.name), index=False, header=True, compression='gzip')
        elif pts!=0:
            # 正規表現でcolからfeatures名を抽出する必要がある
            nopts_dict[col].append(result)


def create_key(data, time=0):

    print(time)
    if time != 0:
        data['key_{}'.format(time)] = str(data['o']) + '_' + str(data['d']) + '_' + str(data['a']) + '_' + str(data['c']) + '_' + str(data[time])
    elif time == 0:
        data['key'] = str(data['o']) + '_' + str(data['d']) + '_' + str(data['a']) + '_' + str(data['c'])

    return data


def path_info(path):

    path_dict = {}
    path_dict['filename'] = re.search(r'/([^/.]*).csv', path).group(1)  # Linux
    path_dict['particle'] = re.search(r'feature_([0-9]+)@', path).group(1)
    path_dict['time'] = re.search(r'@([^.]*)@', path).group(1)
    path_dict['elem'] = re.search(r'\D@([^.]*).csv', path).group(1)

    return path_dict


def dispersion_feature_set(data):
    global df, features, part_flg, nopts_dict
    part_flg = 0

    # timeありの集計
    for t in tqdm(time_list):
        # paricleを指定
        for p in tqdm(range(1,5,1)):

            df = data.copy()
            features = data.copy()
            base_time_agg(t, p, part_flg)

            del df, features
            gc.collect()

    # ここから先はvalidationを含めない部分で集計を行う
    part_flg = 1

    # paricleを指定
    # timeは指定せず、o,d,a,cの粒度でのみ学習を行う
    for p in tqdm(range(1,5,1)):
        for pts in range(1,4,1):
            df = data[data['pts'] != pts].copy()
            features = df.copy()
            base_agg(p)

            # ここでfeature名がわかるので、それをkeyにまとめて辞書を作る
            if pts==1:
                nopts_dict = {}
                for ele in features.columns:
                    if ele.count('@'):
                        nopts_dict[ele] = []

            split_features(features, p, 0, pts)

            del df, features
            gc.collect()

        for col, pts_list in nopts_dict.items():
            tmp_result = pd.concat(pts_list, axis=0)
            tmp_result.rename(columns={col:col+'_avg'}, inplace=True)
            tmp_result.sort_values(by='key_dhm', inplace=True)
            result = tmp_result[col+'_avg']
            result.fillna(0, inplace=True)

            print(len(result))
            result.to_frame().to_csv('../features/{}_feature_{}_notime.csv'.format(start_time[:11], result.name), index=False, header=True, compression='gzip')

    #  sys.exit()

    # 一部のparticle, validationを除いて特徴量を生成する
    for t in tqdm(leak_time_list):
        # paricleを指定
        for p in tqdm(range(3, 5, 1)):

            for pts in range(1, 4, 1):

                if t != 'val':
                    df = data[data['pts'] != pts].copy()
                else:
                    df = data[data['val'] != 0]
                features = df.copy()
                base_time_agg(t, p, part_flg)

                # ここでfeature名がわかるので、それをkeyにまとめて辞書を作る
                if pts==1:
                    nopts_dict = {}
                    for ele in features.columns:
                        if ele.count('@'):
                            nopts_dict[ele] = []

                split_features(features, p, t, pts)

                del df, features
                gc.collect()

            for col, pts_list in nopts_dict.items():
                tmp_result = pd.concat(pts_list, axis=0)
                tmp_result.rename(columns={col:col+'_avg'}, inplace=True)
                tmp_result.sort_values(by='key_dhm', inplace=True)
                result = tmp_result[col+'_avg']
                result.fillna(0, inplace=True)

                print(len(result))
                result.to_frame().to_csv('../features/{}_feature_{}_leak.csv'.format(start_time[:11], result.name), index=False, header=True, compression='gzip')


def main():
    global data

    #  tmp_time = time_list.copy() + [0]
    #  tmp_time.remove('dhm')
    #  for t in tmp_time:
    #      data = create_key(data, t)
    #  data.to_csv('../input/20180420_sql_talk_5Mrow_noid_no_duplicates.csv', index=False)
    #  print(data.columns)
    #  sys.exit()

    dispersion_feature_set(data)


if __name__ == '__main__':

    main()
