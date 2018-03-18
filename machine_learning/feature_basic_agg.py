import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import datetime
import sys
sys.path.append('../module')
sys.path.append('../clustering')
from load_data import load_data, x_y_split, extract_set
import plsa_core
from plsa_kmeans import plsa_kmeans

start_time = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())
input_path = '../input/*.csv'
#  delimiter = '\t'

# main value*******************************
ID = 'teamid'
key_list = []

# load_data********************************
data, fs_name = load_data(input_path, key_list)
#  print(data.head())
sys.exit()


# preprocessing****************************

cluster = plsa_kmeans(data, ID, 8, 10)
#  print(cluster.head())
#  sys.exit()

def standard(data, value):
    val = data[value].values
    mean = val.mean()
    std = val.std()
    for v in val:
        if ((v-mean)/std)>1.96:
            print('outlier:{}'.format(v))
    return (val - mean)/std


def label_encord(data, value, fe_name):
    label = LabelEncoder()
    data[fe_name] = label.fit_transform(feature_set[value])
    return data


def stats_select_range(data, start, end, range_col, value):
    df_range = data[(data[range_col] >= start) & (data[range_col] < end)].drop_duplicates()

    result = df_range.groupby([ID], as_index=False)[value].agg(
        {'{}_sum_{}_{}'.format(value, start, end): 'sum',
         '{}_avg_{}_{}'.format(value, start, end): 'mean',
         '{}_max_{}_{}'.format(value, start, end): 'max',
         '{}_min_{}_{}'.format(value, start, end): 'min',
         '{}_var_{}_{}'.format(value, start, end): 'var',
         '{}_median_{}_{}'.format(value, start, end): 'median'
         })

    result.fillna(-1, inplace=True)  # varは成分が一つだとNaNになるため
    return result

    #  # range分のdataをもたない場合、-1とする
    #  result.set_index(ID, inplace=True)
    #  columns = result.columns
    #  exclude = 
    #  result.loc[exclude, columns] = -1
    #  return result.reset_index()


# 重み付き平均
def weight_avg_all(data, start, end, range_col, weight, value):
    df_range = data[(data[range_col] >= start) & (data[range_col] < end)].drop_duplicates()
    df_range['num'] = np.abs(df_range[range_col] - end + 1)
    df_range['weight'] = df_range['num'].map(lambda x: ratio**x)
    df_range[value] = df_range[value]*df_range['weight']
    result = df_range.groupby([ID], as_index=False)[value, 'weight'].sum()
    result['{}_wmean'.format(value)] = result[value] / result['weight']
    result.drop([value, 'weight'], axis=1, inplace=True)
    return result


