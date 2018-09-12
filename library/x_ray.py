import numpy as np
import pandas as pd
import lightgbm as lgb
import datetime
from tqdm import tqdm
import sys
sys.path.append('../library')
import xgboost as xgb
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, ridge
from convinience_function import pararell_process


def x_ray_caliculation(valid, col, value, model):
    print(col)
    valid[col] = value
    pred = model.predict(valid)
    tmp = np.mean(pred)
    return col, value, tmp

def x_ray_wrapper(args):
    return x_ray_caliculation(*args)

def x_ray(model, valid, columns=False):

    x_ray = False
    arg_list = []
    if not(columns):
        columns = valid.columns
    for col in columns:
        xray_list = []
        value_ser = valid[col].drop_duplicates()
        val_cnt = value_ser.value_counts()
        if len(val_cnt)>100:
            length = 100
        else:
            length = len(val_cnt)
        df_valid = val_cnt.head(length).reset_index()
        print(df_valid.shape)
        print(df_valid)
        sys.exit()

        for value in value_list:
            arg_list.append([df_valid, col, value, model])
    x_array = pararell_process(x_ray_wrapper, arg_list)
    x_ray = pd.DataFrame(x_array)
    #  tmp_xray = valid[col].to_frame('value')
    #  tmp_xray['feature'] = col
    #  tmp_xray['x_ray'] = xray_list

    #  if x_ray:
    #      x_ray = pd.concat([x_ray, tmp_xray], axis=0)
    #  else:
    #      x_ray = tmp_xray.copy()

    return x_ray
