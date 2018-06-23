import pandas as pd
import numpy as np
import sys


' データをチェックする系 '
def col_heta_shape_cnt_check(data, col=False, heta=False, shape=False, count=False, nunique=False, all_check=False):

    if all_check:
        col=1
        part=1
        shape=1
        count=1
        nunique=1

    if col:
        for col in data.columns:
            print(col)

    if part:
        print(data.head())
        print(data.tail())

    if shape:
        print(data.shape)

    if count:
        print(data.count())

    if nunique:
        print(data.nunique())


def dframe_dtype(data):
    ' データフレームのデータ型をカラム毎に確認する '
    for col in data.columns:
        print(f'column: {col} dtype: {data[col].dtype}')


def list_check(check_list):
    ' listの中身を一覧で '
    for ele in check_list:
        print(ele)
    sys.exit()


def null_check(data):
    for col in data.columns:
        print(col)
        print(len(data[data[col].isnull()]))
    sys.exit()
