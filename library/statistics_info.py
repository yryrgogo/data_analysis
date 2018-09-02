import pandas as pd
import numpy as np
import sys, re, glob
import gc
from load_data import pararell_load_data
from multiprocessing import Pool
import multiprocessing


def correlation(df):

    corr = df.corr(method='pearson')


    return corr


def correlation_selection():

    thres = 0.95

    ' そのfeatureと相関がxx以上のfeatureを取り出す '
    for feature in corr.columns:
        coef_ary = corr[feature].values
        coef_ary = coef_ary[coef_ary>thres]


