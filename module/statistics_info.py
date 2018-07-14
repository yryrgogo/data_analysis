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

