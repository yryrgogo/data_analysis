import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_log_error
from itertools import combinations
import datetime
import sys
import pickle
import plsa_core
from plsa_kmeans import plsa_kmeans
#  , plsa_main, kmeans_main

start_time = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())

output_path = '../output/' + start_time + '_ncaa_submit.csv'

# load_data********************************
sample = pd.read_csv('/mnt/c/Git/go/kaggle/ncaa/input/SampleSubmissionStage1.csv')
predict = pd.read_csv('../submit/0308/0308_submit1_20180308_083107_0.30276311852899207_2014_2017_ncaa_submit_pred.csv')
#  print(sample.head(1))

submit = predict.apply(lambda x: pd.Series([str(int(x.season)) + '_' + str(int(x.teamid)) + '_' + str(int(x.teamid_2)), x.prediction]), axis=1).rename(columns={0:'ID', 1:'Pred'})

submit.to_csv(output_path, index=False)
