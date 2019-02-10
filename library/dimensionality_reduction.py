"""
@package
@brief
@author stfate
"""

import scipy as sp
from scipy.sparse.csgraph import connected_components
import sklearn.base
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import bhtsne
import umap

#  TSNE
#  from MulticoreTSNE import MulticoreTSNE as TSNE
import numpy as np
import pandas as pd
import time, datetime
import sys, re

sys.path.append('../../../github/module/')
from utils import get_categorical_features, get_numeric_features


start_time = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())
#  logger = logger_func()

unique_id = 'SK_ID_CURR'
target = 'TARGET'

ignore_features = [unique_id, target, 'valid_no', 'is_train', 'is_test']



class BHTSNE(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    def __init__(self, dimensions=2, perplexity=30.0, theta=0.5, rand_seed=-1):
        self.dimensions = dimensions
        self.perplexity = perplexity
        self.theta = theta
        self.rand_seed = rand_seed


    def fit_transform(self, X):
        return bhtsne.tsne(
            X.astype(sp.float64),
            dimensions=self.dimensions,
            perplexity=self.perplexity,
            theta=self.theta,
            rand_seed=self.rand_seed,
        )


def go_bhtsne(logger, data, D):

    params = {'dimensions':D, 'perplexity':30.0, 'theta':0.5, 'rand_seed':1208}
    start_time = time.time()
    logger.info(f't_SNE train start: {start_time}')

    # t-SNE
    #  bhtsne = BHTSNE(dimensions=2, perplexity=30.0, theta=0.5, rand_seed=-1, max_iter=10000)
    bhtsne = BHTSNE(**params)
    embedding = bhtsne.fit_transform(data)

    logger.info(f't_SNE train end: {time.time() - start_time}')

    return embedding


def kmeans(df, cluster=10):

    print(f'\nKMEANS INPUT DATA:\n{df.head()}')
    seed = 1208

    params = {'n_clusters':cluster,
              'n_init' : 10,
              'max_iter' : 300,
              'tol' : 1e-4,
              'precompute_distances' : 'auto',
              'verbose' : 0,
              'random_state' : seed,
              'copy_x' : True,
              'n_jobs' : -1,
              'algorithm' : 'auto'
              }

    kmeans = KMeans(**params).fit(df)

    df['cluster'] = kmeans.labels_

    return df


def t_SNE(logger, data, D):

    params = {'n_jobs':-1, 'n_components':D}
    start_time = time.time()
    logger.info(f't_SNE train start: {start_time}')
    # t-SNE
    tsne_model = TSNE(**params)
    embedding = tsne_model.fit_transform(data)

    logger.info(f't_SNE train end: {time.time() - start_time}')

    return embedding


def UMAP(logger, data, D):

    params = {'n_components':D}
    start_time = time.time()
    logger.info(f'UMAP train starttime: {start_time}')
    # UMAP
    embedding = umap.UMAP(**params).fit_transform(data)

    logger.info(f'UMAP train end. caliculation time: {time.time() - start_time}')

    return embedding


def main():
    pass


if __name__ == "__main__":
    main()
