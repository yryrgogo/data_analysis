#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@package
@brief
@author stfate
"""
import pandas as pd
import scipy as sp
import sklearn.base
import bhtsne
import sys
import numpy as np
from sklearn.manifold import TSNE


class BHTSNE(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    def __init__(self, dimensions=2, perplexity=30.0, theta=0.5, rand_seed=-1):
        self.dimensions = dimensions
        self.perplexity = perplexity
        self.theta = theta
        self.rand_seed = rand_seed
        #  self.max_iter = max_iter

    def fit_transform(self, X):
        return bhtsne.tsne(
            X.astype(sp.float64),
            dimensions=self.dimensions,
            perplexity=self.perplexity,
            theta=self.theta,
            rand_seed=1208
            #  ,max_iter=self.max_iter
        )


def t_SNE(data, dimensions=2, perplexity=30.0, theta=0.5, rand_seed=1208, max_iter=10000):

    bhtsne = BHTSNE(dimensions=dimensions, perplexity=perplexity, theta=theta, rand_seed=rand_seed)

    tsne = bhtsne.fit_transform(data)

    print(tsne)
    print(tsne.shape)
    return tsne
