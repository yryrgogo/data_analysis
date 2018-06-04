# http://qiita.com/HZama/items/561cb240620991d3a0e2

import pandas as pd
import numpy as np


class PLSA(object):
    def __init__(self, N, Z):
        self.N = N
        self.X = N.shape[0]
        self.Y = N.shape[1]
        self.Z = Z

        # P(z)
        self.Pz = np.random.rand(self.Z)
        # P(x|z)
        self.Px_z = np.random.rand(self.Z, self.X)
        # P(y|z)
        self.Py_z = np.random.rand(self.Z, self.Y)

        # 正規化
        self.Pz /= np.sum(self.Pz)
        self.Px_z /= np.sum(self.Px_z, axis=1)[:, None]
        self.Py_z /= np.sum(self.Py_z, axis=1)[:, None]

    def train(self, k=200, t=1.0e-7):
        '''
        対数尤度が収束するまでEステップとMステップを繰り返す
        '''
        prev_llh = 100000
        for i in range(k):
            self.em_algorithm()
            llh = self.llh()

            if abs((llh - prev_llh) / prev_llh) < t:
                break

            prev_llh = llh

    def em_algorithm(self):
        '''
        EMアルゴリズム
        P(z), P(x|z), P(y|z)の更新
        '''
        tmp = self.N / np.einsum('k,ki,kj->ij', self.Pz, self.Px_z, self.Py_z)
        tmp[np.isnan(tmp)] = 0
        tmp[np.isinf(tmp)] = 0

        Pz = np.einsum('ij,k,ki,kj->k', tmp, self.Pz, self.Px_z, self.Py_z)
        Px_z = np.einsum('ij,k,ki,kj->ki', tmp, self.Pz, self.Px_z, self.Py_z)
        Py_z = np.einsum('ij,k,ki,kj->kj', tmp, self.Pz, self.Px_z, self.Py_z)

        self.Pz = Pz / np.sum(Pz)
        self.Px_z = Px_z / np.sum(Px_z, axis=1)[:, None]
        self.Py_z = Py_z / np.sum(Py_z, axis=1)[:, None]

    def llh(self):
        '''
        対数尤度
        '''
        Pxy = np.einsum('k,ki,kj->ij', self.Pz, self.Px_z, self.Py_z)
        Pxy /= np.sum(Pxy)
        lPxy = np.log(Pxy)
        lPxy[np.isinf(lPxy)] = -1000

        return np.sum(self.N * lPxy)
