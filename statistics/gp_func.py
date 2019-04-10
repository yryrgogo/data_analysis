from matplotlib import pyplot as plt
import numpy as np
import math


class DataSet:
    
    def __init__(self, xmin, xmax, num_data, noise_level):
        
        self.xmin = xmin
        self.xmax = xmax
        self.x = (xmax - xmin) * np.random.rand(num_data) + xmin
        self.y = np.empty(num_data)
        
        for i in range(num_data):
            self.y[i] = self.make_y(self.x[i], noise_level)
            
        self.x, self.y = self.dual_sort(self.x, self.y)
        
        
    @staticmethod
    def make_y(x, noise_level):
        if x > 0:
            return 1.0/2.0 * x**(1.2) + math.cos(x) + np.random.normal(0, noise_level)
        else:
            return math.sin(x) + 1.1**x + np.random.normal(0, noise_level)
        
    @staticmethod
    def dual_sort(p, q):
        tmp = np.argsort(p)
        q = q[tmp]
        p = np.sort(p)
        
        return p, q
    
    def plot(self):
        plt.figure(figsize=(15, 10))
        plt.xlabel("x", fontsize=16)
        plt.ylabel("y", fontsize=16)
        plt.xlim(self.xmin, self.xmax)
        plt.tick_params(labelsize=16)
        plt.scatter(self.x, self.y)
        plt.show()
        
        
class GaussianProcess:
    
    def __init__(self, theta1, theta2, theta3):
        # thetax are used in rbf kernel elements
        self.theta1 = theta1
        self.theta2 = theta2
        self.theta3 = theta3
        self.xtrain = None
        self.ytrain = None
        self.xtest = None
        self.ytest = None
        self.kernel_inv = None # matrix
        self.mean_arr = None
        self.var_arr = None
        
        
    def rbf(self, p, q):
        return self.theta1 * math.exp(- (p-q)**2 / self.theta2)
    
    
    def train(self, data):
        """Summary line.
        trainではカーネル行列とその逆行列を求めてセットする
        1次元のxとyと受け取り、RBFカーネルでカーネル行列を作成
        クロネッカーのデルタは行番号=列番号の時に1をとり、ノイズを乗せる
        """
        self.xtrain = data.x
        self.ytrain = data.y
        N = len(self.ytrain)
        
        kernel = np.empty((N, N))
        for n1 in range(N):
            for n2 in range(N):
                kernel[n1, n2] = self.rbf(self.xtrain[n1], self.xtrain[n2]) + chr(n1, n2)*self.theta3
                
        # 逆行列を求める
        self.kernel_inv = np.linalg.inv(kernel)
        
        
    def test(self, test_data):
        """Summary line.
        testでは新しいデータ点xをセットし、
        num_train * num_testの部分カーネル行列
        num_test * num_testの部分カーネル行列
        を算出して新しい予測y*の平均と分散をそれぞれ求める
        """
        self.xtest = test_data.x
        
        N = len(self.xtrain)
        M = len(self.xtest)
        
        partial_kernel_train_test = np.empty((N, M))
        
        for m in range(M):
            for n in range(N)
                # 1列目から順に値を埋めていく感じ
                partial_kernel_train_test[n, m] = self.rbf(self.xtrain[n], self.xtest[m])
        
        partial_kernel_test_test = np.empty((M, M))
        
        for m1 in range(M):
            for m2 in range(M):
                partial_kernel_test_test[m1, m2] = self.rbf(self.xtest[m1], self.xtest[m2]) + chr(m1, m2)*self.theta3
                
        self.mean_arr = partial_kernel_train_test.T @ self.kernel_inv @ self.ytrain
        self.var_arr = partial_kernel_test_test - partial_kernel_train_test.T @ self.kernel_inv @ partial_kernel_train_test
        
        
def chr(a, b):
    if a == b:
        return 1
    else:
        return 0