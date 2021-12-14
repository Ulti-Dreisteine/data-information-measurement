# -*- coding: utf-8 -*-
"""
Created on 2021/12/14 20:39:33

@File -> knn_based.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 信息熵计算
"""

__doc__ = """
    本代码用于对一维和多维离散或连续变量数据的信息熵和互信息进行计算.
    连续变量信息熵使用Kraskov和Lombardi等人的方法计算, Lord等人文献可作为入门;离散变量信息熵则直接进行计算.

    求得信息熵和互信息后, 便可对条件熵进行求取:

    H(Y|X) = H(Y) - I(X;Y)

    此处不再补充条件熵的相关代码.

    参考文献
    ------
    1. W. M. Lord, J. Sun, E. M. Bolt: Geometric K-Nearest Neighbor Estimation of Entropy and Mutual information, 2018.
    2. A. Kraskov, H. Stoegbauer, P. Grassberger: Estimating Mutual Information, 2003.
    3. D. Lombardi, S. Pant: A Non-Parametric K-Nearest Neighbor Entropy Estimator, 2015.
    4. B. C. Ross: Mutual Information between Discrete and Continuous Data Sets, 2014.
    
    TODO
    ------  
    1. 检查对数的底数为e还是2
"""

from sklearn.neighbors import BallTree, KDTree
from scipy.special import digamma, gamma
from numpy import log
import numpy as np

BASE = np.e             # NOTE: 计算信息熵所采用的底数为e, 因为文献推导转为psi函数是以e为底数进行的
DTYPES = ['d', 'c']     # 数据类型, 离散或连续


# ---- 工具函数 -------------------------------------------------------------------------------------

# 数据处理.

def _convert_1d_series2int(x):
    """将一维数据按照标签进行编码为连续整数"""
    x = x.flatten()
    x_unique = np.unique(x)
    print('too many labels > 100 for the discrete data') if len(
        x_unique) > 100 else ...
    x = np.apply_along_axis(lambda x: np.where(
        x_unique == x)[0][0], 1, x.reshape(-1, 1))
    return x


def _convert_arr2int(arr):
    """将一维数据按照标签进行编码为连续整数"""
    _, D = arr.shape
    for d in range(D):
        arr[:, d] = _convert_1d_series2int(arr[:, d])
    return arr.astype(int)


def preprocess_values(x: np.ndarray, d_type: str, intens: float = 1e-10):
    x = x.copy()
    assert d_type in DTYPES

    # shape统一为(N, D).
    x = x.reshape(-1, 1) if len(x.shape) == 1 else x

    # 连续数据加入噪声.
    if d_type == 'd':
        x = _convert_arr2int(x)  # 检查数据类型并转化成对应值
    else:
        x += intens * np.random.random_sample(x.shape)  # 加入噪声避免数据退化.

    return x


# KNN计算参数.

def deter_k(x):
    """
    根据样本量确定近邻点数目

    参考文献
    ------ 
    J. B. Kinney, G. S. Atwal, Equitability, Mutual Information, and the Maximal Information Coefficient, 2014
    """
    if len(x) < 1000:
        k = 3
    elif (len(x) < 3000) & (len(x) >= 1000):
        k = 2
    else:
        k = 1
    return k


def build_tree(x):
    """建立近邻查询树. 低维用具有欧式距离特性的KDTree; 高维用具有更一般距离特性的BallTree"""
    if x.shape[1] <= 3:
        metric = 'euclidean'
    else:
        metric = 'minkowski'  # 'chebyshev'

    tree = BallTree(x, metric=metric) if x.shape[1] >= 20 else KDTree(
        x, metric=metric)
    return tree


def query_neighbors_dist(tree: BallTree or KDTree, x, k: int):
    """求得x样本在tree上的第k个近邻样本"""
    return tree.query(x, k=k + 1)[0][:, -1]


# ---- 信息熵 ---------------------------------------------------------------------------------------

class Entropy(object):
    """单变量(一维或多维)信息熵"""

    def __init__(self, x: np.array, d_type: str):
        self.x = preprocess_values(x, d_type)
        self.dtype = d_type

    def __call__(self, method: str = 'Kraskov', k: int = None):
        """计算信息熵. 参数method只有对连续变量起作用"""
        x = self.x

        if self.dtype == 'd':
            _, count = np.unique(x, return_counts=True, axis=0)
            proba = count.astype(float) / len(x)
            proba = proba[proba > 0.0]
            return np.sum(proba * np.log(1. / proba)) / log(BASE)
        else:
            """关于连续变量的计算, 以下分别按照Kraskov和Lombardi的方式实现"""
            k = deter_k(x) if k is None else k
            assert k <= len(x) - 1

            N, D = x.shape
            tree = build_tree(x)
            nn_distc = query_neighbors_dist(tree, x, k)  # 获得了各样本第k近邻的距离

            # Kraskov方法.
            if method == 'Kraskov':
                const = digamma(N) - digamma(k) + D * log(2)
                return (const + D * np.log(nn_distc).mean()) / log(BASE)

            # Lombardi方法.
            elif method == 'Lombardi':
                volume_unit_ball = (np.pi ** (.5 * D)) / gamma(.5 * D + 1)  # 高维空间内单位球的体积
                return (D * np.mean(np.log(nn_distc + np.finfo(x.dtype).eps)) + \
                    np.log(volume_unit_ball) + digamma(N) - digamma(k))

            else:
                raise ValueError


if __name__ == '__main__':
    pass
