# -*- coding: utf-8 -*-
"""
Created on 2021/12/20 12:18:21

@File -> __init__.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 初始化
"""

__all__ = ['unit_ball_volume', 'preprocess_values', 'deter_k', 'build_tree', 'query_neighbors_dist']

from sklearn.neighbors import BallTree, KDTree
from scipy.special import gamma
import numpy as np

BASE = np.e             # NOTE: 计算信息熵所采用的底数为e, 因为文献推导转为psi函数是以e为底数进行的
DTYPES = ['d', 'c']     # 数据类型, 离散或连续


# 空间计算.

def unit_ball_volume(d: int, metric: str = 'euclidean'):
    """d维空间中按照euclidean或chebyshev距离计算所得的单位球体积"""
    if metric == 'euclidean':
        # see: https://en.wikipedia.org/wiki/Volume_of_an_n-ball.
        return (np.pi ** (d / 2)) / gamma(1 + d / 2)  
    elif metric == 'chebyshev':
        return 1
    else:
        raise ValueError('unsupported metric "%s"' % metric)


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


def build_tree(x, metric: str = 'chebyshev'):
    """建立近邻查询树. 低维用具有欧式距离特性的KDTree; 高维用具有更一般距离特性的BallTree"""
    tree = BallTree(x, metric=metric) if x.shape[1] >= 20 else \
        KDTree(x, metric=metric)
    return tree


def query_neighbors_dist(tree: BallTree or KDTree, x, k: int):
    """求得x样本在tree上的第k个近邻样本"""
    return tree.query(x, k=k + 1)[0][:, -1]
