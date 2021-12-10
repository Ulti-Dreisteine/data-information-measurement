# -*- coding: utf-8 -*-
"""
Created on 2021/12/10 20:27:50

@File -> entropy.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 数据信息熵计算
"""

from sklearn.neighbors import BallTree, KDTree
from scipy.special import digamma
from numpy import log
import numpy as np

__doc__ = """
    (补充参考文献)
"""


# ---- 工具函数 -------------------------------------------------------------------------------------

def add_noise(x, intens=1e-10):
    """加入噪声"""
    # small noise to break degeneracy, see doc.
    return x + intens * np.random.random_sample(x.shape)


def build_tree(x, metric: str = 'chebyshev'):
    """建立近邻查询树. 低维用KDTree; 高维用BallTree"""
    # NOTE: x.shape必须为(N, D)
    tree = BallTree(x, metric=metric) if x.shape[1] >= 20 else KDTree(
        x, metric=metric)
    return tree


def query_neighbors(tree, x, k: int):
    """求得x样本在tree上的前k个近邻样本"""
    return tree.query(x, k=k + 1)[0][:, k]


# ---- 信息熵计算 -----------------------------------------------------------------------------------

def cal_c_entropy(x, k=3, base=2):
    """连续数据信息熵, 使用Kraskov的KNN方法计算"""
    # TODO: 检查高维数据适应性
    assert k <= len(x) - 1
    N, D = x.shape
    x = add_noise(x)  # small noise to break degeneracy, see doc.
    tree = build_tree(x)
    nn = query_neighbors(tree, x, k)
    const = digamma(N) - digamma(k) + D * log(2)
    return (const + D * np.log(nn).mean()) / log(base)


def cal_d_entropy(x, base=2):
    """离散数据信息熵: 计算离散熵"""
    # TODO: 检查高维数据适应性
    _, count = np.unique(x, return_counts=True, axis=0)
    proba = count.astype(float) / len(x)
    proba = proba[proba > 0.0]
    return np.sum(proba * np.log(1. / proba)) / log(base)


def cal_cc_cond_entropy(x, y, **kwargs):
    """计算连续数据之间的条件熵 H(x|y) = H(x,y) - H(y)"""
    xy = np.hstack((x, y))
    entropy_xy = cal_c_entropy(xy, **kwargs)
    entropy_y = cal_c_entropy(y, **kwargs)
    return entropy_xy - entropy_y


def entropy(x, y=None, x_type: str = 'c', y_type: str = 'c', **kwargs):
    """变量的(条件)信息熵H(x)或H(x|y)计算"""
    # TODO: 封装以上三个方法
    if y is None:
        return cal_c_entropy(x, kwargs['k'], kwargs['base']) if x_type == 'c' else cal_d_entropy(x, kwargs['base'])
    else:
        if (x_type == 'c') & (y_type == 'c'):
            return cal_cc_cond_entropy(x, y, kwargs['k'], kwargs['base'])
        elif (x_type == 'd') & (y_type == 'd'):
            ...
        else:
            raise RuntimeError('not implemented')


# class Entropy(object):
#     """一维或多维连续或离散数据的信息熵计算"""

#     def __init__(self, x: np.ndarray, dtype: str):
#         try:
#             self.N, self.D = x.shape
#         except:
#             self.N, self.D = len(x), 1

#         self.x = x.reshape(self.N, self.D)  # 统一为(N, D)数组
#         self.dtype = dtype

#     def cal_entropy(self, base=2, **kwargs):
#         if 'int' in str(self.dtype):
#             return cal_d_entropy(self.x, base)
#         elif 'float' in str(self.dtype):
#             return cal_c_entropy(self.x, base, kwargs)
#         else:
#             raise RuntimeError('Unknown dtype %s' % self.dtype)


if __name__ == '__main__':
    x = np.array([1, 2, 3])
