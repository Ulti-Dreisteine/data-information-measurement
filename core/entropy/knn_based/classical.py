# -*- coding: utf-8 -*-
"""
Created on 2021/02/11 15:51

@Project -> File: data-information-measure -> classical.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 基于KNN的经典信息熵估计器
"""

from sklearn.neighbors import NearestNeighbors
from scipy.special import gamma, psi
import numpy as np


def _nearest_distances(x, k: int, metric: str):
    """Returns the distance to the kth nearest neighbor for every point in X"""
    knn = NearestNeighbors(n_neighbors=k, metric=metric)
    knn.fit(x)
    d, _ = knn.kneighbors(x)
    return d[:, -1]


def _entropy(x: np.ndarray, k: int, metric: str):
    """
    根据X的样本计算对应的信息熵

    Reference:
    1. Kozachenko, Leonenko:
    """
    try:
        N, D = x.shape
    except:
        raise ValueError('X应为二维数组')
    r = _nearest_distances(x, k, metric)
    volume_unit_ball = (np.pi ** (.5 * D)) / gamma(.5 * D + 1)  # 高维空间单位球体积
    return D * np.mean(np.log(r + np.finfo(x.dtype).eps)) + np.log(volume_unit_ball) + psi(N) - psi(k)


class KLEstimator(object):
    """
    基于KNN的经典信息熵估计器, 又名KL估计器

    Reference:
    1. D. Lombardi, S. Pant: "A Non-Parametric K-Nearest Neighbor Entropy Estimator", Physical Review E., 2016
    """

    # TODO: 分别针对x为连续型或离散型进行计算.
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x, self.y = x.astype(
            np.float32).flatten(), y.astype(np.float32).flatten()

    def cal_mi_cc(self, k: int, metric: str = 'chebyshev'):
        x, y = self.x.reshape(-1, 1), self.y.reshape(-1, 1)
        vars = np.hstack((x, y))
        return _entropy(x, k, metric) + _entropy(y, k, metric) - _entropy(vars, k, metric)


if __name__ == '__main__':

    import pandas as pd
    import sys
    import os

    BASE_DIR = os.path.abspath(os.path.join(
        os.path.abspath(__file__), '../../../../'))
    sys.path.append(BASE_DIR)

    from src.settings import *
    from core.dataset.data_generator import DataGenerator

    # ---- 测试代码 --------------------------------------------------------------------------------

    from core.dataset.data_generator import FUNC_NAMES

    N = 5000
    entropy_results = []
    for func in FUNC_NAMES:
        # 生成样本.
        data_gener = DataGenerator(N=N)
        x, y, _, _ = data_gener.gen_data(func)

        # 计算MIC.
        kl_estimator = KLEstimator(x, y)
        mic = kl_estimator.cal_mi_cc(k=10)

        entropy_results.append([func, mic])

    # 整理结果.
    entropy_results = pd.DataFrame(
        entropy_results, columns=['func', 'entropy'])
