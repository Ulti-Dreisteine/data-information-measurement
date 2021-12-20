# -*- coding: utf-8 -*-
"""
Created on 2021/12/20 15:02:34

@File -> knn_entropy.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 基于K近邻估计的信息熵
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
"""

from scipy.special import psi
from numpy import log
import numpy as np

from . import BASE
from . import unit_ball_volume, preprocess_values, deter_k, build_tree, query_neighbors_dist


class KNeighborsEntropy(object):
    """单变量(一维或多维)信息熵"""

    def __init__(self, x: np.array):
        self.x = preprocess_values(x, d_type='c')

    def __call__(self, method: str = 'kraskov', k: int = None):
        """计算信息熵. 参数method只有对连续变量起作用"""
        x = self.x
        k = deter_k(x) if k is None else k
        assert k <= len(x) - 1

        N, D = x.shape

        # 构建距离树.
        tree = None
        if method == 'kraskov':
            tree = build_tree(x, 'chebyshev')
        elif method == 'lombardi':
            tree = ...
        else:
            pass

        # 计算结果.
        if method == 'kraskov':
            # 参见Kraskov, et al. Estimating Mutual Information, 2004. Eq. (20).
            nn_distc = query_neighbors_dist(tree, x, k)  # 获得了各样本第k近邻的距离
            v = unit_ball_volume(D, metric='chebyshev')
            return (-psi(k) + psi(N) + np.log(v) + D * np.log(nn_distc).mean()) / log(BASE)
        elif method == 'lombardi':
            # TODO: Implement
            raise ValueError('the method is not supported yet')
        else:
            raise ValueError
