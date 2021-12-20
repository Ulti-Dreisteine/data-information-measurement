# -*- coding: utf-8 -*-
"""
Created on 2021/12/20 17:12:59

@File -> _mi_cd.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 计算连续和离散变量之间的互信息
"""

from scipy.special import psi
import numpy as np

from . import preprocess_values, deter_k, build_tree, query_neighbors_dist


class MutualInfoCD(object):
    """连续X和离散Y之间的互信息计算"""

    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x, self.y = preprocess_values(x, d_type='c'), preprocess_values(y, d_type='d')
        self.N = len(x)

    def __call__(self, method: str = 'ross', **kwargs):
        x, y = self.x, self.y

        if method == 'ross':
            N, _ = x.shape
            assert (x.shape[1] >= 1) & (y.shape[1] == 1)  # NOTE: 此处y必须为1维
            y = y.flatten()

            k = deter_k(x) if 'k' not in kwargs else kwargs['k']
            assert k <= N - 1

            # 统计各类Y的总数.
            classes = np.unique(y)
            Nx_class = np.zeros_like(y)
            for i in range(len(y)):
                Nx_class[i] = np.sum(y == y[i])

            # 逐类进行K近邻计算.
            nn_distc_classes = np.zeros_like(y, dtype=float)
            for c in classes:
                mask = np.where(y == c)[0]
                tree = build_tree(x[mask, :], 'chebyshev')
                nn_distc_classes[mask] = query_neighbors_dist(
                    tree, x[mask, :], k)  # 获得了各样本第k近邻的距离

            # 所有样本中的K近邻计算.
            tree = build_tree(x, 'chebyshev')
            m = tree.query_radius(x, nn_distc_classes)
            m = [p.shape[0] for p in m]
            return psi(N) - np.mean(psi(Nx_class)) + psi(k) - np.mean(psi(m))
        else:
            raise ValueError('the method is not implemented')


        

