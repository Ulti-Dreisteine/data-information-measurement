# -*- coding: utf-8 -*-
"""
Created on 2021/12/20 16:47:24

@File -> mi_cc.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 连续变量之间的互信息
"""

from sklearn.preprocessing import MinMaxScaler
from scipy.special import psi
import numpy as np

from . import preprocess_values, deter_k, build_tree, query_neighbors_dist
from ..partition.marginal_equiquant import Cell, recursively_partition


def _minmax_norm(arr: np.ndarray):
    D = arr.shape[1]
    scaler = MinMaxScaler()
    arr_norm = None
    for i in range(D):
        a = scaler.fit_transform(arr[:, i: i + 1])

        if arr_norm is None:
            arr_norm = a
        else:
            arr_norm = np.hstack((arr_norm, a))
    return arr_norm


class MutualInfoCC(object):
    """连续变量之间的互信息计算"""

    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x, self.y = preprocess_values(x, d_type='c'), preprocess_values(y, d_type='c')
        self.N = len(x)

    def __call__(self, method: str = 'kraskov', **kwargs):
        x, y = self.x, self.y

        if method == 'kraskov':
            k = deter_k(x) if 'k' not in kwargs else kwargs['k']
            N, _ = x.shape
            assert k <= len(x) - 1

            xy = np.c_[x, y]
            tree = build_tree(xy, 'chebyshev')
            nn_distc = query_neighbors_dist(tree, xy, k)  # 获得了各样本第k近邻的距离

            tree = build_tree(x, 'chebyshev')
            nn_distc_x = nn_distc - 1e-12
            Nx = tree.query_radius(x, nn_distc_x, count_only=True)

            tree = build_tree(y, 'chebyshev')
            nn_distc_y = nn_distc - 1e-12
            Ny = tree.query_radius(y, nn_distc_y, count_only=True)

            # 根据Kraskov文献报道, 第二种结果更好.
            # return psi(N) + psi(k) - np.mean(psi(Nx) + psi(Ny))
            return psi(k) - 1 / k + psi(N) - np.mean(psi(Nx) + psi(Ny))

        elif method == 'lombardi':
            # TODO: implement.
            raise ValueError('the method is not implemented')

        # TODO: 递归算法有BUG.
        elif method == 'darbellay':
            assert (x.shape[1] == 1) & (y.shape[1] == 1)  # 该方法目前只支持两个一维数据之间的计算

            # 数据预处理.
            arr = np.c_[x, y]
            arr = _minmax_norm(arr)

            # 空间离散化.
            cell = Cell(arr)
            cell.def_cell_bounds([(0.0, 1.0), (0.0, 1.0)])
            leaf_cells = recursively_partition(cell, min_samples_split=30, p_eps=1e-3)
            leaf_cells = [c for c in leaf_cells if c.N > 0]

            # 计算互信息.
            n_leafs = len(leaf_cells)

            mi = 0.0
            for i in range(n_leafs):
                cell = self.leaf_cells[i]  # type: Cell
                (xl, xu), (yl, yu) = cell.bounds

                Nxy = len(cell.arr)
                Nx = len(
                    np.where((self.arr[:, 0] >= xl) & (self.arr[:, 0] < xu))[0])
                Ny = len(
                    np.where((self.arr[:, 1] >= yl) & (self.arr[:, 1] < yu))[0])
                gain = Nxy * np.log(Nxy / Nx / Ny)
                mi += gain

            mi = mi / self.N + np.log(self.N)
            return mi
