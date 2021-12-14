# -*- coding: utf-8 -*-
"""
Created on 2021/12/14 21:10:08

@File -> mutual_info.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 互信息和条件互信息计算
"""

from scipy.special import digamma
import numpy as np

from .entropy import DTYPES
from .entropy import preprocess_values, deter_k, build_tree, query_neighbors_dist, Entropy


class MutualInfo(object):
    """
    两个随机变量之间的互信息计算

    1. X和Y均为'd': 直接离散计算互信息H(X) + H(Y) - H(X, Y)
    2. X和Y均为'c': 采用Kraskov的方案
    3. X和Y分别为'c'和'd': 采用Ross方案
    4. X和Y分别为'd'和'c': 采用Ross方案
    """

    def __init__(self, x: np.ndarray, y: np.ndarray, x_type: str, y_type: str):
        assert len(x) == len(y)
        for type_ in [x_type, y_type]:
            assert type_ in DTYPES
        self.x = preprocess_values(x, x_type)
        self.y = preprocess_values(y, y_type)
        self.x_type, self.y_type = x_type, y_type

    @staticmethod
    def _mi_cc(x, y, k: int = None):
        """计算连续变量之间的互信息

        参考文献: 
        ------
        A. Kraskov, H. Stoegbauer, P. Grassberger: Estimating Mutual Information, 2003.
        """
        N, _ = x.shape
        k = deter_k(x) if k is None else k
        assert k <= len(x) - 1

        xy = np.c_[x, y]
        tree = build_tree(xy)
        nn_distc = query_neighbors_dist(tree, xy, k)  # 获得了各样本第k近邻的距离

        tree = build_tree(x)
        nn_distc_x = nn_distc - 1e-15
        Nx = tree.query_radius(x, nn_distc_x, count_only=True)

        tree = build_tree(y)
        nn_distc_y = nn_distc - 1e-15
        Ny = tree.query_radius(y, nn_distc_y, count_only=True)

        return digamma(N) + digamma(k) - np.mean(digamma(Nx)) - np.mean(digamma(Ny))

    @staticmethod
    def _mi_cd(x, y, k: int = None):
        """
        计算连续的X和离散的Y之间的互信息

        参考文献: 
        ------
        B. C. Ross: Mutual Information between Discrete and Continuous Data Sets, 2014.
        """
        N, _ = x.shape
        assert (x.shape[1] >= 1) & (y.shape[1] == 1)  # NOTE: 此处y必须为1维
        y = y.flatten()

        k = deter_k(x) if k is None else k
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
            tree = build_tree(x[mask, :])
            nn_distc_classes[mask] = query_neighbors_dist(
                tree, x[mask, :], k)  # 获得了各样本第k近邻的距离

        # 所有样本中的K近邻计算.
        tree = build_tree(x)
        m = tree.query_radius(x, nn_distc_classes)
        m = [p.shape[0] for p in m]
        return digamma(N) - np.mean(digamma(Nx_class)) + digamma(k) - np.mean(digamma(m))

    def __call__(self, **kwargs):
        """计算互信息"""
        x, y = self.x, self.y

        if (self.x_type == 'd') & (self.y_type == 'd'):
            xy = np.c_[x, y]
            return (sum([Entropy(v, 'd')() for v in [x, y]]) - Entropy(xy, 'd')())
        elif (self.x_type == 'c') & (self.y_type == 'c'):
            return self._mi_cc(x, y, **kwargs)
        elif (self.x_type == 'c') & (self.y_type == 'd'):
            return self._mi_cd(x, y, **kwargs)
        elif (self.x_type == 'd') & (self.y_type == 'c'):
            """计算离散X和连续Y之间的互信息, 利用互信息的交换律"""
            assert (x.shape[1] == 1) & (y.shape[1] >= 1)  # NOTE: 此处x必须为1维
            return self._mi_cd(y, x, **kwargs)
        else:
            pass


class CondMutualInfo(object):
    """条件互信息"""

    def __init__(self, x, y, z, x_type, y_type, z_type):
        assert len(x) == len(y)
        assert len(x) == len(z)
        for type_ in [x_type, y_type, z_type]:
            assert type_ in DTYPES
        self.x = preprocess_values(x, x_type)
        self.y = preprocess_values(y, y_type)
        self.z = preprocess_values(z, z_type)
        self.x_type, self.y_type, self.z_type = x_type, y_type, z_type
        
    def __call__(self, method: str = 'mutual_info', **kwargs):
        # TODO: 根据数据类型自行匹配最优算法.
        x, y, z = self.x, self.y, self.z
        
        # 利用互信息减法.
        if method == 'mutual_info':
            if self.y_type == self.z_type:
                """I(X;Y|Z) = I(X;Y,Z) - I(X;Z)"""
                yz = np.c_[y, z]
                return MutualInfo(x, yz, self.x_type, self.y_type)(**kwargs) - \
                    MutualInfo(x, z, self.x_type, self.z_type)(**kwargs)
            else:
                raise ValueError
        elif method == 'binning':
            ...  # TODO: 完成这部分
            raise ValueError
        elif method == 'kraskov':
            assert self.x_type == 'c'
            assert self.y_type == 'c'
            assert self.z_type == 'c'
            
            k = deter_k(x) if 'k' not in kwargs.keys() is None else kwargs['k']
            assert k <= len(x) - 1

            xz = np.c_[x, z]
            yz = np.c_[y, z]
            xyz = np.c_[x, y, z]
            
            tree = build_tree(xyz)
            nn_distc = query_neighbors_dist(tree, xyz, k)  # 获得了各样本第k近邻的距离
            
            tree = build_tree(xz)
            nn_distc_xz = nn_distc - 1e-15
            Nxz = tree.query_radius(xz, nn_distc_xz, count_only=True)

            tree = build_tree(yz)
            nn_distc_yz = nn_distc - 1e-15
            Nyz = tree.query_radius(yz, nn_distc_yz, count_only=True)
            
            tree = build_tree(z)
            nn_distc_z = nn_distc - 1e-15
            Nz = tree.query_radius(z, nn_distc_z, count_only=True)

            return np.mean(digamma(Nz)) + digamma(k) - np.mean(digamma(Nxz)) \
                - np.mean(digamma(Nyz))
        else:
            raise ValueError
        
        
        


if __name__ == '__main__':
    pass
