# -*- coding: utf-8 -*-
"""
Created on 2021/12/20 17:55:21

@File -> cond_mutual_info.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 条件互信息
"""

from scipy.special import psi   
import pandas as pd
import numpy as np

from . import DTYPES, build_tree
from . import preprocess_values, build_tree, deter_k, query_neighbors_dist
from .mutual_info import MutualInfo

class CondMutualInfo(object):

    def __init__(self, x, y, z, x_type, y_type, z_type):
        assert len(x) == len(y)
        assert len(x) == len(z)
        for type_ in [x_type, y_type, z_type]:
            assert type_ in DTYPES

        self.x = preprocess_values(x, x_type)
        self.y = preprocess_values(y, y_type)
        self.z = preprocess_values(z, z_type)
        self.x_type, self.y_type, self.z_type = x_type, y_type, z_type

    def __call__(self, method: str = 'binning_z', n: int = None, **kwargs):
        # TODO: 根据数据类型自行匹配最优算法.
        x, y, z = self.x, self.y, self.z
        
        if method == 'mutual_info':
            # 利用互信息减法, 通用于任意数据类型之间的计算.
            if self.y_type == self.z_type:
                # I(X;Y|Z) = I(X;Y,Z) - I(X;Z).
                yz = np.c_[y, z]
                return MutualInfo(x, yz, self.x_type, self.y_type)(**kwargs) - \
                    MutualInfo(x, z, self.x_type, self.z_type)(**kwargs)
            else:
                raise ValueError

        elif method == 'binning_z':
            # 利用离散Z上结果加和, 通用于任意数据类型之间的计算.
            n = 100 if n is None else n
            if self.z_type == 'd':
                z_enc = z 
            else:
                z_enc = pd.qcut(
                    z.flatten(), int(len(z) // n), labels=False, duplicates='drop').reshape(-1, 1)
            z_labels = set(np.unique(z_enc))
            arr = np.c_[x, y, z_enc]

            if len(z_labels) < 5:
                print('the number of discretized labels is too low')

            cmi = 0.0
            for label in z_labels:
                arr_sub = arr[np.where(arr[:, 2] == label)[0], :-1]
                prob = arr_sub.shape[0] / arr.shape[0]
                x_sub, y_sub = arr_sub[:, 0], arr_sub[:, 1]
                mi_sub = MutualInfo(x_sub, y_sub, self.x_type, self.y_type)(**kwargs)
                cmi += prob * mi_sub

            return cmi

        elif method == 'kraskov':
            # 利用Kraskov算法, 仅适用于连续变量.
            assert self.x_type == 'c'
            assert self.y_type == 'c'
            assert self.z_type == 'c'
            
            k = deter_k(x) if 'k' not in kwargs.keys() is None else kwargs['k']
            assert k <= len(x) - 1

            xz = np.c_[x, z]
            yz = np.c_[y, z]
            xyz = np.c_[x, y, z]
            
            tree = build_tree(xyz, 'chebyshev')
            nn_distc = query_neighbors_dist(tree, xyz, k)  # 获得了各样本第k近邻的距离
            
            tree = build_tree(xz, 'chebyshev')
            nn_distc_xz = nn_distc - 1e-15
            Nxz = tree.query_radius(xz, nn_distc_xz, count_only=True)

            tree = build_tree(yz, 'chebyshev')
            nn_distc_yz = nn_distc - 1e-15
            Nyz = tree.query_radius(yz, nn_distc_yz, count_only=True)
            
            tree = build_tree(z, 'chebyshev')
            nn_distc_z = nn_distc - 1e-15
            Nz = tree.query_radius(z, nn_distc_z, count_only=True)

            return np.mean(psi(Nz)) + psi(k) - np.mean(psi(Nxz) + psi(Nyz))
        else:
            raise ValueError
