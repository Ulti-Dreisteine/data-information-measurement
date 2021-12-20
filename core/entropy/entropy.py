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
"""

import numpy as np

from . import DTYPES
from ._discrete_entropy import DiscreteEntropy
from ._knn_entropy import KNeighborsEntropy
from ._partition_entropy import PartitionEntropy


class Entropy(DiscreteEntropy, PartitionEntropy, KNeighborsEntropy):
    """单个一维或多维变量的信息熵"""

    def __init__(self, x: np.ndarray, d_type: str):
        assert d_type in DTYPES
        self.x = x 
        self.d_type = d_type

    def __call__(self, method: str = 'knn', **kwargs):
        # 对于多继承问题, 使用super(Entropy, self)的可能有问题, 会把Entropy所有父类都执行一遍. 因此下面分别
        # 根据不同父类具体实例化后计算.
        if self.d_type == 'd':
            return DiscreteEntropy(self.x).__call__(**kwargs)  
        else:
            if method == 'knn':
                return KNeighborsEntropy(self.x).__call__(**kwargs)
            elif method == 'partition':
                return PartitionEntropy(self.x).__call__(**kwargs)
            else:
                raise ValueError('method not supported')
