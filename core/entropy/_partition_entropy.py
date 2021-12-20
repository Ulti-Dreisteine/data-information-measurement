# -*- coding: utf-8 -*-
"""
Created on 2021/12/20 15:08:12

@File -> partition_entropy.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 基于离散分箱的信息熵
"""

__doc__ = """
    参考文献:
    ------   
    1. G. A. Darbellay, I. Vajda: Estimation of the Information by an Adaptive Partitioning of the Observation Space, 1999.
"""

import numpy as np

from . import preprocess_values


class PartitionEntropy(object):
    """基于分箱的信息熵"""

    def __init__(self, x: np.ndarray):
        self.x = preprocess_values(x, d_type='c')

    def __call__(self, method: str = 'darbellay'):
        # TODO: implement.
        raise RuntimeError('the method is not supported')
