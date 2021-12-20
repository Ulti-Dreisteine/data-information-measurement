# -*- coding: utf-8 -*-
"""
Created on 2021/12/20 15:03:21

@File -> discrete_entropy.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 离散数据信息熵
"""

from numpy import log
import numpy as np

from . import BASE
from . import preprocess_values


class DiscreteEntropy(object):
    """离散变量的信息熵"""

    def __init__(self, x: np.ndarray):
        self.x = preprocess_values(x, d_type='d')
    
    def __call__(self):
        _, count = np.unique(self.x, return_counts=True, axis=0)
        proba = count.astype(float) / len(self.x)
        proba = proba[proba > 0.0]
        return np.sum(proba * np.log(1. / proba)) / log(BASE)