# -*- coding: utf-8 -*-
"""
Created on 2021/12/20 17:12:59

@File -> _mi_dd.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 计算离散变量之间的互信息
"""

import numpy as np

from . import preprocess_values
from ..entropy.entropy import Entropy


class MutualInfoDD(object):
    """连续X和离散Y之间的互信息计算"""

    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x, self.y = preprocess_values(x, d_type='d'), preprocess_values(y, d_type='d')
        self.N = len(x)

    def __call__(self):
        x, y = self.x, self.y
        xy = np.c_[x, y]
        return Entropy(x, 'd')() + Entropy(y, 'd')() - Entropy(xy, 'd')()

