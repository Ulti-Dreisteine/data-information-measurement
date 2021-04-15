# -*- coding: utf-8 -*-
"""
Created on 2021/02/12 21:51

@Project -> File: data-information-measure -> normalize.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe:
"""

import numpy as np


def min_max_norm(x: np.ndarray):
    x = x.copy()
    return (x - np.min(x)) / (np.max(x) - np.min(x))
