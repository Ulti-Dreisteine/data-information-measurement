# -*- coding: utf-8 -*-
"""
Created on 2021/02/12 18:40

@Project -> File: data-information-measure -> add_noise.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe:
"""

import numpy as np


def add_noise(x: np.ndarray, dtype: str, noise_coeff: float) -> np.ndarray:
    """
    生成数据样本
    :param x: 自变量序列
    :param dtype: x的数据类型
    :param noise_coeff: 噪音相对于标准差的系数, >= 0.0
    """
    x_cp = x.flatten()

    if dtype == 'numeric':
        x_cp = x_cp.astype(np.float32)
        N = len(x_cp)
        std_x = np.std(x_cp, axis=0).reshape(1, -1)
        noise_x = 2 * (np.random.random([N, 1]) - 0.5)  # 介于-1到1的随机数组
        x_cp += noise_coeff * \
            np.multiply(np.dot(np.ones([N, 1]), std_x), noise_x).flatten()
    else:
        pass

    return x_cp
