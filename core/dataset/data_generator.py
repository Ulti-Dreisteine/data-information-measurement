# -*- coding: utf-8 -*-
"""
Created on 2021/02/10 14:04

@Project -> File: dataset-information-measure -> data_generator.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 数据生成器
"""

# from sklearn.preprocessing import minmax_scale
import numpy as np
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(
    os.path.abspath(__file__), '../../../'))
sys.path.append(BASE_DIR)

from core.dataset import *
from mod.data_process.normalize import min_max_norm

FUNC_NAMES = [
    # 连续值.
    'linear_periodic_low_freq',
    'linear_periodic_med_freq',
    'linear_periodic_high_freq',
    'linear_periodic_high_freq_2',
    'non_fourier_freq_cos',
    'cos_high_freq',
    'cubic',
    'cubic_y_stretched',
    'l_shaped',  # MIC != 1.0
    'exp_base_2',
    'exp_base_10',
    'line',
    'parabola',
    'random',  # MIC != 1.0
    'non_fourier_freq_sin',
    'sin_low_freq',
    'sin_high_freq',
    'sigmoid',
    'vary_freq_cos',
    'vary_freq_sin',
    'spike',
    'lopsided_l_shaped',  # MIC hard = 1

    # 离散值.
    'categorical'  # 非连续值
]


class DataGenerator(object):
    """
    数据生成器

    Reference:
    1. D.N. Reshef, Y.A. Reshef, et al.: "Supporting Online Material for Detecting Novel Associations
            in Large Data Sets" (Table S3), Science, 2012.
    """

    def __init__(self, N: int):
        self.N = N

    def _init_x(self, func: str) -> np.ndarray:
        func_groups = {
            0: [
                'linear_periodic_low_freq', 'linear_periodic_med_freq', 'linear_periodic_high_freq',
                'linear_periodic_high_freq_2', 'non_fourier_freq_cos', 'cos_high_freq', 'l_shaped',
                'line', 'random', 'non_fourier_freq_sin', 'sin_low_freq', 'sin_high_freq', 'sigmoid',
                        'vary_freq_cos', 'vary_freq_sin', 'spike', 'lopsided_l_shaped'
            ],
            1: ['cubic', 'cubic_y_stretched'],
            2: ['exp_base_2', 'exp_base_10'],
            3: ['parabola'],
            4: ['categorical']
        }

        if func in func_groups[0]:
            return np.random.uniform(0.0, 1.0, self.N)
        elif func in func_groups[1]:
            return np.random.uniform(-1.3, 1.1, self.N)
        elif func in func_groups[2]:
            return np.random.uniform(0.0, 10.0, self.N)
        elif func in func_groups[3]:
            return np.random.uniform(-0.5, 0.5, self.N)
        elif func in func_groups[4]:
            return np.random.randint(1, 6, self.N, dtype=int)  # 随机生成1~5的随机整数
        else:
            raise RuntimeError('Invalid func = "{}"'.format(func))

    def gen_data(self, func: str, normalize: bool = False):
        x = self._init_x(func).astype(np.float32)

        try:
            y = eval('{}'.format(func))(x).astype(np.float32)

            if normalize:
                x_norm = min_max_norm(x)
                y_norm = min_max_norm(y)
                return x, y, x_norm, y_norm
            else:
                return x, y, None, None
        except Exception as e:
            raise ValueError('Invalid func = "{}"'.format(func))


if __name__ == '__main__':

    from src.settings import *

    # ---- 测试代码 ----------------------------------------------------------------------------------

    self = DataGenerator(N=1000)

    func = 'linear_periodic_low_freq'
    x, y, x_norm, y_norm = self.gen_data(func, normalize=True)
    proj_plt.scatter(x, y)
