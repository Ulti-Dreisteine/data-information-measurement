# -*- coding: utf-8 -*-
"""
Created on 2021/03/18 11:16

@Project -> File: refined-association-process-causality-analysis -> data_generator.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 初始化
"""

import numpy as np
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), '../' * 3))
sys.path.append(BASE_DIR)

from . import *

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
    'l_shaped',  # association != 1.0
    'exp_base_2',
    'exp_base_10',
    'line',
    'parabola',
    'random',  # association != 1.0
    'non_fourier_freq_sin',
    'sin_low_freq',
    'sin_high_freq',
    'sigmoid',
    'vary_freq_cos',
    'vary_freq_sin',
    'spike',
    'lopsided_l_shaped',  # association hard = 1
    
    # 离散值.
    'categorical'  # 非连续值
]


def _min_max_norm(x: np.ndarray):
    x = x.copy()
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def _random_sampling(arr: np.ndarray, keep_n: int, seed: int = 0):
    arr = arr.copy()
    
    # 获取数据参数.
    N = arr.shape[0]
    idxes = list(range(N))
    
    # 随机采样.
    np.random.seed(seed)
    idxes = np.random.permutation(idxes)
    arr = arr[idxes[: keep_n]][:]
    return arr


class DataGenerator(object):
    """
    数据生成器

    Reference:
    1. D.N. Reshef, Y.A. Reshef, et al.: "Supporting Online Material for Detecting Novel Associations
            in Large Data Sets" (Table S3), Science, 2012.
    """
    
    def __init__(self, N_ticks: int=int(1e4)):
        self.N_ticks = N_ticks  # ! 该值太大会影响计算效率
        self.func_groups = {
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
        
    
    def _init_x_ticks(self, func: str) -> np.ndarray:
        if func in self.func_groups[0]:
            h = 1.0 / self.N_ticks
            return np.arange(0.0 - h, 1.0 + 2 * h, h)  # 这里多一个h是为了抵消后面差分h变少
        elif func in self.func_groups[1]:
            h = 2.4 / self.N_ticks
            return np.arange(-1.3 - h, 1.1 + 2 * h, h)
        elif func in self.func_groups[2]:
            h = 10.0 / self.N_ticks
            return np.arange(0.0 - h, 10.0 + 2 * h, h)
        elif func in self.func_groups[3]:
            h = 1.0 / self.N_ticks
            return np.arange(-0.5 - h, 0.5 + 2 * h, h)
        elif func in self.func_groups[4]:
            return np.random.randint(1, 6, self.N_ticks, dtype = int)  # 随机生成1~5的随机整数
        else:
            raise RuntimeError('Invalid func = "{}"'.format(func))
    
    def gen_data(self, N: int, func: str, normalize: bool = False):
        """这里对数据进行了采样"""
        if N > self.N_ticks:
            raise ValueError('self.N_ticks < N, 减少N或增加self.N_ticks')
        
        x_ticks = self._init_x_ticks(func)
        
        # try:
        y_ticks = eval('{}'.format(func))(x_ticks)
        if func in self.func_groups[4]:
            arr = _random_sampling(np.vstack((x_ticks, y_ticks)).T, N)
            x, y = arr[:, 0], arr[:, 1]
        else:
            # 计算梯度, 并换算为概率密度.
            y_derivs_l = y_ticks[1: -1] - y_ticks[:-2]     # 向右差分
            y_derivs_r = y_ticks[2:] - y_ticks[1: -1]     # 向左差分
            p_derivs = np.abs((y_derivs_l + y_derivs_r) / 2)
            p_derivs = p_derivs / np.sum(p_derivs)
            
            # 进行采样.
            x_ticks_s = x_ticks.copy()[1:-1]
            y_ticks_s = y_ticks.copy()[1:-1]
            
            x = np.array([x_ticks_s[0], x_ticks_s[-1]])
            y = np.array([y_ticks_s[0], y_ticks_s[-1]])
            
            x_ = np.random.choice(x_ticks_s, size=N-2, replace=True, p=list(p_derivs))
            y_ = eval('{}'.format(func))(x_)
            
            x = np.hstack((x, x_))
            y = np.hstack((y, y_))
            
        if normalize:
            x_norm = _min_max_norm(x)
            y_norm = _min_max_norm(y)
            return x, y, x_norm, y_norm
        else:
            return x, y, None, None
        # except Exception:
        #     raise ValueError('Invalid func = "{}"'.format(func))
        

# if __name__ == '__main__':
#     import sys
#     import os
    
#     BASE_DIR = os.path.abspath(os.path.join(os.getcwd(), '../' * 2))
#     sys.path.append(BASE_DIR)
    
#     from src.setting import plt
    
#     N = 500
#     self = DataGenerator()
    
#     func = 'exp_base_2'
#     x, y, _, _ = self.gen_data(N, func)
    
#     plt.figure()
#     plt.scatter(x, y, s = 12)
    
    
