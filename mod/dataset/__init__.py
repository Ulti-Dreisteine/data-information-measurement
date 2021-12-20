# -*- coding: utf-8 -*-
"""
Created on 2021/03/18 11:15

@Project -> File: refined-association-process-causality-analysis -> __init__.py.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 初始化
"""

import numpy as np

PI = np.pi


def linear_periodic_low_freq(x: np.ndarray):
    return 0.2 * np.sin(4 * (2 * x - 1)) + 1.1 * (2 * x - 1)


def linear_periodic_med_freq(x: np.ndarray):
    return np.sin(10 * PI * x) + x


def linear_periodic_high_freq(x: np.ndarray):
    return 0.1 * np.sin(10.6 * (2 * x - 1)) + 1.1 * (2 * x - 1)


def linear_periodic_high_freq_2(x: np.ndarray):
    return 0.2 * np.sin(10.6 * (2 * x - 1)) + 1.1 * (2 * x - 1)


def non_fourier_freq_cos(x: np.ndarray):
    return np.cos(7 * PI * x)


def cos_high_freq(x: np.ndarray):
    return np.cos(14 * PI * x)


def cubic(x: np.ndarray):
    return 4 * np.power(x, 3) + np.power(x, 2) - 4 * x


def cubic_y_stretched(x: np.ndarray):
    return 41 * (4 * np.power(x, 3) + np.power(x, 2) - 4 * x)


def l_shaped(x: np.ndarray):
    y = np.apply_along_axis(lambda p: p / 99 if p <= 0.99 else 99 * p - 98, 1, x.reshape(-1, 1))
    y = y.flatten()
    return y


def exp_base_2(x: np.ndarray):
    return np.power(2, x)


def exp_base_10(x: np.ndarray):
    return np.power(10, x)


def line(x: np.ndarray):
    return x


def parabola(x: np.ndarray):
    return 4 * np.power(x, 2)


def random(x: np.ndarray):
    return np.random.uniform(0.0, 1.0, len(x))


def non_fourier_freq_sin(x: np.ndarray):
    return np.sin(9 * PI * x)


def sin_low_freq(x: np.ndarray):
    return np.sin(8 * PI * x)


def sin_high_freq(x: np.ndarray):
    return np.sin(16 * PI * x)


def sigmoid(x: np.ndarray):
    def _f(x):
        if x <= 0.49:
            return 0.0
        elif (x >= 0.49) & (x <= 0.51):
            return 50.0 * (x - 0.5) + 0.5
        else:
            return 1.0
    
    y = np.apply_along_axis(_f, 1, x.reshape(-1, 1))
    y = y.astype(np.float64).flatten()
    return y


def vary_freq_cos(x: np.ndarray):
    return np.cos(5 * PI * x * (1 + x))


def vary_freq_sin(x: np.ndarray):
    return np.sin(6 * PI * x * (1 + x))


def spike(x: np.ndarray):
    def _f(x: float):
        if x < 0.05:
            return 20 * x
        elif (x >= 0.05) & (x < 0.1):
            return -18 * x + 1.9
        else:
            return -x / 9 + 1 / 9
    
    y = np.apply_along_axis(_f, 1, x.reshape(-1, 1))
    y = y.flatten()
    return y


def lopsided_l_shaped(x: np.ndarray):
    def _f(x: float):
        if x < 0.005:
            return 200 * x
        elif (x >= 0.005) & (x < 0.01):
            return -198 * x + 1.99
        else:
            return -x / 99 + 1 / 99
    
    y = np.apply_along_axis(_f, 1, x.reshape(-1, 1))
    y = y.flatten()
    return y


def categorical(x: np.ndarray):
    def _f(x: float):
        if x == 1:
            return 0.287
        elif x == 2:
            return 0.796
        elif x == 3:
            return 0.290
        elif x == 4:
            return 0.924
        elif x == 5:
            return 0.717
        else:
            raise RuntimeError('')
    
    y = np.apply_along_axis(_f, 1, x.reshape(-1, 1))
    y = y.flatten()
    return y
