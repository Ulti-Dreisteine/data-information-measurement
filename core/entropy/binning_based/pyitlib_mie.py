# -*- coding: utf-8 -*-
"""
Created on 2021/05/24 16:52:30

@File -> pyitlib_mie.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 直接使用pyitlib包计算信息熵
"""
from pyitlib import discrete_random_variable as drv
from sklearn.preprocessing import KBinsDiscretizer
from typing import Tuple
import numpy as np
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), '../../../../'))
sys.path.append(BASE_DIR)


def cal_mie(x: np.ndarray, y: np.ndarray, n_bins: int = 10):
    # 首先进行离散化.
    enc = KBinsDiscretizer(n_bins = n_bins, encode = 'ordinal', strategy = 'quantile')
    xs = enc.fit_transform(x.reshape(-1, 1)).astype(np.int)
    ys = enc.fit_transform(y.reshape(-1, 1)).astype(np.int)
    mie = drv.information_mutual(xs.flatten(), ys.flatten())
    return mie


def cal_rho(x: np.ndarray, y: np.ndarray, n_bins: int = 10):
    mie = cal_mie(x, y, n_bins)
    rho = np.sqrt(1 - np.power(2, -2 * mie))
    return rho


if __name__ == '__main__':
    from src.settings import *

    # ---- 载入数据 ---------------------------------------------------------------------------------

    def load_data(func: str, radius: float) -> Tuple[np.ndarray]:
        """载入数据
        """
        from core.dataset.data_generator import DataGenerator

        N = 5000
        data_gener = DataGenerator(N=N)
        x, y, _, _ = data_gener.gen_data(func, normalize=True)

        # 加入噪音.
        from mod.data_process.add_noise import add_circlular_noise

        x, y = add_circlular_noise(x, y, radius=radius)

        return x, y

    # ---- 生成数据 ---------------------------------------------------------------------------------

    from core.dataset.data_generator import FUNC_NAMES

    proj_plt.figure(figsize=[6, 6])
    for func in FUNC_NAMES[:1]:
        radius_lst = [10.0]  # np.arange(0.1, 10.0, 0.1)
        mie_lst = []
        for radius in radius_lst:
            x, y = load_data(func, radius)
            # mie = cal_mie(x, y)
            mie = cal_rho(x, y)
            mie_lst.append(mie)

        proj_plt.scatter(radius_lst, mie_lst, s=6)