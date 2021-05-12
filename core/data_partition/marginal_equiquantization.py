# -*- coding: utf-8 -*-
"""
Created on 2021/05/12 16:45:21

@File -> marginal_equiquantization.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 数据边际等概率离散化
"""

from sklearn.preprocessing import MinMaxScaler
from typing import Tuple
import numpy as np
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(
    os.path.abspath(__file__), '../../../'))
sys.path.append(BASE_DIR)


# ---- 测试 -----------------------------------------------------------------------------------------

def load_data() -> Tuple[np.ndarray, np.ndarray]:
    """载入数据
    """
    from core.dataset.data_generator import DataGenerator

    N = 1001
    func = 'parabola'
    data_gener = DataGenerator(N=N)
    x, y, _, _ = data_gener.gen_data(func, normalize=True)

    # 加入噪音.
    from mod.data_process.add_noise import add_circlular_noise

    x, y = add_circlular_noise(x, y, radius=0.2)

    return x, y


if __name__ == '__main__':
    from src.settings import *

    # ---- 载入数据 ---------------------------------------------------------------------------------

    x, y = load_data()
    
    # ---- 预处理 -----------------------------------------------------------------------------------

    # 数据归一化.
    scaler = MinMaxScaler()
    x = scaler.fit_transform(x.reshape(-1, 1))
    y = scaler.fit_transform(y.reshape(-1, 1))
    # proj_plt.scatter(x, y, s=3)

    # ---- tmp -------------------------------------------------------------------------------------

    # 有序列表按照元素个数二等分.
    N = len(x)
    
    



