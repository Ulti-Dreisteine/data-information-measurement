# -*- coding: utf-8 -*-
"""
Created on 2021/12/11 14:15:57

@File -> test_entropy.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 测试信息熵计算代码
"""

import numpy as np
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), '../' * 3))
sys.path.append(BASE_DIR)
print(sys.path[-1])

from mod.dataset.data_generator import DataGenerator
from core.entropy.entropy import Entropy

if __name__ == '__main__':
    
    # ---- 生成数据样本 -----------------------------------------------------------------------------
    
    N = 10000
    data_generator = DataGenerator(N_ticks=int(1e5))
    
    func = 'categorical'
    x1, y1, _, _ = data_generator.gen_data(N, func)
    x1_type, y1_type = 'd', 'd'
    
    func = 'parabola'
    x2, y2, _, _ = data_generator.gen_data(N, func)
    x2_type, y2_type = 'c', 'c'
    
    # plt.scatter(x2, y2)
    
    # ---- 计算信息熵 -------------------------------------------------------------------------------
    
    # 一维数组信息熵.
    print('x1, shape = {}, d_type = {}'.format(x1.shape, x1_type))
    print('y1, shape = {}, d_type = {}'.format(y1.shape, y1_type))
    print('x2, shape = {}, d_type = {}'.format(x2.shape, x2_type))
    print('y2, shape = {}, d_type = {}'.format(y2.shape, y2_type))
    
    print('\n单变量信息熵')
    print('H(x1)')
    print(Entropy(x1, d_type=x1_type)())
    
    print('H(y1)')
    print(Entropy(y1, d_type=y1_type)())
    
    print('H(x2) by Kraskov')
    print(Entropy(x2, d_type=x2_type)(method='knn', k=5))
    
    print('H(y2) by Kraskov')
    print(Entropy(y2, d_type=y2_type)(method='knn', k=5))
    
    
    
    
    
    