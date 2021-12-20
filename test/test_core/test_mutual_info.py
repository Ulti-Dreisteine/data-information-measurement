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

from src.setting import plt
from mod.dataset.data_generator import DataGenerator
from core.mutual_info.mutual_info import MutualInfo
from core.mutual_info.cond_mutual_info import CondMutualInfo

if __name__ == '__main__':
    
    # ---- 生成数据样本 -----------------------------------------------------------------------------
    
    N = 1000
    data_generator = DataGenerator(N_ticks=int(1e5))
    
    func = 'categorical'
    x1, y1, _, _ = data_generator.gen_data(N, func)
    x1_type, y1_type = 'd', 'd'
    
    func = 'cos_high_freq'
    x2, y2, _, _ = data_generator.gen_data(N, func)
    x2_type, y2_type = 'c', 'c'
    
    # plt.scatter(x2, y2)
    
    # ---- 计算信息熵 -------------------------------------------------------------------------------
    
    # 变量信息熵.
    print('x1, shape = {}, d_type = {}'.format(x1.shape, x1_type))
    print('y1, shape = {}, d_type = {}'.format(y1.shape, y1_type))
    print('x2, shape = {}, d_type = {}'.format(x2.shape, x2_type))
    print('y2, shape = {}, d_type = {}'.format(y2.shape, y2_type))
    
    print('\n互信息')
    print('I(X1;Y1)')
    print(MutualInfo(x1, y1, x1_type, y1_type)())
    
    print('\nI(X2;Y2)')
    print(MutualInfo(x2, y2, x2_type, y2_type)())
    
    print('\nI(X1;Y2)')
    print(MutualInfo(x1, y2, x1_type, y2_type)(k=3))
    
    print('\nI(X2;Y1)')
    print(MutualInfo(x2, y1, x2_type, y1_type)(k=3))
    
    print('\n条件互信息')
    print('I(X1;Y1|X1)')
    print(CondMutualInfo(x1, y1, x1, x1_type, y1_type, x1_type)())
    
    print('\nI(X2;Y2|X2) by mutual info')
    print(CondMutualInfo(x2, y2, x2, x2_type, y2_type, x2_type)(method='mutual_info', k=5))  # TODO: 连续变量条件熵计算结果有系统误差

    print('\nI(X2;Y2|X2) by binning z')
    print(CondMutualInfo(x2, y2, x2, x2_type, y2_type, x2_type)(method='binning_z', n=200, k=3))
    
    print('\nI(X2;Y2|X2) by Kraskov')
    print(CondMutualInfo(x2, y2, x2, x2_type, y2_type, x2_type)(method='kraskov', k=3))
    
    print('\nI(X1;Y2|X2) by Mutual_Info')
    print(CondMutualInfo(x1, y2, x2, x1_type, y2_type, x2_type)(method='mutual_info', k=5))
    
    
    
    
    
    
    