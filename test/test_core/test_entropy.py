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

# BASE_DIR = os.path.abspath(os.path.join(os.getcwd(), '../' * 2))
BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), '../' * 3))
sys.path.append(BASE_DIR)
print(sys.path[-1])

from src.setting import plt
from core.mi.entropy import cal_entropy, cal_mutual_info, cal_cond_entropy, cal_cond_mutual_info
from core.dataset.data_generator import DataGenerator

if __name__ == '__main__':
    
    # ---- 生成数据样本 -----------------------------------------------------------------------------
    
    N = 100000
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
    print('x1, shape = {}, x_type = {}'.format(x1.shape, x1_type))
    print('y1, shape = {}, x_type = {}'.format(y1.shape, y1_type))
    print('x2, shape = {}, x_type = {}'.format(x2.shape, x2_type))
    print('y2, shape = {}, x_type = {}'.format(y2.shape, y2_type))
    
    print('\n单变量信息熵')
    print('H(x1)')
    print(cal_entropy(x1, x_type=x1_type))
    
    print('H(y1)')
    print(cal_entropy(y1, x_type=y1_type))
    
    print('H(x2)')
    print(cal_entropy(x2, x_type=x2_type))
    
    print('H(y2)')
    print(cal_entropy(y2, x_type=y2_type))
    
    # 一维数据之间的条件熵.
    print('\n两个变量之间的条件熵')
    print('H(x1|y1):')
    print(cal_cond_entropy(x1, y1, x1_type, y1_type))
    
    print('H(x2|y2):')
    print(cal_cond_entropy(x2, y2, x2_type, y2_type))
    
    print('H(x2|y1):')
    print(cal_cond_entropy(x2, y1, x2_type, y1_type))
    
    # ---- 计算互信息 -------------------------------------------------------------------------------
    
    print('\n两个变量之间的互信息')
    print('I(x1;y1):')
    print(cal_mutual_info(x1, y1, x1_type, y1_type))
    
    print('I(x2;y2):')
    print(cal_mutual_info(x2, y2, x2_type, y2_type))
    
    # 满足交换律.
    print('-' * 10)
    print('I(x2;y1):')
    print(cal_mutual_info(x2, y1, x2_type, y1_type))
    
    print('I(y1;x2):')
    print(cal_mutual_info(y1, x2, y1_type, x2_type))
    
    print('-' * 10)
    print('I(x1;y2):')
    print(cal_mutual_info(x1, y2, x1_type, y2_type))
    
    print('I(y2;x1):')
    print(cal_mutual_info(y2, x1, y2_type, x1_type))
    
    
    
    
    
    