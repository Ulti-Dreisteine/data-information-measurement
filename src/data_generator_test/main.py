# -*- coding: utf-8 -*-
"""
Created on 2021/02/12 18:38

@Project -> File: data-information-measure -> main.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 数据生成器测试
"""

import sys
import os

BASE_DIR = os.path.abspath(os.path.join(
    os.path.abspath(__file__), '../../../'))
sys.path.append(BASE_DIR)

from src.settings import *
from core.dataset.data_generator import FUNC_NAMES, DataGenerator


if __name__ == '__main__':

    # ---- 测试代码 ---------------------------------------------------------------------------------

    N = 2000
    func = FUNC_NAMES[22]
    print('func = "{}"'.format(func))

    data_gener = DataGenerator(N)
    x, y, _, _ = data_gener.gen_data(func)

    proj_plt.scatter(x, y)
