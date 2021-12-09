# -*- coding: utf-8 -*-
"""
Created on 2021/12/09 21:19:39

@File -> setting.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 项目设置
"""

from sklearn.ensemble import RandomForestRegressor
from typing import Tuple
import pandas as pd
import numpy as np
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), '../' * 2))
sys.path.append(BASE_DIR)

from mod.config.config_loader import config_loader

PROJ_CMAP = config_loader.proj_cmap
plt = config_loader.proj_plt

# 载入项目变量配置.
ENC_CONFIG = config_loader.environ_config
MODEL_CONFIG = config_loader.model_config
TEST_PARAMS = config_loader.test_params

# ---- 定义环境变量 ---------------------------------------------------------------------------------

# ---- 定义模型参数 ---------------------------------------------------------------------------------

# ---- 定义测试参数 ---------------------------------------------------------------------------------

# ---- 定义通用函数 ---------------------------------------------------------------------------------
