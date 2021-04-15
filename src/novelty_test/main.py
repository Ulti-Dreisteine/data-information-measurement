# -*- coding: utf-8 -*-
"""
Created on 2021/02/12 20:04

@Project -> File: data-information-measure -> main.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: novelty测试代码
"""

# from sklearn.preprocessing import minmax_scale
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np
import seaborn as sns
import random as rd
import math
import sys
import os

sys.path.append('../../')

from src import *
from src.novelty_test import FUNCS2TEST
from core.dataset import *
from core.dataset.data_generator import DataGenerator
from core.entropy.binning_based.mic import MutualInfoCoeff
from core.entropy.knn_based.classical import KLEstimator
from mod.data_process.time_stamp_serial import DataTimeStampSerialization
from mod.data_process.normalize import min_max_norm

NORMALIZE = True


# ---- 样本生成 -------------------------------------------------------------------------------------


def gen_samples(N: int, func: str):
	data_gener = DataGenerator(N = N)
	x, y, x_norm, y_norm = data_gener.gen_data(func, normalize = NORMALIZE)
	return x, y, x_norm, y_norm


# ---- 加入噪音 -------------------------------------------------------------------------------------


def add_circle_noise(x: np.ndarray, y: np.ndarray, radius: float = None, radius_ratio: float = None):
	"""以样本点为圆心, 以随机半径在圆内均匀分布采样"""
	arr = np.vstack((x.copy(), y.copy())).T  # arr.shape = (-1, 2)
	
	rad = radius if radius is not None else (np.max(x) - np.min(x)) * radius_ratio
	
	def _uniform_sampling(x: np.ndarray):
		theta = rd.uniform(0, 2 * PI)
		r = rd.uniform(0, rad)
		return x[0] + r * math.cos(theta), x[1] + r * math.sin(theta)
	
	arr_noise = np.apply_along_axis(_uniform_sampling, 1, arr)
	
	return arr_noise[:, 0], arr_noise[:, 1]


# ---- Entropy计算 ----------------------------------------------------------------------------------


def mic(x: np.ndarray, y: np.ndarray) -> float:
	mutual_info_coeff = MutualInfoCoeff(x, y)
	entropy = mutual_info_coeff.cal_mic()
	return entropy


def kl_knn(x: np.ndarray, y: np.ndarray, k: int = 5):
	kl_estimator = KLEstimator(x, y)
	entropy = kl_estimator.cal_mi_cc(k = k)
	return entropy

	
if __name__ == '__main__':
	
	# ---- 生成数据样本 ------------------------------------------------------------------------------
	
	N = 1000
	novelty_test = None
	method = 'kl_knn'
	
	# ---- Novelty测试 ------------------------------------------------------------------------------
	
	for func in FUNCS2TEST:
		print('testing "{}"'.format(func))
		x, y, x_norm, y_norm = gen_samples(N, func)

		# y_true = eval(func)(x)
		# if NORMALIZE:
		# 	y_true = min_max_norm(y_true)

		# ---- 计算Novelty测试曲线 -------------------------------------------------------------------

		entropy_results = []
		for radius in np.arange(0.0, 2.0 + 0.025, 0.025):
			x_noise, y_noise = add_circle_noise(x_norm, y_norm, radius = radius + EPS)

			entropy = eval(method)(x_noise, y_noise)
			# entropy_results.append([1 - r2_score(y_norm, y_noise), entropy])
			entropy_results.append([radius, entropy])

		# 整理结果.
		entropy_results = pd.DataFrame(entropy_results, columns = ['time', '{}'.format(func)])
		dts = DataTimeStampSerialization(entropy_results, start_stp = 0.0, end_stp = 2.0, stp_step = 0.025)
		entropy_results, _ = dts.serialize_time_stamps()

		if novelty_test is None:
			novelty_test = entropy_results
		else:
			novelty_test = pd.concat([novelty_test, entropy_results[['{}'.format(func)]]], axis = 1)

	# 保存结果.
	novelty_test.to_csv(
		os.path.join(proj_dir, 'file/novelty_test/{}_novelty_test.csv'.format(method)),
		index = False
	)