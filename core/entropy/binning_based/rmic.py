# -*- coding: utf-8 -*-
"""
Created on 2021/02/10 16:17

@Project -> File: data-information-measure -> rmic.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 改进的MIC算法
"""

from minepy import MINE
import numpy as np
import sys

sys.path.append('../../../')

from core.entropy.binning_based._univar_encoding import UnsuperCategorEncoding, SuperCategorEncoding

X_TYPES = ['numeric', 'categoric']


class RefinedMutualInfoCoeff(object):
	"""改进的互信息系数"""
	
	def __init__(self, x: np.ndarray, y: np.ndarray, x_type: str, mic_params: dict = None):
		assert x_type in X_TYPES
		self.x, self.y = x.astype(np.float32).flatten(), y.astype(np.float32).flatten()
		self.x_type = x_type
		self.mic_params = mic_params
		
	def cal_mic(self, encod_method: str = None):
		x = self.x.copy()
		if self.x_type == 'categoric':
			if encod_method in ['label', 'random', 'freq']:
				unsuper_encoding = UnsuperCategorEncoding(x)
				x = unsuper_encoding.encode(method = encod_method)
			elif encod_method in ['mhg']:
				super_encoding = SuperCategorEncoding(x, self.y)
				x = super_encoding.encode()
			else:
				raise RuntimeError('Invalid encod_method = "{}"'.format(encod_method))
		else:
			pass
		
		if self.mic_params is not None:
			mine = MINE(**self.mic_params)
		else:
			mine = MINE()
			
		mine.compute_score(x, self.y)
		return mine.mic()
	

if __name__ == '__main__':
	
	import pandas as pd
	import sys
	
	sys.path.append('../../../')
	
	from src import *
	from core.dataset.data_generator import DataGenerator
	
	# ---- 测试代码 ----------------------------------------------------------------------------------
	
	from core.dataset.data_generator import FUNC_NAMES
	
	N = 1000
	entropy_results = []
	for func in FUNC_NAMES:
		# 生成样本.
		data_gener = DataGenerator(N = N)
		x, y = data_gener.gen_data(func)
		
		# 计算MIC.
		if func == 'categorical':
			x_type = 'categoric'
			encod_method = 'mhg'
		else:
			x_type = 'numeric'
			encod_method = None
			
		mutual_info_coeff = RefinedMutualInfoCoeff(x, y, x_type = x_type)
		mic = mutual_info_coeff.cal_mic(encod_method = encod_method)
		
		entropy_results.append([func, mic])
	
	# 整理结果.
	entropy_results = pd.DataFrame(entropy_results, columns = ['func', 'entropy'])
