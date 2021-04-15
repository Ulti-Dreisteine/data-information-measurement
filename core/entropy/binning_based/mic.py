# -*- coding: utf-8 -*-
"""
Created on 2021/02/10 15:58

@Project -> File: data-information-measure -> mic.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 最大信息系数MIC
"""

from minepy import MINE
import numpy as np


class MutualInfoCoeff(object):
	"""
	计算两个变量间的互信息系数
	
	Reference:
	1. D.N. Reshef, Y.A. Reshef, et al.: "Supporting Online Material for Detecting Novel Associations
		in Large Data Sets" (Table S3), Science, 2012.
	"""
	
	def __init__(self, x: np.ndarray, y: np.ndarray, mic_params: dict = None):
		self.x, self.y = x.astype(np.float32).flatten(), y.astype(np.float32).flatten()
		self.mic_params = mic_params
		
	def cal_mic(self):
		if self.mic_params is not None:
			mine = MINE(**self.mic_params)
		else:
			mine = MINE()
		mine.compute_score(self.x, self.y)
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
		mutual_info_coeff = MutualInfoCoeff(x, y)
		mic = mutual_info_coeff.cal_mic()
		
		entropy_results.append([func, mic])
	
	# 整理结果.
	entropy_results = pd.DataFrame(entropy_results, columns = ['func', 'entropy'])
	
	
	
	

