# -*- coding: utf-8 -*-
"""
Created on 2020/11/4 21:29

@Project -> File: refined-mic-chemical-process -> _univar_encoding.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 单变量样本编码
"""

from collections import defaultdict
import category_encoders as ce
import pandas as pd
import numpy as np


class UnsuperCategorEncoding(object):
	"""将多元序列编码为一元序列"""

	def __init__(self, x: np.ndarray):
		self.x = x.astype(np.int).flatten()
		self.N = len(x)

	@staticmethod
	def get_map(x: np.ndarray):
		labels_discret = np.unique(x, axis = 0).astype(int).astype(str)
		map_ = defaultdict(int)
		for i in range(len(labels_discret)):
			map_[labels_discret[i]] = i
		return map_

	@staticmethod
	def _convert2label(x, map_):
		x = x.astype(int).astype(str)
		key = ''.join(list(x))
		return map_[key]

	def encode(self, method: str):
		map_ = self.get_map(self.x)
		series_encoded = np.apply_along_axis(lambda x: self._convert2label(x, map_), 1, self.x.reshape(-1, 1))
		df = pd.DataFrame(series_encoded, columns = ['label'])

		if method == 'label':
			return df.values.flatten()
		elif method == 'random':
			values = list(map_.values())
			np.random.seed(None)
			mapping = dict(zip(values, np.random.permutation(values)))
			encoder = ce.OrdinalEncoder(
				cols = ['Degree'],
				return_df = True,
				mapping = [
					{'col': 'label', 'mapping': mapping}
				]
			)
			return encoder.fit_transform(df).values.flatten()
		elif method == 'freq':
			mapping = dict(df['label'].value_counts())
			encoder= ce.OrdinalEncoder(
				cols=['Degree'],
				return_df=True,
				mapping=[
					{'col':'label', 'mapping': mapping}
				]
			)
			return encoder.fit_transform(df).values.flatten()
		else:
			raise RuntimeError('')


class SuperCategorEncoding(object):
	"""将多元序列编码为一元序列"""

	def __init__(self, x: np.ndarray, y: np.ndarray):
		self.x = x.astype(np.int).flatten()
		self.y = y.astype(np.float32).flatten()
		self.N = len(x)

	@staticmethod
	def get_map(x: np.ndarray):
		labels_discret = np.unique(x, axis = 0).astype(int).astype(str)
		map_ = defaultdict(int)
		for i in range(len(labels_discret)):
			map_[labels_discret[i]] = i
		return map_

	@staticmethod
	def _convert2label(x, map_):
		x = x.astype(int).astype(str)
		key = ''.join(list(x))
		return map_[key]

	def encode(self) -> np.ndarray:
		map_ = self.get_map(self.x)
		# 先暂时打上标签.
		series_encoded = np.apply_along_axis(lambda x: self._convert2label(x, map_), 1, self.x.reshape(-1, 1))

		# 进行均值编码.
		arr = np.vstack((series_encoded, self.y)).T

		# 获取所有的label.
		labels = list(np.unique(arr[:, 0]))

		# 计算每个label上对应目标的均值.
		label_avg = {}
		for label in labels:
			sub_arr = arr.copy()[arr[:, 0] == label]
			series = sub_arr[:, 1]
			label_avg[label] = np.sum(series) / series.shape[0]

		# 将label按照目标均值大小进行排序, 获得排序后的label列表.
		lst = sorted(zip(label_avg.values(), label_avg.keys()))
		labels_sorted = [p[1] for p in lst]

		# 类别值映射为数值.
		map_ = dict(zip(labels_sorted, range(len(labels_sorted))))
		x_encoded = np.apply_along_axis(
			lambda x: float(map_[x[0]]), 1, arr[:, 0].reshape(-1, 1)
		)
		return x_encoded


if __name__ == '__main__':
	pass



