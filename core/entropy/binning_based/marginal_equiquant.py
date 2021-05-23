# -*- coding: utf-8 -*-
"""
Created on 2021/05/13 18:53:54

@File -> marginal_equiquant.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 基于Marginal Equiquantization方法的互信息计算
"""

__doc__ = """
    参考文献: 
    Georges A. Darbellay: Predictability: An Information-Theoretic Perspective, Signal Analysis \
        and Prediction, 1998.
"""

from typing import Tuple
import numpy as np
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(
    os.path.abspath(__file__), '../../../../'))
sys.path.append(BASE_DIR)

from core.data_partition.marginal_equiquantization import Cell, minmax_norm, recursively_partition


class MutualInfoEntropy(object):

    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = x.reshape(-1, 1)
        self.y = y.reshape(-1, 1)
        self.arr = minmax_norm(np.hstack((self.x, self.y)))
        self.N = self.arr.shape[0]

    def _equiquantize(self, **kwargs):
        cell = Cell(self.arr)
        cell.def_cell_bounds([(0.0, 1.0), (0.0, 1.0)])
        leaf_cells = recursively_partition(cell, **kwargs)

        leaf_cells = [c for c in leaf_cells if c.N > 0]

        return leaf_cells

    def equiquantize(self, **kwargs):
        self.leaf_cells = self._equiquantize(**kwargs)

    def cal_mie(self):
        n_leafs = len(self.leaf_cells)

        mie = 0.0
        for i in range(n_leafs):
            cell = self.leaf_cells[i]  # type: Cell
            (xl, xu), (yl, yu) = cell.bounds

            Nxy = len(cell.arr)
            Nx = len(np.where((self.arr[:, 0] >= xl)
                              & (self.arr[:, 0] < xu))[0])
            Ny = len(np.where((self.arr[:, 1] >= yl)
                              & (self.arr[:, 1] < yu))[0])

            gain = Nxy * np.log2(Nxy / Nx / Ny)
            mie += gain

        mie = mie / self.N + np.log2(self.N)
        return mie


def cal_mie(x: np.ndarray, y: np.ndarray, **kwargs):
    mutual_info_entropy = MutualInfoEntropy(x, y)
    mutual_info_entropy.equiquantize(**kwargs)
    mie = mutual_info_entropy.cal_mie()
    return mie


def cal_rho(x: np.ndarray, y: np.ndarray, **kwargs):
    mie = cal_mie(x, y, **kwargs)
    rho = np.sqrt(1 - np.power(2, -2 * mie))
    return rho


if __name__ == '__main__':
    from src.settings import *

    # ---- 载入数据 ---------------------------------------------------------------------------------

    def load_data(func: str, radius: float) -> Tuple[np.ndarray]:
        """载入数据
        """
        from core.dataset.data_generator import DataGenerator

        N = 10000
        data_gener = DataGenerator(N=N)
        x, y, _, _ = data_gener.gen_data(func, normalize=True)

        # 加入噪音.
        from mod.data_process.add_noise import add_circlular_noise

        x, y = add_circlular_noise(x, y, radius=radius)

        return x, y

    # ---- 生成数据 ---------------------------------------------------------------------------------

    from core.dataset.data_generator import FUNC_NAMES

    proj_plt.figure(figsize=[6, 6])
    for func in FUNC_NAMES[:1]:
        radius_lst = np.arange(0.1, 10.0, 0.1)
        mie_lst = []
        params = {'p_eps': 1e-3, 'min_samples_split': 100}
        for radius in radius_lst:
            x, y = load_data(func, radius)
            # mie = cal_mie(x, y, **params)
            mie = cal_rho(x, y, **params)
            mie_lst.append(mie)

        proj_plt.scatter(radius_lst, mie_lst, s=6)
        # proj_plt.pause(0.1)
