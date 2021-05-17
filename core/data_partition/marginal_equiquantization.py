# -*- coding: utf-8 -*-
"""
Created on 2021/05/13 14:21:04

@File -> marginal_equiquantization.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 边际等概率离散化
"""

from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from typing import List, Tuple
import numpy as np
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(
    os.path.abspath(__file__), '../../../'))
sys.path.append(BASE_DIR)


# ---- 预处理 ----------------------------------------------------------------------------------------

def minmax_norm(arr: np.ndarray):
    D = arr.shape[1]
    scaler = MinMaxScaler()
    arr_norm = None
    for i in range(D):
        a = scaler.fit_transform(arr[:, i: i + 1])

        if arr_norm is None:
            arr_norm = a
        else:
            arr_norm = np.hstack((arr_norm, a))
    return arr_norm


# ---- 离散化 ---------------------------------------------------------------------------------------

# 单个cell.
class Cell(object):
    """离散化中的单元格对象
    """

    def __init__(self, arr: np.ndarray) -> None:
        """初始化

        :param arr: 存储有x,y坐标的二维数组, shape = (N, D=2)
        """
        self.arr = arr.copy()
        self.N, self.D = arr.shape

        if self.D != 2:
            raise ValueError('the input dimension is not equal to 2')

    def define_cell_bounds(self, bounds: List[tuple]):
        """定义cell边界

        :param bounds: list of tuples, 如[(x_min, x_max), (y_min, y_max)]
        """
        self.bounds = bounds

    def cal_area(self):
        self.area = (self.bounds[0][1] - self.bounds[0][0]) \
            * (self.bounds[1][1] - self.bounds[1][0])

    def _get_marginal_partition_thres(self):
        """各维度上按照边际概率(即样本数)相等的方式划分为两个子集
        """
        # if self.N == 0:
        #     return None
        # else:
        part_idx = self.N // 2  # 离散化位置idx
        part_thres = []
        for i in range(self.D):
            arr_srt = self.arr[np.argsort(self.arr[:, i]), :]  # 对应维度值升序排列
            marginal_part_value = arr_srt[part_idx, i]  # >=该值的样本被划分入新的cell
            part_thres.append(marginal_part_value)
        return part_thres

    def get_marginal_partition_thres(self) -> list:
        """获得各边际离散化阈值
        """
        self.part_thres = self._get_marginal_partition_thres()

    def equiprob_partition(self) -> tuple:
        """等概率离散化
        """
        # 先在x方向上分为左右两部分.
        part_arr_l = self.arr[
            np.where((self.arr[:, 0] < self.part_thres[0]) &
                     (self.arr[:, 0] >= self.bounds[0][0]))
        ]
        part_arr_r = self.arr[
            np.where((self.arr[:, 0] >= self.part_thres[0])
                     & (self.arr[:, 0] <= self.bounds[0][1]))
        ]

        # 再在y方向上继续切分.
        part_arr_ul = part_arr_l[np.where(
            (part_arr_l[:, 1] >= self.part_thres[1]) & (part_arr_l[:, 1] <= self.bounds[1][1]))]
        part_arr_ll = part_arr_l[np.where(
            (part_arr_l[:, 1] < self.part_thres[1]) & (part_arr_l[:, 1] >= self.bounds[1][0]))]

        part_arr_ur = part_arr_r[np.where(
            (part_arr_r[:, 1] >= self.part_thres[1]) & (part_arr_r[:, 1] <= self.bounds[1][1]))]
        part_arr_lr = part_arr_r[np.where(
            (part_arr_r[:, 1] < self.part_thres[1]) & (part_arr_r[:, 1] >= self.bounds[1][0]))]

        cell_ul, cell_ur, cell_ll, cell_lr = Cell(part_arr_ul), Cell(part_arr_ur), \
            Cell(part_arr_ll), Cell(part_arr_lr)

        # 确定边界.
        (xl, xu), (yl, yu) = self.bounds
        x_thres, y_thres = self.part_thres
        cell_ul.define_cell_bounds([(xl, x_thres), (y_thres, yu)])
        cell_ur.define_cell_bounds([(x_thres, xu), (y_thres, yu)])
        cell_ll.define_cell_bounds([(xl, x_thres), (yl, y_thres)])
        cell_lr.define_cell_bounds([(x_thres, xu), (yl, y_thres)])
        return cell_ul, cell_ur, cell_ll, cell_lr

    def cal_P(self):
        self.cal_area()
        if self.area == 0.0:
            return 0.0
        else:
            return self.N / self.area

    def show(self, linewidth: float = 0.5):
        (xl, xu), (yl, yu) = self.bounds
        plt.plot([xl, xu], [yl, yl], '-', c='k', linewidth=linewidth)
        plt.plot([xu, xu], [yl, yu], '-', c='k', linewidth=linewidth)
        plt.plot([xu, xl], [yu, yu], '-', c='k', linewidth=linewidth)
        plt.plot([xl, xl], [yu, yl], '-', c='k', linewidth=linewidth)


def recursive_partition(cell: Cell, p_eps: float = 1e-3, min_samples_split: int = 100, min_leaf_samples: int = 0) -> tuple:
    # TODO: 关于这里的参数应该再讨论一下, p_eps可以保留, 但是cell的最小分裂样本min_samples_split可以=1，分裂后
    # min_leaf_samples可以为0.
    leaf_cells = []

    def partition(cell):

        def _try_partition(cell: Cell):
            P = cell.cal_P()

            # 尝试分裂一下, 并检查分裂效果.
            cell.get_marginal_partition_thres()
            cell_ul, cell_ur, cell_ll, cell_lr = cell.equiprob_partition()

            is_valid_split = True if cell.N >= min_samples_split else False
            is_P_converged = True
            is_N_limited = False

            for c in [cell_ul, cell_ur, cell_ll, cell_lr]:
                if (np.abs(c.cal_P() - P) / P > p_eps):
                    is_P_converged = False
                if c.N < min_leaf_samples:
                    is_N_limited = True

            if is_valid_split & (not is_P_converged) & (not is_N_limited):
                return cell_ul, cell_ur, cell_ll, cell_lr
            else:
                return None, None, None, None

        part_ul, part_ur, part_ll, part_lr = _try_partition(cell)

        if part_ul is None:
            leaf_cells.append(cell)
        else:
            partition(part_ul)
            partition(part_ur)
            partition(part_ll)
            partition(part_lr)

    partition(cell)

    return leaf_cells


if __name__ == '__main__':
    from src.settings import *

    # ---- 测试用函数 -------------------------------------------------------------------------------

    def load_data(func: str) -> Tuple[np.ndarray, np.ndarray]:
        """载入数据
        """
        from core.dataset.data_generator import DataGenerator

        N = 1000
        data_gener = DataGenerator(N=N)
        x, y, _, _ = data_gener.gen_data(func, normalize=True)

        # 加入噪音.
        from mod.data_process.add_noise import add_circlular_noise

        x, y = add_circlular_noise(x, y, radius=0.2)

        return x, y

    # ---- 生成数据 ---------------------------------------------------------------------------------

    func = 'cubic'
    x, y = load_data(func)
    arr = np.vstack((x, y)).T
    arr = minmax_norm(arr)

    # ---- 测试代码 ---------------------------------------------------------------------------------

    cell = Cell(arr)
    cell.define_cell_bounds([(0.0, 1.0), (0.0, 1.0)])

    proj_plt.figure(figsize=[6, 6])
    proj_plt.scatter(cell.arr[:, 0], cell.arr[:, 1], s=3)

    leaf_cells = recursive_partition(cell)

    empty_c_lst = []
    for c in leaf_cells:
        c.get_marginal_partition_thres()
        if c.part_thres is None:
            empty_c_lst.append(c)
        else:
            c.show()

    proj_plt.title('{}'.format(func), fontsize=18, fontweight='bold')
    proj_plt.xlabel('$\it{x}$')
    proj_plt.ylabel('$\it{y}$')
    proj_plt.savefig(os.path.join(
        PROJ_DIR, 'img/partitions/{}.png'.format(func)), dpi=450)
