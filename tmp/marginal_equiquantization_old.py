# -*- coding: utf-8 -*-
"""
Created on 2021/05/12 16:45:21

@File -> marginal_equiquantization.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 数据边际等概率离散化
"""

from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, List
import numpy as np
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(
    os.path.abspath(__file__), '../../../'))
sys.path.append(BASE_DIR)


# ---- 测试 -----------------------------------------------------------------------------------------

def load_data(func: str) -> Tuple[np.ndarray, np.ndarray]:
    """载入数据
    """
    from core.dataset.data_generator import DataGenerator

    N = 10001
    data_gener = DataGenerator(N=N)
    x, y, _, _ = data_gener.gen_data(func, normalize=True)

    # 加入噪音.
    from mod.data_process.add_noise import add_circlular_noise

    x, y = add_circlular_noise(x, y, radius=0.15)

    return x, y


# ---- 离散化代码 -----------------------------------------------------------------------------------

# 单个Cell.
class Cell(object):
    """离散化中的Cell, 即单元格
    """

    def __init__(self, arr: np.ndarray) -> None:
        """初始化

        :param arr: 存储有x,y坐标的二维数组, shape = (N, D=2)
        :param bounds: x和y方向上的边界值, 如[[x_min, x_max], [y_min, y_max]]
        """
        self.arr = arr.copy()
        self.N, self.D = arr.shape

    def get_cell_bounds(self, bounds: List[tuple]):
        self.bounds = bounds

    def cal_area(self):
        self.area = (self.bounds[0][1] - self.bounds[0]
                     [0]) * (self.bounds[1][1] - self.bounds[1][0])

    def _get_marginal_partition_thres(self) -> list:
        """各维度上按照边际概率(即样本数)相等的方式划分为两个子集
        """
        if self.N == 0:
            return None
        else:
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
        part_arr_l = self.arr[np.where(self.arr[:, 0] < self.part_thres[0])]
        part_arr_r = self.arr[np.where(self.arr[:, 0] >= self.part_thres[0])]

        # 再在y方向上继续切分.
        part_arr_ul = part_arr_l[np.where(
            part_arr_l[:, 1] >= self.part_thres[1])]
        part_arr_ll = part_arr_l[np.where(
            part_arr_l[:, 1] < self.part_thres[1])]

        part_arr_ur = part_arr_r[np.where(
            part_arr_r[:, 1] >= self.part_thres[1])]
        part_arr_lr = part_arr_r[np.where(
            part_arr_r[:, 1] < self.part_thres[1])]

        cell_ul, cell_ur, cell_ll, cell_lr = Cell(part_arr_ul), Cell(
            part_arr_ur), Cell(part_arr_ll), Cell(part_arr_lr)

        # 确定边界.
        (xl, xu), (yl, yu) = self.bounds
        x_thres, y_thres = self.part_thres
        cell_ul.get_cell_bounds([(xl, x_thres), (y_thres, yu)])
        cell_ur.get_cell_bounds([(x_thres, xu), (y_thres, yu)])
        cell_ll.get_cell_bounds([(xl, x_thres), (yl, y_thres)])
        cell_lr.get_cell_bounds([(x_thres, xu), (yl, y_thres)])
        return cell_ul, cell_ur, cell_ll, cell_lr

    def cal_P(self):
        self.cal_area()
        return self.N / self.area

    def draw(self, linewidth: float = 0.5):
        (xl, xu), (yl, yu) = self.bounds
        proj_plt.plot([xl, xu], [yl, yl], '-', c='k', linewidth=linewidth)
        proj_plt.plot([xu, xu], [yl, yu], '-', c='k', linewidth=linewidth)
        proj_plt.plot([xu, xl], [yu, yu], '-', c='k', linewidth=linewidth)
        proj_plt.plot([xl, xl], [yu, yl], '-', c='k', linewidth=linewidth)


# 递归分裂.
def single_step_partition(cell: Cell) -> Tuple[Cell, Cell, Cell, Cell]:
    cell.get_marginal_partition_thres()
    return cell.equiprob_partition()


def recursively_partition(cell: Cell, eps: float = 1E-3, N_min: int = 50) -> tuple:
    """递归分裂离散化

    :param arr: 待离散化样本数据
    :param eps: 分裂前后概率密度相对变化, defaults to 1E-2
    :param N_min: 叶子节点样本数下界, defaults to 10
    """

    cell_ul, cell_ur, cell_ll, cell_lr = single_step_partition(cell)

    is_P_converged = True
    is_N_limited = False

    P = cell.cal_P()
    for c in [cell_ul, cell_ur, cell_ll, cell_lr]:
        if (np.abs(c.cal_P() - P) / P > eps):
            is_P_converged = False
        if c.N < N_min:
            is_N_limited = True

    if (not is_P_converged) & (not is_N_limited):
        return recursively_partition(cell_ul), recursively_partition(cell_ur), recursively_partition(cell_ll), recursively_partition(cell_lr)
    else:
        return cell_ul, cell_ur, cell_ll, cell_lr


# 解析结果.
def parse_partition_result(partition: tuple or Cell):
    leafs = []

    def _parse(p: tuple):
        for pi in p:
            if type(pi) == Cell:
                leafs.append(pi)
            else:
                _parse(pi)

    if type(partition) == Cell:
        leafs.append(partition)
    else:
        _parse(partition)  # TODO: 此处解析可能在eps过小时存在Bug

    return leafs


if __name__ == '__main__':
    from src.settings import *

    # ---- 载入数据 ---------------------------------------------------------------------------------

    func = 'cubic'
    x, y = load_data(func)

    # ---- 预处理 -----------------------------------------------------------------------------------

    # 数据归一化.
    scaler = MinMaxScaler()
    x = scaler.fit_transform(x.reshape(-1, 1))
    y = scaler.fit_transform(y.reshape(-1, 1))

    proj_plt.figure(figsize=[6, 6])
    proj_plt.scatter(x, y, s=3)

    # ---- 单个Cell测试 -----------------------------------------------------------------------------

    arr = np.hstack((x, y))
    cell = Cell(arr)
    cell.get_cell_bounds([(0.0, 1.0), (0.0, 1.0)])

    # cell.get_marginal_partition_thres()
    # cell_ul, cell_ur, cell_ll, cell_lr = cell.equiprob_partition()

    # cell_ul, cell_ur, cell_ll, cell_lr = single_step_partition()(cell)

    partition_ul, partition_ur, partition_ll, partition_lr = recursively_partition(
        cell)

    leaf_cells = []
    for partition in [partition_ul, partition_ur, partition_ll, partition_lr]:
        leaf_cells += parse_partition_result(partition)

    empty_c_lst = []
    for c in leaf_cells:
        c.get_marginal_partition_thres()
        if c.part_thres is None:
            empty_c_lst.append(c)
        else:
            c.draw()

    proj_plt.title('{}'.format(func), fontsize=18, fontweight='bold')
    proj_plt.xlabel('$\it{x}$')
    proj_plt.ylabel('$\it{y}$')
    proj_plt.savefig(os.path.join(
        PROJ_DIR, 'img/partitions/{}.png'.format(func)), dpi=450)
