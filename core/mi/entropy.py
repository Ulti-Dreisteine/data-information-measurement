# -*- coding: utf-8 -*-
"""
Created on 2021/12/14 12:38:06

@File -> entropy.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 信息熵和互信息计算
"""

__doc__ = """
    本代码用于对一维和多维离散或连续变量数据的信息熵和互信息进行计算.
    连续变量信息熵使用Kraskov和Lombardi等人的方法计算, Lord等人文献可作为入门;离散变量信息熵则直接进行计算.

    求得信息熵和互信息后, 便可对条件熵进行求取:

    H(Y|X) = H(Y) - I(X;Y)

    此处不再补充相关代码.

    参考文献
    ------
    1. W. M. Lord, J. Sun, E. M. Bolt: Geometric K-Nearest Neighbor Estimation of Entropy and Mutual information, 2018.
    2. A. Kraskov, H. Stoegbauer, P. Grassberger: Estimating Mutual Information, 2003.
    3. D. Lombardi, S. Pant: A Non-Parametric K-Nearest Neighbor Entropy Estimator, 2015.
    4. B. C. Ross: Mutual Information between Discrete and Continuous Data Sets, 2014.
"""

from sklearn.neighbors import BallTree, KDTree
from scipy.special import digamma, gamma, psi
from numpy import log
import numpy as np

BASE = 2                # 计算信息熵所采用的底数
DTYPES = ['d', 'c']     # 数据类型, 离散或连续


# ---- 工具函数 -------------------------------------------------------------------------------------

# 数据处理.

def _convert_1d_series2int(x):
    """将一维数据按照标签进行编码为连续整数"""
    x = x.flatten()
    x_unique = np.unique(x)
    print('too many labels > 100 for the discrete data') if len(x_unique) > 100 else ...
    x = np.apply_along_axis(lambda x: np.where(
        x_unique == x)[0][0], 1, x.reshape(-1, 1))
    return x


def convert_arr2int(arr):
    """将一维数据按照标签进行编码为连续整数"""
    _, D = arr.shape
    for d in range(D):
        arr[:, d] = _convert_1d_series2int(arr[:, d])
    return arr.astype(int)


def preprocess_values(x: np.ndarray, d_type: str, intens: float = 1e-10):
    x = x.copy()
    assert d_type in DTYPES

    x = x.reshape(-1, 1) if len(x.shape) == 1 else x

    if d_type == 'd':
        x = convert_arr2int(x)  # 检查数据类型并转化成对应值
    else:
        x += intens * np.random.random_sample(x.shape)  # 加入噪声避免数据退化.

    return x


# KNN参数.

def deter_k(x):
    """
    根据样本量确定近邻点数目

    参考文献
    ------ 
    J. B. Kinney, G. S. Atwal, Equitability, Mutual Information, and the Maximal Information Coefficient, 2014
    """
    if len(x) < 1000:
        k = 3
    elif (len(x) < 3000) & (len(x) >= 1000):
        k = 2
    else:
        k = 1
    return k


def build_tree(x):
    """建立近邻查询树. 低维用具有欧式距离特性的KDTree; 高维用具有更一般距离特性的BallTree"""
    if x.shape[1] <= 3:
        metric = 'euclidean'
    else:
        metric = 'minkowski'  # 'chebyshev' 

    tree = BallTree(x, metric=metric) if x.shape[1] >= 20 else KDTree(
        x, metric=metric)
    return tree


def query_neighbors_dist(tree: BallTree or KDTree, x, k: int):
    """求得x样本在tree上的第k个近邻样本"""
    return tree.query(x, k=k + 1)[0][:, -1]


# ---- 信息熵 ---------------------------------------------------------------------------------------

class Entropy(object):
    """单变量(一维或多维)信息熵"""

    def __init__(self, x: np.array, d_type: str):
        self.x = preprocess_values(x, d_type)
        self.dtype = d_type

    # @staticmethod
    # def _add_noise(x, intens=1e-10):
    #     return x + intens * np.random.random_sample(x.shape)  # 加入噪声避免数据退化.

    def __call__(self, method: str = 'Kraskov'):
        """计算信息熵. 参数method只有对连续变量起作用"""
        x = self.x

        if self.dtype == 'd':
            _, count = np.unique(x, return_counts=True, axis=0)
            proba = count.astype(float) / len(x)
            proba = proba[proba > 0.0]
            return np.sum(proba * np.log(1. / proba)) / log(BASE)
        else:
            """关于连续变量的计算, 以下分别按照Kraskov和Lombardi的方式实现"""
            k = deter_k(x)
            assert k <= len(x) - 1

            N, D = x.shape
            tree = build_tree(x)
            nn_distc = query_neighbors_dist(tree, x, k)  # 获得了各样本第k近邻的距离

            # Kraskov方法.
            if method == 'Kraskov':
                const = digamma(N) - digamma(k) + D * log(2)
                return (const + D * np.log(nn_distc).mean()) / log(BASE)

            # Lombardi方法.
            elif method == 'Lombardi':
                volume_unit_ball = (np.pi ** (.5 * D)) / gamma(.5 * D + 1)
                return (D * np.mean(np.log(nn_distc + np.finfo(x.dtype).eps)) +
                        np.log(volume_unit_ball) + digamma(N) - digamma(k))

            else:
                raise ValueError


class MutualInfo(object):
    """互信息. 当X和Y数据类型相同时, 直接使用H(X) + H(Y) - H(X, Y); 当二者数据类型不同时, 则采用Ross的方法"""

    def __init__(self, x: np.ndarray, y: np.ndarray, x_type: str, y_type: str):
        self.x = preprocess_values(x, x_type)
        self.y = preprocess_values(y, y_type)
        self.x_type, self.y_type = x_type, y_type

    @staticmethod
    def _mi_cd(x, y, k: int = None):
        """
        计算连续的X和离散的Y之间的互信息

        参考文献: 
        ------
        B. C. Ross: Mutual Information between Discrete and Continuous Data Sets, 2014.
        """
        N, _ = x.shape
        assert (x.shape[1] >= 1) & (y.shape[1] == 1)  # NOTE: 此处y必须为1维
        y = y.flatten()

        # k = deter_k(x) if k is None else k
        k = 3 if k is None else k  # NOTE: 设置为3比较合适
        assert k <= N - 1

        # 统计各类Y的总数.
        classes = np.unique(y)
        Nx_class = np.zeros_like(y)
        for i in range(len(y)):
            Nx_class[i] = np.sum(y == y[i])

        # 逐类进行K近邻计算.
        nn_distc_classes = np.zeros_like(y, dtype=float)
        for c in classes:
            mask = np.where(y == c)[0]
            tree = build_tree(x[mask, :])
            nn_distc_classes[mask] = query_neighbors_dist(
                tree, x[mask, :], k)  # 获得了各样本第k近邻的距离

        # 所有样本中的K近邻计算.
        tree = build_tree(x)
        m = tree.query_radius(x, nn_distc_classes)
        m = [p.shape[0] for p in m]
        return digamma(N) - np.mean(digamma(Nx_class)) + digamma(k) - np.mean(digamma(m))

    @staticmethod
    def _mi_cc(x, y, k: int = None):
        """计算连续变量之间的互信息

        参考文献: 
        ------
        A. Kraskov, H. Stoegbauer, P. Grassberger: Estimating Mutual Information, 2003.
        """
        N, _ = x.shape
        k = deter_k(x) if k is None else k
        assert k <= len(x) - 1

        xy = np.hstack((x, y))
        tree = build_tree(xy)
        nn_distc = query_neighbors_dist(tree, xy, k)  # 获得了各样本第k近邻的距离

        tree = build_tree(x)
        nn_distc_x = nn_distc - 1e-15  
        Nx = tree.query_radius(x, nn_distc_x, count_only=True)

        tree = build_tree(y)
        nn_distc_y = nn_distc - 1e-15  
        Ny = tree.query_radius(y, nn_distc_y, count_only=True)

        return digamma(N) + digamma(k) - np.mean(digamma(Nx)) - np.mean(digamma(Ny))

    def __call__(self, **kwargs):
        """计算互信息"""
        x, y = self.x, self.y

        if (self.x_type == 'd') & (self.y_type == 'd'):
            xy = np.hstack((x, y))
            return (sum([Entropy(v, 'd')() for v in [x, y]]) - Entropy(xy, 'd')())

        elif (self.x_type == 'c') & (self.y_type == 'c'):
            return self._mi_cc(x, y, **kwargs)

        elif (self.x_type == 'c') & (self.y_type == 'd'):
            return self._mi_cd(x, y, **kwargs)

        elif (self.x_type == 'd') & (self.y_type == 'c'):
            """计算离散X和连续Y之间的互信息, 利用互信息的交换律"""
            assert (x.shape[1] == 1) & (y.shape[1] >= 1)  # NOTE: 此处x必须为1维
            return self._mi_cd(y, x, **kwargs)
        
        else:
            pass


def cond_mutual_info(x, y, z, x_type, y_type, z_type, approach: str = 'binning_z', **kwargs):
    """计算条件互信息I(X;Y|Z)"""
    x = preprocess_values(x, x_type)
    y = preprocess_values(y, y_type)
    z = preprocess_values(z, z_type)

    # 对Z数据进行离散化, 然后对所有区间上的I(X;Y|Z=zi)进行加和.
    if approach == 'binning_z':
        ...

    # 直接采用互信息做减法.
    elif approach == 'mutual_info':
        if y_type == z_type:
            """I(X;Y|Z) = I(X;Y,Z) - I(X;Z)"""
            yz = np.hstack((y, z))
            return MutualInfo(x, yz, x_type, y_type)(**kwargs) - MutualInfo(x, z, x_type, z_type)(**kwargs)
        else:
            raise ValueError
        

if __name__ == '__main__':
    import sys
    import os

    BASE_DIR = os.path.abspath(os.path.join(
        os.path.abspath(__file__), '../' * 3))
    sys.path.append(BASE_DIR)

    from core.dataset.data_generator import DataGenerator

    # ---- 生成数据样本 -----------------------------------------------------------------------------

    N = 10000
    data_generator = DataGenerator(N_ticks=int(1e5))

    func = 'categorical'
    x1, y1, _, _ = data_generator.gen_data(N, func)
    x1_type, y1_type = 'd', 'd'

    func = 'parabola'
    x2, y2, _, _ = data_generator.gen_data(N, func)
    x2_type, y2_type = 'c', 'c'

    # ---- 测试 ------------------------------------------------------------------------------------

    # self = Entropy(x1, x1_type)
    # print(self('Kraskov'))
    # print(self('Lombardi'))

    # ---- 测试 ------------------------------------------------------------------------------------

    # 互信息计算.
    mi_xy = MutualInfo(x1, y1, x1_type, y1_type)
    print(mi_xy())
    
    # ---- 测试 -------------------------------------------------------------------------------------

    cmi = cond_mutual_info(x1, y1, x1, x1_type, y1_type, x1_type, approach='mutual_info')
    print('cmi: %f' % cmi)

    cmi = cond_mutual_info(x2, y2, x2, x2_type, y2_type, x2_type, approach='mutual_info')
    print('cmi: %f' % cmi)
    
