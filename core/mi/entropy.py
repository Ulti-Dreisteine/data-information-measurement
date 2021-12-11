# -*- coding: utf-8 -*-
"""
Created on 2021/12/10 20:27:50

@File -> entropy.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 数据信息熵和互信息计算
"""

from sklearn.neighbors import BallTree, KDTree
from scipy.special import digamma
from numpy import log
import numpy as np
import warnings

__doc__ = """
    (补充参考文献)
"""


# ---- 工具函数 -------------------------------------------------------------------------------------

def add_noise(x, intens=1e-10):
    """加入噪声"""
    # small noise to break degeneracy, see doc.
    return x + intens * np.random.random_sample(x.shape)


def build_tree(x, metric: str = 'chebyshev'):
    """建立近邻查询树. 低维用KDTree; 高维用BallTree"""
    # NOTE: x.shape必须为(N, D)
    tree = BallTree(x, metric=metric) if x.shape[1] >= 20 else KDTree(
        x, metric=metric)
    return tree


def query_neighbors(tree, x, k: int):
    """求得x样本在tree上的前k个近邻样本"""
    return tree.query(x, k=k + 1)[0][:, k]


def count_neighbors(tree, x, r):
    return tree.query_radius(x, r, count_only=True)


def avgdigamma(points, dvec):
    # This part finds number of neighbors in some radius in the marginal space
    # returns expectation value of <psi(nx)>
    tree = build_tree(points)
    dvec = dvec - 1e-15
    num_points = count_neighbors(tree, points, dvec)
    return np.mean(digamma(num_points))


def _convert_1d_series2int(x):
    """将一维数据按照标签进行编码为连续整数"""
    x = x.flatten()
    x_unique = np.unique(x)
    x = np.apply_along_axis(lambda x: np.where(
        x_unique == x)[0][0], 1, x.reshape(-1, 1))
    return x


def convert_arr2int(arr):
    """将一维数据按照标签进行编码为连续整数"""
    _, D = arr.shape
    for d in range(D):
        arr[:, d] = _convert_1d_series2int(arr[:, d])
    return arr.astype(int)


def preprocess_values(x, x_type):
    x = x.copy()
    x = x.reshape(-1, 1) if len(x.shape) == 1 else x
    x = convert_arr2int(x) if x_type == 'd' else x  # 检查数据类型并转化成对应值
    return x


# ---- 信息熵计算 -----------------------------------------------------------------------------------

# 单变量的信息熵.
# NOTE: 变量可以是一维的也可以是多维的.

def _c_entropy(x, k: int = None, base: int = None):
    """连续数据信息熵, 使用Kraskov的KNN方法计算"""
    k = 3 if k is None else k
    base = 2 if base is None else base
    assert k <= len(x) - 1
    N, D = x.shape
    x = add_noise(x)  # small noise to break degeneracy, see doc.
    tree = build_tree(x)
    nn = query_neighbors(tree, x, k)
    const = digamma(N) - digamma(k) + D * log(2)
    return (const + D * np.log(nn).mean()) / log(base)


def _d_entropy(x, base: int = None):
    """离散数据信息熵: 计算离散熵"""
    base = 2 if base is None else base
    _, count = np.unique(x, return_counts=True, axis=0)
    proba = count.astype(float) / len(x)
    proba = proba[proba > 0.0]
    return np.sum(proba * np.log(1. / proba)) / log(base)


def cal_entropy(x, x_type, **kwargs):
    """单变量信息熵"""
    x = preprocess_values(x, x_type)

    k = kwargs['k'] if 'k' in kwargs.keys() else None
    base = kwargs['base'] if 'base' in kwargs.keys() else None

    if x_type == 'c':
        return _c_entropy(x, k, base)
    elif x_type == 'd':
        return _d_entropy(x, base)
    else:
        raise ValueError


# 两个变量之间的条件熵.

def _cc_cond_entropy(x, y, k, base):
    """连续-数据之间的条件熵 H(x|y) = H(x,y) - H(y)"""
    xy = np.hstack((x, y))
    entropy_xy = _c_entropy(xy, k, base)
    entropy_y = _c_entropy(y, k, base)
    return entropy_xy - entropy_y


def _cd_cond_entropy(x, y, k, base, warning=True):
    """连续-离散数据之间的条件熵"""
    k = 3 if k is None else k

    entropy_x = _c_entropy(x, k, base)

    y_unique, y_count = np.unique(y, return_counts=True, axis=0)
    y_proba = y_count / y.shape[0]

    entropy_x_given_y = 0.0
    for yval, py in zip(y_unique, y_proba):
        x_given_y = x[(y == yval).all(axis=1)]
        if k <= len(x_given_y) - 1:
            entropy_x_given_y += py * _c_entropy(x_given_y, k, base)
        else:
            if warning:
                warnings.warn(
                    "Warning, after conditioning, on y={yval} insufficient data. \
                    Assuming maximal entropy in this case.".format(yval=yval))
            entropy_x_given_y += py * entropy_x
    return entropy_x_given_y


def _dd_cond_entropy(x, y):
    """离散-离散数据之间的条件熵"""
    xy = np.hstack((x, y))
    return _d_entropy(xy) - _d_entropy(y)


def cal_cond_entropy(x, y, x_type, y_type, **kwargs):
    """双变量条件熵"""
    x = preprocess_values(x, x_type)
    y = preprocess_values(y, y_type)

    k = kwargs['k'] if 'k' in kwargs.keys() else None
    base = kwargs['base'] if 'base' in kwargs.keys() else None

    if (x_type == 'c') & (y_type == 'c'):
        return _cc_cond_entropy(x, y, k, base)
    elif (x_type == 'c') & (y_type == 'd'):
        return _cd_cond_entropy(x, y, k, base)
    elif (x_type == 'd') & (y_type == 'd'):
        return _dd_cond_entropy(x, y)
    else:
        raise ValueError


# ---- 互信息计算 -----------------------------------------------------------------------------------

def cal_mutual_info(x, y, x_type, y_type, **kwargs):
    """互信息"""
    x = preprocess_values(x, x_type)
    y = preprocess_values(y, y_type)

    k = kwargs['k'] if 'k' in kwargs.keys() else 3
    base = kwargs['base'] if 'base' in kwargs.keys() else 2

    if (x_type == 'c') & (y_type == 'c'):  # NOTE: 这里没有使用 I(x;y) = H(x) + H(y) - H(x,y)
        assert k <= len(x) - 1
        x, y = add_noise(x), add_noise(y)
        points = np.hstack([x, y])
        tree = build_tree(points)
        dvec = query_neighbors(tree, points, k)
        a, b, c, d = avgdigamma(x, dvec), avgdigamma(
            y, dvec), digamma(k), digamma(len(x))
        return (-a - b + c + d) / log(base)
    elif (x_type == 'd') & (y_type == 'd'):  # I(x;y) = H(x) + H(y) - H(x,y)
        xy = np.hstack((x, y))
        return cal_entropy(x, x_type=x_type) + cal_entropy(y, x_type=y_type) - cal_entropy(xy, 'd')

    # TODO: 混合类型的还算得不准, 互信息甚至小于0.
    elif (x_type == 'c') & (y_type == 'd'):  # I(x;y) = H(x) - H(x|y)
        return abs(cal_entropy(x, x_type=x_type) - cal_cond_entropy(x, y, x_type=x_type, y_type=y_type))  # TODO: 为何要取绝对值?
    elif (x_type == 'd') & (y_type == 'c'):  # I(x;y) = H(y) - H(y|x)
        return abs(cal_entropy(y, x_type=y_type) - cal_cond_entropy(y, x, x_type=y_type, y_type=x_type))


def cal_cond_mutual_info(x, y, z, x_type, y_type, z_type, **kwargs):
    """条件互信息"""
    x = preprocess_values(x, x_type)
    y = preprocess_values(y, y_type)
    z = preprocess_values(z, z_type)

    k = kwargs['k'] if 'k' in kwargs.keys() else 3
    base = kwargs['base'] if 'base' in kwargs.keys() else 2

    if z_type == 'c':
        if (x_type == 'c') & (y_type == 'c'):  # NOTE: 这里没有使用 I(x;y) = H(x) + H(y) - H(x,y)
            assert k <= len(x) - 1
            x, y = add_noise(x), add_noise(y)
            points = np.hstack([x, y, z])
            tree = build_tree(points)
            dvec = query_neighbors(tree, points, k)
            xz = np.c_[x, z]
            yz = np.c_[y, z]
            a, b, c, d = avgdigamma(xz, dvec), avgdigamma(
                yz, dvec), avgdigamma(z, dvec), digamma(k)
            return (-a - b + c + d) / log(base)
        else:
            raise ValueError
    elif z_type == 'd':
        z_unique, z_count = np.unique(z, return_counts=True, axis=0)
        z_proba = z_count / z.shape[0]

        mi_given_z = 0.0
        for zval, pz in zip(z_unique, z_proba):
            x_given_z = x[(z == zval).all(axis=1)]
            y_given_z = y[(z == zval).all(axis=1)]
            mi_xy = cal_mutual_info(
                x_given_z, y_given_z, x_type, y_type, **kwargs)
            mi_given_z += pz * mi_xy
        return mi_given_z
