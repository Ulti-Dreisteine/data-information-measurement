# -*- coding: utf-8 -*-
"""
Created on 2021/12/20 17:21:19

@File -> mutual_info.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 
"""

__doc__ = """
    本代码用于对一维和多维离散或连续变量数据的信息熵和互信息进行计算.
    连续变量信息熵使用Kraskov和Lombardi等人的方法计算, Lord等人文献可作为入门;离散变量信息熵则直接进行计算.

    求得信息熵和互信息后, 便可对条件熵进行求取:

    H(Y|X) = H(Y) - I(X;Y)

    此处不再补充条件熵的相关代码.

    参考文献
    ------
    1. W. M. Lord, J. Sun, E. M. Bolt: Geometric K-Nearest Neighbor Estimation of Entropy and Mutual information, 2018.
    2. A. Kraskov, H. Stoegbauer, P. Grassberger: Estimating Mutual Information, 2003.
    3. D. Lombardi, S. Pant: A Non-Parametric K-Nearest Neighbor Entropy Estimator, 2015.
    4. B. C. Ross: Mutual Information between Discrete and Continuous Data Sets, 2014.
"""

import numpy as np

from . import DTYPES
from ._mi_cc import MutualInfoCC
from ._mi_cd import MutualInfoCD
from ._mi_dd import MutualInfoDD


class MutualInfo(MutualInfoCC, MutualInfoCD, MutualInfoDD):
    """单个一维或多维变量的信息熵"""

    def __init__(self, x: np.ndarray, y: np.ndarray, x_type: str, y_type: str):
        assert (x_type in DTYPES) & (y_type in DTYPES)
        self.x, self.y = x, y 
        self.x_type, self.y_type = x_type, y_type

    def __call__(self, **kwargs):
        # 对于多继承问题, 使用super(MutualInfo, self)的可能有问题, 会把MutualInfo所有父类都执行一遍. 因此下面分别
        # 根据不同父类具体实例化后计算.
        x, y = self.x, self.y

        if (self.x_type == 'c') & (self.y_type == 'c'):
            return MutualInfoCC(x, y)(**kwargs)
        elif (self.x_type == 'c') & (self.y_type == 'd'):
            return MutualInfoCD(x, y)(**kwargs)
        elif (self.x_type == 'd') & (self.y_type == 'c'):
            return MutualInfoCD(y, x)(**kwargs)
        elif (self.x_type == 'd') & (self.y_type == 'd'):
            return MutualInfoDD(x, y)()
        else:
            raise ValueError('method not supported')