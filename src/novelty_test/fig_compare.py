# -*- coding: utf-8 -*-
"""
Created on 2021/02/13 10:11

@Project -> File: data-information-measure -> fig_compare.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 各函数分布图像对比
"""

import seaborn as sns
import numpy as np
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(
    os.path.abspath(__file__), '../../../'))
sys.path.append(BASE_DIR)

from src.settings import *
from src.novelty_test import FUNCS2TEST
from src.novelty_test.main import mic, kl_knn, gen_samples, add_circle_noise

if __name__ == '__main__':

    # ---- 参数设置 ---------------------------------------------------------------------------------
    
    N = 1000
    method = 'mic'
    entropy_set = 0.2

    # ---- 测试代码 ---------------------------------------------------------------------------------

    proj_plt.figure(figsize=[9, 2])
    for i, func in enumerate(FUNCS2TEST[:5]):
        print('i = {}, func = {}'.format(i, func))
        x, y, x_norm, y_norm = gen_samples(N, func)
        rad_c = None
        for radius in np.arange(0.0, 2.0 + 0.005, 0.005):
            x_noise, y_noise = add_circle_noise(
                x_norm, y_norm, radius=radius + EPS)
            entropy = eval(method)(x_noise, y_noise)

            if entropy < entropy_set:
                rad_c = radius
                break

        x_noise, y_noise = add_circle_noise(x_norm, y_norm, radius=rad_c)
        proj_plt.subplot(1, 5, i + 1)
        # proj_plt.scatter(x_noise, y_noise, s = 2, c = 'k')
        sns.kdeplot(x_noise, y_noise, shade=True, shade_lowest=True)
        sns.scatterplot(x_noise, y_noise, s=1, marker='x')
        proj_plt.legend([func], loc='upper right', fontsize=6)
        proj_plt.tight_layout()
        proj_plt.pause(0.05)
    proj_plt.tight_layout()
    proj_plt.show()

    # 保存至本地.
    proj_plt.savefig(
        os.path.join(
            PROJ_DIR, 'img/novelty_test/{}/entropy_set_{}.png'.format(method, entropy_set)),
        dpi=450
    )
