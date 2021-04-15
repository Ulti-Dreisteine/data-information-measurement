# -*- coding: utf-8 -*-
"""
Created on 2019/12/12 上午10:39

@Project -> File: guodian-desulfuration-optimization -> time_stamp_serial.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 将数据在时间上进行连续化处理
"""

import logging

logging.basicConfig(level=logging.INFO)

import pandas as pd
import numpy as np
import bisect


def search_nearest_neighbors_in_list(lst, x):
    """
    寻找x在有序lst中的两侧（或单侧）邻点值.
    :param x: float
    :param lst: list, 必须有序排列
    :return: neighbors, tuple (left_neighbor, right_neighbor)
    """
    if x in lst:
        return [x]
    else:
        if x <= lst[0]:
            neighbors = [lst[0]]
        elif x >= lst[-1]:
            neighbors = [lst[-1]]
        else:
            left_idx = bisect.bisect_left(lst, x) - 1
            right_idx = left_idx + 1
            neighbors = [lst[left_idx], lst[right_idx]]
        return neighbors


class DataTimeStampSerialization(object):
    """
    数据时间戳连续化处理, 使用已有记录对未知记录未知进行线性插值填充.
    """

    def __init__(self, data: pd.DataFrame, start_stp: int or float, end_stp: int or float, stp_step: int or float, cols2serialize: list = None):
        """
        初始化
        :param data: pd.DataFrame, cols = {'time', vars, ...}, 数据总表
        :param start_stp: int, 起始时间戳
        :param end_stp: int, 终止时间戳
        :param stp_step: int, 时间步
        :param cols2serialize: list of strs, 需要按照'time'字段进行序列化处理的字段.

        Note:
                1. 待处理的数据表必需含有以"time"字段标记的时间戳;
                2. 数据表中其他字段值必需为int或float类型;

        Example:
        ------------------------------------------------------------
        # %% 载入数据
        import matplotlib.pyplot as plt
        import sys
        import os

        sys.path.append('../..')

        from lib import proj_dir

        data = pd.read_csv(os.path.join(proj_dir, 'data/provided/weather/raw_records.csv'))

        # 查看数据时间戳连续情况
        # 可以发现原始数据时间戳不连续
        plt.plot(list(data['time']))

        # %% 时间戳连续化处理
        hr = 3600
        dts = DataTemporalSerialization(data, data['time'].min(), data['time'].max(), hr)
        ------------------------------------------------------------
        """
        # 数据格式检查
        # data中必须含有'time'字段.
        try:
            assert 'time' in data.columns
        except Exception:
            raise ValueError('data does not have the field "time"')

        # data中所有值类型必须为int或float.
        for col in data.columns:
            dtype = str(data[col].dtypes)

            if ('int' not in dtype) & ('float' not in dtype):
                raise ValueError(
                    'Value type of the column "{}" is "{}", not "int" or "float", cannot continue.'.format(col, dtype))

        self.expected_stps_list = list(
            np.arange(start_stp, end_stp + stp_step, stp_step))
        self.exist_stps_list = data['time'].tolist()
        self.exist_stps_list.sort(reverse=False)  # 升序排列
        self.stp_step = stp_step
        self.data = data.copy().drop_duplicates(['time']).sort_values(
            by='time', ascending=True)  # 待处理数据按照time去重并升序排列

        if cols2serialize is not None:
            self.data = self.data[['time'] + cols2serialize]

    def serialize_time_stamps(self, categorical_cols: list = None, insert_values: bool = True) -> (pd.DataFrame, int):
        """
        时间戳连续化.
        :param categorical_cols: list, 指定为类别型变量的字段
        :param insert_values: bool, 是否对缺失值进行填补

        Example:
        ------------------------------------------------------------
        data_srlzd, miss_n = dts.temporal_serialize()

        # 可以发现处理后数据时间戳连续
        plt.plot(list(data_srlzd['time']))
        ------------------------------------------------------------
        """
        exist_data = self.data.copy()  # 备份原数据, 之后的数据处理以此为准.
        cols = exist_data.columns

        stps2fill = list(
            set(self.expected_stps_list).difference(self.exist_stps_list))
        miss_n = len(stps2fill)
        # print('miss_n = {}'.format(miss_n))

        for i in range(miss_n):
            # 显示进度.
            # if (i + 1) % 100 == 0:
            # 	print('DTS Proceeding: {}'.format(str(int(i / miss_n * 100)) + '%') + "\r", end = "")

            stp = stps2fill[i]

            # 从已有数据exist_data中提取时间戳两侧最接近的数据.
            neighbor_stps = search_nearest_neighbors_in_list(
                self.exist_stps_list, stp)
            neighbors = exist_data[exist_data.time.isin(
                neighbor_stps)]  # 获取前后相邻时间戳的数据

            if insert_values:
                # 根据时间戳距离计算权重并进行线性加权.
                if len(neighbor_stps) == 1:
                    insert_row = neighbors.copy()
                elif len(neighbor_stps) == 2:
                    weights = [neighbor_stps[1] - stp, stp - neighbor_stps[0]]
                    insert_row = (weights[0] * neighbors.iloc[0, :] +
                                  weights[1] * neighbors.iloc[1, :]) / (np.sum(weights))
                    insert_row = pd.DataFrame(insert_row).T.copy()
                else:
                    raise RuntimeError('the length of neighbors is not 1 or 2')

                insert_row.reset_index(drop=True, inplace=True)
                insert_row.loc[0, ('time',)] = stp
            else:
                insert_row = {'time': stp}
                for col in cols:
                    if col != 'time':
                        insert_row.update({col: np.nan})
                insert_row = pd.DataFrame.from_dict(
                    insert_row, orient='index').T

            if categorical_cols is not None:
                for col in categorical_cols:
                    insert_row.loc[0, (col,)] = neighbors.iloc[0][col]

            self.data = self.data.append(insert_row, ignore_index=True)

        # 时间戳筛选.
        # 由于原始数据中可能包含不在期望时间戳内的其他时间戳, 需要进行筛选和剔除.
        self.data = self.data[self.data.time.isin(self.expected_stps_list)]

        # 时间戳去重.
        # print('DTS drop_duplicates "time"')
        self.data = self.data.drop_duplicates(['time'])

        # 按照时间戳由小到大顺序排序.
        # print('DTS sort "time"\n')
        self.data = self.data.sort_values(by=['time'], ascending=True)

        # 重设index.
        self.data.reset_index(drop=True, inplace=True)

        return self.data, miss_n
