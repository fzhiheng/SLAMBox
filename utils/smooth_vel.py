#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import numpy as np
from scipy.interpolate import CubicSpline

# Project ：SLAMBox
# File    ：smooth_vel.py
# author  ：fzhiheng
# date    ：2023/7/12 下午3:41


"""
速度平滑方案
"""

# 三次样条插值平滑
def CubicSplineSmooth(wheel: np.ndarray):
    assert wheel.ndim == 2 and wheel.shape[-1] == 2

    # 创建三次样条插值对象
    time = wheel[:, 0]
    speed = wheel[:, 1]
    cs = CubicSpline(time, speed)

    # 在原始时间序列数据的范围内生成平滑后的数据
    time_smooth = np.linspace(time[0], time[-1], num=len(time), endpoint=True)
    speed_smooth = cs(time_smooth)

    wheel_smooth = np.concatenate([time_smooth[..., None], speed_smooth[..., None]], axis=-1)
    return wheel_smooth

# 移动平均平滑
def MoveAverage(wheel: np.ndarray, window_size=3):
    time = wheel[:, :1]
    speeds = wheel[:, 1:]
    # 使用numpy的convolve函数实现移动平均
    weights = np.repeat(1.0, window_size) / window_size
    speeds_smooth = []
    for speed in speeds.T:
        speed_smooth = np.convolve(speed, weights, 'valid')  # 'valid'选项表示不使用填充
        speeds_smooth.append(speed_smooth[..., None])

    speeds_smooth = np.concatenate(speeds_smooth, axis=-1)

    # 可以将y_smooth插入到x数组中，使其与原始数据对齐
    time_smooth = time[window_size // 2: len(time) - (window_size - 1) // 2]  # 头尾的数据被删去了

    wheel_smooth = np.concatenate([time_smooth, speeds_smooth], axis=-1)
    return wheel_smooth

# 移动加权平滑
def MoveAverageWithExpWeight(wheel: np.ndarray, alpha=0.2):
    time = wheel[:, :1]
    speed = wheel[:, 1:]
    # 初始化平滑后的第一个数据点
    speed_smooth = [speed[0]]

    # 逐个处理每个数据点
    for i in range(1, len(speed)):
        # 计算当前数据点的平滑值
        speed_smooth.append(alpha * speed[i] + (1 - alpha) * speed_smooth[i - 1])

    wheel_smooth = np.concatenate([time, np.array(speed_smooth)], axis=-1)
    return wheel_smooth
