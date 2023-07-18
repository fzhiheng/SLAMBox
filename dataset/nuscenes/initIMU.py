# -*- coding: UTF-8 -*-
import os.path

import numpy as np
import matplotlib.pyplot as plt
from nuscenes.nuscenes import NuScenes
from nuscenes.can_bus.can_bus_api import NuScenesCanBus

from utils.smooth_vel import CubicSplineSmooth,MoveAverage,MoveAverageWithExpWeight

# Project ：SLAMBox 
# File    ：initIMU.py
# Author  ：fzhiheng
# Date    ：2023/7/13 上午9:35

"""
IMU初始化测试
"""


if __name__ == '__main__':
    dataset_root = "./v1.0-mini" # 数据集路径

    nusc = NuScenes(version='v1.0-mini', dataroot=dataset_root, verbose=True)
    nusc_can = NuScenesCanBus(dataroot=dataset_root)

    for scene in nusc.scene:
        scene_name = scene['name']
        print(scene_name)
        # nusc_can.print_all_message_stats(scene_name)
        # 这里得到的信息是全部序列的信息，不是sampel的信息
        wheel_ = nusc_can.get_messages(scene_name, 'zoe_veh_info')
        imu_ = nusc_can.get_messages(scene_name, 'ms_imu')

        # 获取车辆的纵向加速度和横向加速度
        wheel_time = np.array([m['utime'] for m in wheel_])
        longitudinal_accel = np.array([m['longitudinal_accel'] for m in wheel_])
        transversal_accel = np.array([m['transversal_accel'] for m in wheel_])

        # 获取IMU的三轴加速度
        imu_time = np.array([m['utime'] for m in imu_])
        linear_accel = np.array([m['linear_accel'] for m in imu_])

        # 计算IMU和Wheel的时间差
        td = (imu_time[0]-wheel_time[0])/1e6

        # 将IMU和Wheel的时间转换成s
        wheel_time = wheel_time/1e6
        imu_time = imu_time/1e6

        # IMU初始化使用前0.1s的数据（图像帧率大于为12）
        cur_time = imu_time[0]
        last_time = imu_time[0] + 0.1
        imu_sample = np.where((imu_time >= cur_time) & (imu_time <= last_time))[0]
        imu_sample_data = linear_accel[imu_sample]

        wheel_sample = np.where((wheel_time >= cur_time) & (wheel_time <= last_time))[0]
        wheel_sample_data1 = longitudinal_accel[wheel_sample]
        wheel_sample_data2 = transversal_accel[wheel_sample]

        # 计算IMU和Wheel的加速度
        imu_accel = np.mean(imu_sample_data, axis=0)
        wheel_accel1 = np.mean(wheel_sample_data1, axis=0)
        wheel_accel2 = np.mean(wheel_sample_data2, axis=0)
        wheel_acc = np.array([wheel_accel1, wheel_accel2, 0])
        real_acc = imu_accel - wheel_acc



        print(imu_accel, np.linalg.norm(imu_accel))
        print(wheel_acc, np.linalg.norm(wheel_acc))
        print(real_acc, np.linalg.norm(real_acc))







