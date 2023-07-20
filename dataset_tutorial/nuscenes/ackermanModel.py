# -*- coding: UTF-8 -*-
import os.path

import numpy as np
import matplotlib.pyplot as plt
from nuscenes.nuscenes import NuScenes
from nuscenes.can_bus.can_bus_api import NuScenesCanBus

from utils.smooth_vel import CubicSplineSmooth,MoveAverage,MoveAverageWithExpWeight
# Project ：SLAMBox 
# File    ：ackermanModel.py
# Author  ：fzhiheng
# Date    ：2023/7/18 下午8:49


"""
    Ackerman模型
"""

if __name__ == '__main__':
    dataset_root = "./v1.0-mini"  # 数据集路径

    # 方向盘的转角获得前轮的转角还需要一个转向比
    # nuscenes使用的雷诺电动车的转向比为 <未知>，初步搜索到的是16.6:1，官方文档里暂时没有找到
    steer_radio = 16.6 # 转向角
    wheel_base = 2.588 # 前后轮轴距

    nusc = NuScenes(version='v1.0-mini', dataroot=dataset_root, verbose=True)
    nusc_can = NuScenesCanBus(dataroot=dataset_root)

    # 显示车辆的转向角
    for scene in nusc.scene:
        scene_name = scene['name']
        print(scene_name)
        # nusc_can.print_all_message_stats(scene_name)
        chassis = nusc_can.get_messages(scene_name, 'zoe_veh_info')
        steer_raw = np.array([m['steer_raw'] for m in chassis])
        steer_corrected = np.array([m['steer_corrected'] for m in chassis])
        steer_offset_can = np.array([m['steer_offset_can'] for m in chassis])
        chassis_time = np.array([m['utime'] for m in chassis]) / 1e6

        feedback = nusc_can.get_messages(scene_name, 'steeranglefeedback')
        steer_feedback = np.array([m['value'] for m in feedback])
        steer_feedback_degree = steer_feedback * 180 / np.pi
        feedback_time = np.array([m['utime'] for m in feedback]) / 1e6

        # 查看转向角
        # plt.title(scene_name)
        # plt.plot(chassis_time-chassis_time[0],steer_raw, color='red')
        # plt.plot(chassis_time-chassis_time[0],steer_corrected, color='green')
        # plt.plot(chassis_time-chassis_time[0],steer_offset_can, color='blue')
        # plt.plot(feedback_time-chassis_time[0],steer_feedback_degree, color='black')
        # plt.legend(['steer_raw', 'steer_corrected', 'steer_offset_can', 'steer_feedback'])
        # plt.show()

        # 获取车轮的速度
        # FL_wheel_speed = np.array([m['FL_wheel_speed'] for m in chassis])
        # FR_wheel_speed = np.array([m['FR_wheel_speed'] for m in chassis])
        RR_wheel_speed = np.array([m['RR_wheel_speed'] for m in chassis])
        RL_wheel_speed = np.array([m['RL_wheel_speed'] for m in chassis])

        vx = (RR_wheel_speed + RL_wheel_speed) / 2
        radius = 0.305  # Known Zoe wheel radius in meters.
        circumference = 2 * np.pi * radius
        vx *= (circumference / 60)

        vy = np.zeros_like(vx)
        wheel_steer = steer_corrected * np.pi / (180 * steer_radio)  # 这里先使用底盘中的steer_corrected，其实应该使用steer_feedback
        w_wheel= vx * np.tan(wheel_steer) / wheel_base  # 由阿克曼模型得到的角速度

        # 和IMU得到的角速度做一个对比
        imu = nusc_can.get_messages(scene_name, 'ms_imu')
        imu_time = np.array([m['utime'] for m in imu])
        imu_time = imu_time/1e6
        rotation_rate = np.array([m['rotation_rate'] for m in imu])
        w_imu = rotation_rate[:,-1]

        plt.title(scene_name)
        plt.plot(imu_time-chassis_time[0],w_imu, color='orange')
        plt.plot(chassis_time - chassis_time[0], w_wheel, color='green')
        plt.legend(['imu','ackerman'])
        plt.show()

























        # print(chassis[0].keys())
        # break
