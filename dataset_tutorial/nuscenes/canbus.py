
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.can_bus.can_bus_api import NuScenesCanBus



if __name__ == '__main__':
    dataset_root = "/home/fzh/MyWork/dataset_tutorial/nuscenes/v1.0-mini"
    nusc = NuScenes(version='v1.0-mini', dataroot=dataset_root, verbose=True)
    nusc_can = NuScenesCanBus(dataroot=dataset_root)

    for scene in nusc.scene:
        scene_name = scene['name']
        # nusc_can.print_all_message_stats(scene_name)

        # 获取不同传感器的数据
        imu = nusc_can.get_messages(scene_name, 'ms_imu')
        wheel = nusc_can.get_messages(scene_name, 'zoe_veh_info')
        zoesensors = nusc_can.get_messages(scene_name, 'zoesensors')
        print(imu[0].keys())

        # 画出IMU三轴加速度图像
        message_name = 'ms_imu'
        key_name = 'linear_accel'
        nusc_can.plot_message_data(scene_name, message_name, key_name)

        # 获取轮速计四个轮子的速度
        FL_wheel_speed = np.array([m['FL_wheel_speed'] for m in wheel])
        FR_wheel_speed = np.array([m['FR_wheel_speed'] for m in wheel])
        RL_wheel_speed = np.array([m['RL_wheel_speed'] for m in wheel])
        RR_wheel_speed = np.array([m['RR_wheel_speed'] for m in wheel])

