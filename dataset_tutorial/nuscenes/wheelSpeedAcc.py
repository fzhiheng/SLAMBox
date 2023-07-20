import os.path

import numpy as np
import matplotlib.pyplot as plt
from nuscenes.nuscenes import NuScenes
from nuscenes.can_bus.can_bus_api import NuScenesCanBus

from utils.smooth_vel import CubicSplineSmooth,MoveAverage,MoveAverageWithExpWeight

"""
将nuscenes场景中wheel数据进行平滑，并使用滑动窗口计算加速度
"""

def show_speed(wheel: np.ndarray, name=None):
    row = wheel.shape[-1]
    titles = ['x_speed', 'y_speed', 'z_speed']
    fig, ax = plt.subplots(row - 1, 1)
    fig.tight_layout()
    for i in range(1, row):
        ax[i - 1].plot(wheel[:, 0], wheel[:, i])
        ax[i - 1].set_title(titles[i - 1])
    plt.show()


# 滑动窗口计算加速度
def calculate_acceleration(data):

    # 计算时间差值
    time_diff = np.diff(data[:, 0])
    mean_time_diff = np.mean(time_diff)

    # 计算加速度
    left = 0
    right = 0
    accs = []
    times = []
    while right<len(data):
        while(data[right, 0] - data[left, 0]<mean_time_diff):
            right += 1
            if right>=len(data):
                break
        if right >= len(data):
            break
        curr_time_diff = data[right, 0] - data[left, 0]
        acc = (data[right, 1:] - data[left, 1:])/curr_time_diff
        accs.append(acc[None,:])
        times.append(np.array([[data[right, 0]]]))
        left += 1
    times = np.concatenate(times,axis=0)
    accs = np.concatenate(accs,axis=0)

    return np.concatenate([times, accs],axis=1)


# 简单差分计算加速度
def vel2acc(wheel_speed):
    dt = wheel_speed[1:, 0] - wheel_speed[:-1, 0]
    dt_speed = wheel_speed[1:, 1] - wheel_speed[:-1, 1]
    acc = dt_speed / dt
    acc = np.concatenate([wheel_speed[:-1, :1], acc[..., None]], axis=-1)
    return acc


if __name__ == '__main__':
    dataset_root = "./v1.0-mini" # 数据集路径
    images_root = "dataset_tutorial/nuscenes/output" # 结果图片保存路径

    nusc = NuScenes(version='v1.0-mini', dataroot=dataset_root, verbose=True)
    nusc_can = NuScenesCanBus(dataroot=dataset_root)

    for scene in nusc.scene:
        scene_name = scene['name']
        print(scene_name)
        # nusc_can.print_all_message_stats(scene_name)
        # 这里得到的信息是全部序列的信息，不是sampel的信息
        wheel_ = nusc_can.get_messages(scene_name, 'zoe_veh_info')

        wheel_speed = np.array([(m['utime'], m['FL_wheel_speed']) for m in wheel_])
        wheel_speed[:,0] = (wheel_speed[:,0] - wheel_speed[0,0]) / 1e6
        radius = 0.305  # Known Zoe wheel radius in meters.
        circumference = 2 * np.pi * radius
        wheel_speed[:,1:] *= (circumference / 60)


        # 速度不同的平滑方式
        wheel_cub = CubicSplineSmooth(wheel_speed)
        wheel_move3 = MoveAverage(wheel_speed)
        wheel_move10 = MoveAverage(wheel_speed, 10)
        wheel_exp_weight = MoveAverageWithExpWeight(wheel_speed, 0.2)

        # 速度图像, 取消注释即可
        # legend = []
        # plt.plot(wheel_speed[:, 0], wheel_speed[:, 1])
        # legend.append('Wheel speed')
        #
        # # plt.plot(wheel_cub[:, 0], wheel_cub[:, 1])
        # # plt.plot(wheel_move3[:, 0], wheel_move3[:, 1])
        # #
        # # plt.plot(wheel_move10[:, 0], wheel_move10[:, 1])
        # # legend.append('wheel_move10')
        #
        # # plt.plot(wheel_exp_weight[:, 0], wheel_exp_weight[:, 1])
        # # legend.append('wheel_exp')
        #
        # plt.xlabel('Time in s')
        # plt.ylabel('Speed in m/s')
        # plt.title(f'{scene_name}')
        # plt.legend(legend)

        # 图像保存路径
        # vel_save_root = os.path.join(images_root, "vel")
        # os.makedirs(vel_save_root, exist_ok=True)
        # plt.savefig(os.path.join(images_root, f'{scene_name}.png'))

        # 加速度图像
        legend = []
        plt.plot(calculate_acceleration(wheel_speed)[:, 0], calculate_acceleration(wheel_speed)[:, 1])
        legend.append('Wheel speed')

        plt.plot(calculate_acceleration(wheel_move10)[:, 0], calculate_acceleration(wheel_move10)[:, 1],alpha=0.5)
        legend.append('wheel_move10')

        plt.xlabel('Time in s')
        plt.ylabel('Acc in m/s')
        plt.title(f'{scene_name}')
        plt.legend(legend)

        acc_save_root = os.path.join(images_root, "acc")
        os.makedirs(acc_save_root,exist_ok=True)
        plt.savefig(os.path.join(acc_save_root, f'{scene_name}.png'))

        plt.show()
