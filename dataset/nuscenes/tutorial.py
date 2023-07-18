import os
import shutil
import json
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion


# 为使用colmap制作的图片数据集
def save_images(dataset_root, save_to_root):
    nusc = NuScenes(version='v1.0-mini', dataroot=dataset_root, verbose=True)
    for scene in nusc.scene:
        # make save root
        scene_name = scene['name']
        save_scene_root = os.path.join(save_to_root, scene_name)
        os.makedirs(save_scene_root, exist_ok=True)

        first_sample_token = scene['first_sample_token']
        sample = nusc.get('sample', first_sample_token)
        sensor = nusc.get('sample_data', sample['data']['CAM_FRONT'])

        # 写入内参与外参
        sensor_calib = nusc.get('calibrated_sensor', sensor['calibrated_sensor_token'])
        translation = sensor_calib['translation']
        rotation = sensor_calib['rotation']
        camera_intrinsic = sensor_calib['camera_intrinsic']
        extrinsic_matrix = transform_matrix(np.array(translation), Quaternion(rotation)).tolist()
        intrinsic = {"fx": camera_intrinsic[0][0], "fy": camera_intrinsic[1][1], "cx": camera_intrinsic[0][2],
                     "cy": camera_intrinsic[1][2], "translation": translation, "rotation": rotation,
                     "matrix": extrinsic_matrix}

        with open(os.path.join(save_scene_root, "intrinsic.json"), "w", encoding='utf8') as fp:
            json.dump(intrinsic, fp, ensure_ascii=False, indent=4)

        def save_image(filename):
            src_path = os.path.join(dataset_root, filename)
            dst_path = os.path.join(save_scene_root, filename)
            dst_root = os.path.dirname(dst_path)
            dst_file = os.path.basename(dst_path)
            dst_file = str(dst_file.split("_")[-1])
            os.makedirs(dst_root, exist_ok=True)
            shutil.copy(src_path, os.path.join(dst_root, os.path.join(dst_root, dst_file)))

        filename = sensor["filename"]
        save_image(filename)
        while (sample["next"]):
            sample = nusc.get('sample', sample['next'])
            sensor = nusc.get('sample_data', sample['data']['CAM_FRONT'])
            filename = sensor["filename"]
            save_image(filename)


def get_all_image_name(dataset_root, save_to_root):
    """
    获取所有图片的名称，注意这里不是sample图片，而是全部的图片
    :param dataset_root:
    :param save_to_root:
    :return:
    """

    camera_names = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']
    nusc = NuScenes(version='v1.0-mini', dataroot=dataset_root, verbose=True)
    for scene in nusc.scene:
        # make save root
        scene_name = scene['name']
        save_scene_root = os.path.join(save_to_root, scene_name)
        os.makedirs(save_scene_root, exist_ok=True)

        print(f"scenen_name: {scene_name}")

        for camera in camera_names:
            images_name = []
            first_sample_token = scene['first_sample_token']
            sample = nusc.get('sample', first_sample_token)
            sensor = nusc.get('sample_data', sample['data'][camera])
            images_name.append(sensor["filename"])
            while (sensor["next"]):
                # 注意这里的sensor["next"]，而不是sample["next"]
                sensor = nusc.get('sample_data', sensor['next'])
                images_name.append(sensor["filename"])

            with open(os.path.join(save_scene_root, camera + ".txt"), "w", encoding='utf8') as fp:
                tt = "\n".join(images_name)
                fp.write(tt)


if __name__ == '__main__':
    dataset_root = "./v1.0-mini"  # 数据集路径

    # 生成colmap使用的数据格式
    # save_images(dataset_root, "./colmap")

    # 获取所有图片的名称
    # get_all_image_name(dataset_root, "./image_name")

    # 使用范例
    nusc = NuScenes(version='v1.0-mini', dataroot=os.path.abspath(dataset_root), verbose=True)
    print(nusc.list_scenes())
    print("------------------------")
    scene_camera_matrix = dict()
    scene_camera_intrinsic = dict()
    for scene in nusc.scene:
        first_sample_token = scene['first_sample_token']
        sample = nusc.get('sample', first_sample_token)
        print(f"first_sample_token: {first_sample_token}")

        # 获取前视相机传感器的外参
        camera_token = sample['data']['CAM_FRONT']
        sensor = nusc.get('sample_data', camera_token)
        filename = sensor["filename"]
        print(f"filename: {filename}")
        sensor_calib = nusc.get('calibrated_sensor', sensor['calibrated_sensor_token'])

        translation = sensor_calib['translation']
        rotation = sensor_calib['rotation']
        camera_intrinsic = sensor_calib['camera_intrinsic']
        print(translation)
        print(rotation)
        print(Quaternion(rotation))

        extrinsic_matrix = transform_matrix(np.array(translation), Quaternion(rotation))
        scene_camera_matrix[scene['name'][-4:]] = extrinsic_matrix.flatten()
        scene_camera_intrinsic[scene['name'][-4:]] = {"fx": camera_intrinsic[0][0],
                                                      "fy": camera_intrinsic[1][1],
                                                      "cx": camera_intrinsic[0][2],
                                                      "cy": camera_intrinsic[1][2]}

        print('Scene ID:', scene['name'])
        print('-----------------------------------')
