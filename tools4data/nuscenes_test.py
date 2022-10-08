import numpy as np
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion

data_root = './data/nuscenes'
def get_dataset_info1(nusc):
    scene_num = len(nusc.scene)
    sample_num = 0
    ann_num = 0

    for scene in nusc.scene:
        sample = None
        while True:
            if sample is None:
                sample = nusc.get('sample', scene['first_sample_token'])

            sample_num += 1
            ann_num += len(sample['anns'])

            if sample['next'] != '':
                sample = nusc.get('sample', sample['next'])
            else:
                break

    print('====== Start from scene')
    print('Scene Num: %d\nSample Num: %d\nAnnotation Num: %d' % (scene_num, sample_num, ann_num))


def get_dataset_info2(nusc):
    sample_num = len(nusc.sample)
    ann_num = 0

    scene_tokens = set()
    for sample in nusc.sample:
        ann_num += len(sample['anns'])

        scene = nusc.get('scene', sample['scene_token'])
        scene_tokens.add(scene['token'])
    scene_num = len(scene_tokens)

    print('====== Start from sample')
    print('Scene Num: %d\nSample Num: %d\nAnnotation Num: %d' % (scene_num, sample_num, ann_num))


def get_dataset_info3(nusc):
    ann_num = len(nusc.sample_annotation)

    scene_tokens = set()
    sample_tokens = set()
    for ann in nusc.sample_annotation:
        sample = nusc.get('sample', ann['sample_token'])
        sample_tokens.add(sample['token'])

        scene = nusc.get('scene', sample['scene_token'])
        scene_tokens.add(scene['token'])
    scene_num = len(scene_tokens)
    sample_num = len(sample_tokens)

    print('====== Start from annotation')
    print('Scene Num: %d\nSample Num: %d\nAnnotation Num: %d' % (scene_num, sample_num, ann_num))


def xx(nusc):

    for sample in nusc.sample:
        if sample['token'] != '3e8750f331d7499e9b5123e9eb70f2e2':
            continue
        print('sample token: ', sample['token'])
        print('scene token: ', sample['scene_token'])
        # 获取采样的标注信息
        print(len(sample['anns']))
        annotation_token = sample['anns'][0]
        # 获取采样的激光雷达传感器信息
        lidar_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        # 可视化激光雷达采样信息及标注信息
        # nusc.render_sample_data(lidar_data['token'])

        # 标注真值到激光坐标系
        ann = nusc.get('sample_annotation', annotation_token)
        calib_data = nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
        ego_data = nusc.get('ego_pose', lidar_data['ego_pose_token'])
        # global frame
        center = np.array(ann['translation'])
        orientation = np.array(ann['rotation'])
        # 从global frame转换到ego vehicle frame
        quaternion = Quaternion(ego_data['rotation']).inverse
        center -= np.array(ego_data['translation'])
        center = np.dot(quaternion.rotation_matrix, center)
        orientation = quaternion * orientation
        # 从ego vehicle frame转换到sensor frame
        quaternion = Quaternion(calib_data['rotation']).inverse
        center -= np.array(calib_data['translation'])
        center = np.dot(quaternion.rotation_matrix, center)
        orientation = quaternion * orientation

        print(center)
        print(orientation)



if __name__ == '__main__':
    nusc = NuScenes(version='v1.0-mini',
                    dataroot=data_root,
                    verbose=True)
    # get_dataset_info1(nusc)
    # get_dataset_info2(nusc)
    # get_dataset_info3(nusc)
    xx(nusc)
