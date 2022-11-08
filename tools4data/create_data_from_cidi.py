from copy import deepcopy
import json
import random
import uuid
import transform_utils as tu

import pickle
import os
from os import path as osp
import numpy as np

import mmcv
from mmcv import track_iter_progress
from mmcv.ops import roi_align
from pycocotools.coco import COCO

from mmdet3d.core.bbox import box_np_ops as box_np_ops
from mmdet3d.datasets import build_dataset
from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps


# 初始数据
infos = []
metadata = {"version": "v1.0-mini"}
init_cam = {
            "data_path": "",
            "type": "",
            "sample_data_token": "",
            "sensor2ego_translation": [],
            "sensor2ego_rotation": [],
            "ego2global_translation": [],
            "ego2global_rotation": [],
            "timestamp": -1,
            "sensor2lidar_rotation": [],
            "sensor2lidar_translation": [],
            "camera_intrinsics": []
            }
init_info = {
            "lidar_path": "",
            "token": "",
            "sweeps": [],
            "cams": {},
            "lidar2ego_translation": [],
            "lidar2ego_rotation": [],
            "ego2global_translation": [],
            "ego2global_rotation": [],
            "timestamp": -1,
            "location": "boston-seaport",
            "gt_boxes": [],
            "gt_names": [],
            "gt_velocity": [],
            "num_lidar_pts": [],
            "num_radar_pts": [],
            "valid_flag": []
            }


def make_data(file_list, nuscenes_infos, cam_calib_info, cams_channel_list, out_name):
    ext_name = 'pcd.bin'
    infos = []

    for ids, file_name in enumerate(file_list):

        info = deepcopy(init_info)
        nuscenes_info = nuscenes_infos['infos'][ids]

        # 加载PCD路径
        info["lidar_path"] = os.path.join(data_dir, pcd_dir_name, file_name)
        # 生成UUID替换token
        token = uuid.uuid1()
        token = str(token)
        info["token"] = token
        
        # 假设激光雷达在ego中心
        info["lidar2ego_translation"] = [0.0, 0.0, 0.0]
        # 假设激光雷达在相对ego无旋转  wxyz   
        info["lidar2ego_rotation"] = [1,0,0,0]
        # lidir相对于car坐标系绕z轴旋转-90度。 wxyz
        # info["lidar2ego_rotation"] = [ 0.70710678  ,0.          ,0.         ,-0.70710678]
        
        # 车辆信息套用nuscenes的原始数据
        info["ego2global_translation"] = nuscenes_info["ego2global_translation"]
        info["ego2global_rotation"] = nuscenes_info["ego2global_rotation"]
        
        # 时间信息 从文件名获取
        info["timestamp"] = int(file_name.replace(ext_name, '').replace('.', ''))
        
        # 地图信息套用nuscenes的原始数据
        info["location"] = nuscenes_info["location"]
        
        # 解析摄像头信息
        cams = {}
        img_file_name = file_name.replace(ext_name, 'jpg')

        for channel in cams_channel_list:
            cam_info = deepcopy(init_cam)
            # 加载IMG路径
            cam_info["data_path"] = os.path.join(data_dir, 'image', channel, img_file_name)
            cam_info["type"] = channel
            # 生成UUID替换token
            token = uuid.uuid1()
            token = str(token)
            cam_info["sample_data_token"] = token
            # 时间信息 从文件名获取
            cam_info["timestamp"] = int(file_name.replace(ext_name, '').replace('.', ''))

            # 获取摄像头标定信息
            calib_info = cam_calib_info[channel]
            cam_info["sensor2ego_rotation"] = calib_info["sensor2ego_rotation"]
            cam_info["sensor2ego_translation"] = calib_info["sensor2ego_translation"]
            cam_info["sensor2lidar_rotation"] = calib_info["sensor2lidar_rotation"]
            cam_info["sensor2lidar_translation"] = calib_info["sensor2lidar_translation"]
            cam_info["camera_intrinsics"] = calib_info["camera_intrinsics"]
            
            # 车辆信息套用和激光一样的参数，假设没有运动差异
            cam_info["ego2global_translation"] = info["ego2global_translation"]
            cam_info["ego2global_rotation"] = info["ego2global_rotation"]

            # 赋值
            cams[channel] = cam_info
        info["cams"] = cams

        # 读取标注文件
        label_file = os.path.join(data_dir, label_dir_name, file_name.replace(ext_name, 'json'))
        with open(label_file, 'r') as f:
            anns = json.load(f)
        # print(len(anns))
        gt_boxes = []
        gt_xyzs = []
        gt_wlhs = []
        gt_rzs = []
        gt_names = []
        gt_velocity = []
        num_lidar_pts = []
        num_radar_pts = []
        valid_flag = []

        for ann in anns:
            obj_type = ann['obj_type']
            # 小写化name
            obj_type = obj_type.lower()
            obj_position = ann['psr']['position']
            obj_xyz = [obj_position['x'], obj_position['y'], obj_position['z']]
            obj_scale = ann['psr']['scale']
            # label.car:标注的尺寸是关于xyz的，nuscenes中的size是w,l,h顺序，所以这里调整顺序
            obj_wlh = [obj_scale['y'], obj_scale['x'], obj_scale['z']]
            # label.imu:在imu下的label就是wlh顺序，所以直接赋值不需要转。
            # obj_wlh = [obj_scale['x'], obj_scale['y'], obj_scale['z']]
            obj_rotation = ann['psr']['rotation']
            obj_rz = obj_rotation['z']  # 标注出来的是弧度
            gt_xyzs.append(obj_xyz)
            gt_wlhs.append(obj_wlh)
            gt_rzs.append(obj_rz)
            gt_names.append(obj_type)
            gt_velocity.append([0.06, 0.06])
            num_lidar_pts.append(6)
            num_radar_pts.append(6)
            valid_flag.append(True)

        # 添加gt数据
        locs = np.array(gt_xyzs).reshape(-1, 3)
        dims = np.array(gt_wlhs).reshape(-1, 3)
        rots = np.array(gt_rzs).reshape(-1, 1)
        gt_boxes = np.concatenate([locs, dims, -rots - np.pi / 2], axis=1)
        info["gt_boxes"] = gt_boxes
        info["gt_names"] = np.array(gt_names)
        info["gt_velocity"] = np.array(gt_velocity).reshape(-1,2)
        info["num_lidar_pts"] = np.array(num_lidar_pts)
        info["num_radar_pts"] = np.array(num_radar_pts)
        info["valid_flag"] = np.array(valid_flag, dtype=bool).reshape(-1)

        # 挑选部分类别mask
        CC = ['car', 'truck', 'pedestrian']
        mask = [True if name in CC else False for name in info["gt_names"] ]
        info["gt_boxes"] = info["gt_boxes"][mask]
        info["gt_names"] = info["gt_names"][mask]
        info["gt_velocity"] = info["gt_velocity"][mask]
        info["num_lidar_pts"] = info["num_lidar_pts"][mask]
        info["num_radar_pts"] = info["num_radar_pts"][mask]
        info["valid_flag"] = info["valid_flag"][mask]
        # print('gt_names ', len(info["gt_names"]), info["gt_names"])

        # kevin 提取原始数据
        print('src ann data')
        if file_name.replace('.'+ext_name, '')== '1639641371.699909925':
            save_gt_boxes = np.concatenate([locs, dims, rots], axis=1)
            print(save_gt_boxes.shape, save_gt_boxes.dtype)
            save_gt_boxes = save_gt_boxes[mask]
            print(save_gt_boxes.shape, save_gt_boxes.dtype)
            lidar_name = file_name.replace('.'+ext_name, '')
            ann_out = 'data/nuscenes_cidi_byd/data4test/0.{}.ann.bin'.format(lidar_name)
            print(ann_out)
            save_gt_boxes.tofile(ann_out)
        # =================
        
        # kevin 提取转换后的存储的数据
        print('saved pkl data')
        if file_name.replace('.'+ext_name, '')== '1639641371.699909925':
            save_gt_boxes = info["gt_boxes"]
            print(save_gt_boxes.shape, save_gt_boxes.dtype)
            lidar_name = file_name.replace('.'+ext_name, '')
            ann_out = 'data/nuscenes_cidi_byd/data4test/1.{}.pkl.bin'.format(lidar_name)
            print(ann_out)
            save_gt_boxes.tofile(ann_out)
        # =================

        # 强制输出6个cams
        # src_cams = deepcopy(info["cams"])
        # if len(src_cams) == 4:
        #     src_cams['75'] = src_cams['73']
        #     src_cams['76'] = src_cams['74']
        # info["cams"] = src_cams
        # ==============

        infos.append(info)


    # 输出数据字典
    data_dict = {
        "infos": infos,
        "metadata": metadata
    }

    out_path = os.path.join(data_dir, out_name)
    mmcv.dump(data_dict, file=out_path)
    print('{} saved.'.format(out_path))


def create_groundtruth_database(
    dataset_class_name,
    data_path,
    info_prefix,
    info_path=None,
    mask_anno_path=None,
    used_classes=None,
    database_save_path=None,
    db_info_save_path=None,
    relative_path=True,
    add_rgb=False,
    lidar_only=False,
    bev_only=False,
    coors_range=None,
    with_mask=False,
    load_augmented=None,
):
    """Given the raw data, generate the ground truth database.

    Args:
        dataset_class_name （str): Name of the input dataset.
        data_path (str): Path of the data.
        info_prefix (str): Prefix of the info file.
        info_path (str): Path of the info file.
            Default: None.
        mask_anno_path (str): Path of the mask_anno.
            Default: None.
        used_classes (list[str]): Classes have been used.
            Default: None.
        database_save_path (str): Path to save database.
            Default: None.
        db_info_save_path (str): Path to save db_info.
            Default: None.
        relative_path (bool): Whether to use relative path.
            Default: True.
        with_mask (bool): Whether to use mask.
            Default: False.
    """
    print(f"Create GT Database of {dataset_class_name}")
    dataset_cfg = dict(
        type=dataset_class_name, dataset_root=data_path, ann_file=info_path
    )
    if dataset_class_name == "NuScenesDataset":
        if not load_augmented:
            dataset_cfg.update(
                use_valid_flag=True,
                pipeline=[
                    dict(
                        type="LoadPointsFromFile",
                        coord_type="LIDAR",
                        load_dim=5,
                        use_dim=5,
                    ),
                    dict(
                        type="LoadPointsFromMultiSweeps",
                        sweeps_num=10,
                        use_dim=[0, 1, 2, 3, 4],
                        pad_empty_sweeps=True,
                        remove_close=True,
                    ),
                    dict(
                        type="LoadAnnotations3D", with_bbox_3d=True, with_label_3d=True
                    ),
                ],
            )
        else:
            dataset_cfg.update(
                use_valid_flag=True,
                pipeline=[
                    dict(
                        type="LoadPointsFromFile",
                        coord_type="LIDAR",
                        load_dim=16,
                        use_dim=list(range(16)),
                        load_augmented=load_augmented,
                    ),
                    dict(
                        type="LoadPointsFromMultiSweeps",
                        sweeps_num=10,
                        load_dim=16,
                        use_dim=list(range(16)),
                        pad_empty_sweeps=True,
                        remove_close=True,
                        load_augmented=load_augmented,
                    ),
                    dict(
                        type="LoadAnnotations3D", with_bbox_3d=True, with_label_3d=True
                    ),
                ],
            )

    else:
        print('dataset_class_name != "NuScenesDataset"')
        exit()

    dataset = build_dataset(dataset_cfg)

    if database_save_path is None:
        database_save_path = osp.join(data_path, f"{info_prefix}_gt_database")
    if db_info_save_path is None:
        db_info_save_path = osp.join(data_path, f"{info_prefix}_dbinfos_train.pkl")
    mmcv.mkdir_or_exist(database_save_path)
    
    # 同步类别 @ nuscenes_dataset.py @ line 112
    #   用于修正KeyError在 
    # File "/home/cao.ke/Workspace/bevfusion/mmdet3d/datasets/pipelines/dbsampler.py", line 186, in filter_by_min_points
    #    for info in db_infos[name]:
    #        KeyError: 'trailer'
    CLASSES = (
        "car",
        "truck",
        # "trailer",
        # "bus",
        # "construction_vehicle",
        # "bicycle",
        # "motorcycle",
        "pedestrian",
        # "traffic_cone",
        # "barrier",
    )
    all_db_infos = dict()
    for key in CLASSES:
        all_db_infos[key] = []
    # ============

    if with_mask:
        coco = COCO(osp.join(data_path, mask_anno_path))
        imgIds = coco.getImgIds()
        file2id = dict()
        for i in imgIds:
            info = coco.loadImgs([i])[0]
            file2id.update({info["file_name"]: i})

    group_counter = 0
    for j in track_iter_progress(list(range(len(dataset)))):
        input_dict = dataset.get_data_info(j)
        dataset.pre_pipeline(input_dict)
        example = dataset.pipeline(input_dict)
        annos = example["ann_info"]
        image_idx = example["sample_idx"]

        # kevin
        # if image_idx == 'a3623a50-4068-11ed-ae4d-8cec4b5c0c34':
        #     print(image_idx)

        points = example["points"].tensor.numpy()
        gt_boxes_3d = annos["gt_bboxes_3d"].tensor.numpy()
        names = annos["gt_names"]
        group_dict = dict()
        if "group_ids" in annos:
            group_ids = annos["group_ids"]
        else:
            group_ids = np.arange(gt_boxes_3d.shape[0], dtype=np.int64)
        difficulty = np.zeros(gt_boxes_3d.shape[0], dtype=np.int32)
        if "difficulty" in annos:
            difficulty = annos["difficulty"]

        num_obj = gt_boxes_3d.shape[0]
        point_indices = box_np_ops.points_in_rbbox(points, gt_boxes_3d)

        if with_mask:
            # prepare masks
            gt_boxes = annos["gt_bboxes"]
            img_path = osp.split(example["img_info"]["filename"])[-1]
            if img_path not in file2id.keys():
                print(f"skip image {img_path} for empty mask")
                continue
            img_id = file2id[img_path]
            kins_annIds = coco.getAnnIds(imgIds=img_id)
            kins_raw_info = coco.loadAnns(kins_annIds)
            kins_ann_info = _parse_coco_ann_info(kins_raw_info)
            h, w = annos["img_shape"][:2]
            gt_masks = [_poly2mask(mask, h, w) for mask in kins_ann_info["masks"]]
            # get mask inds based on iou mapping
            bbox_iou = bbox_overlaps(kins_ann_info["bboxes"], gt_boxes)
            mask_inds = bbox_iou.argmax(axis=0)
            valid_inds = bbox_iou.max(axis=0) > 0.5

            # mask the image
            # use more precise crop when it is ready
            # object_img_patches = np.ascontiguousarray(
            #     np.stack(object_img_patches, axis=0).transpose(0, 3, 1, 2))
            # crop image patches using roi_align
            # object_img_patches = crop_image_patch_v2(
            #     torch.Tensor(gt_boxes),
            #     torch.Tensor(mask_inds).long(), object_img_patches)
            object_img_patches, object_masks = crop_image_patch(
                gt_boxes, gt_masks, mask_inds, annos["img"]
            )

        if image_idx == 'a3623a50-4068-11ed-ae4d-8cec4b5c0c34':
            print(image_idx)

        for i in range(num_obj):
            filename = f"{image_idx}_{names[i]}_{i}.bin"
            abs_filepath = osp.join(database_save_path, filename)
            rel_filepath = osp.join(f"{info_prefix}_gt_database", filename)

            # save point clouds and image patches for each object
            gt_points = points[point_indices[:, i]]
            gt_points[:, :3] -= gt_boxes_3d[i, :3]

            if with_mask:
                if object_masks[i].sum() == 0 or not valid_inds[i]:
                    # Skip object for empty or invalid mask
                    continue
                img_patch_path = abs_filepath + ".png"
                mask_patch_path = abs_filepath + ".mask.png"
                mmcv.imwrite(object_img_patches[i], img_patch_path)
                mmcv.imwrite(object_masks[i], mask_patch_path)

            with open(abs_filepath, "w") as f:
                gt_points.tofile(f)

            if (used_classes is None) or names[i] in used_classes:
                db_info = {
                    "name": names[i],
                    "path": rel_filepath,
                    "image_idx": image_idx,
                    "gt_idx": i,
                    "box3d_lidar": gt_boxes_3d[i],
                    "num_points_in_gt": gt_points.shape[0],
                    "difficulty": difficulty[i],
                }
                local_group_id = group_ids[i]
                # if local_group_id >= 0:
                if local_group_id not in group_dict:
                    group_dict[local_group_id] = group_counter
                    group_counter += 1
                db_info["group_id"] = group_dict[local_group_id]
                if "score" in annos:
                    db_info["score"] = annos["score"][i]
                if with_mask:
                    db_info.update({"box2d_camera": gt_boxes[i]})
                if names[i] in all_db_infos:
                    all_db_infos[names[i]].append(db_info)
                else:
                    all_db_infos[names[i]] = [db_info]

    for k, v in all_db_infos.items():
        print(f"load {len(v)} {k} database infos")

    with open(db_info_save_path, "wb") as f:
        pickle.dump(all_db_infos, f)


def get_cam_calib_data(data_dir, cams_channel_list):
    # 采集相机通道标定信息
    cam_calib_info = {}
    for channel in cams_channel_list:
        print(channel)

        calib_file = os.path.join(data_dir, 'calib', channel+'.json')
        with open(calib_file, 'r') as f:
            calib = json.load(f)
        calib_info = {}

        # 提取标定数据
        cam2frt_euler = np.array(calib['euler'], dtype=np.float32)
        cam2frt_euler = cam2frt_euler * np.pi / 180.0    # 角度到弧度
        cam2frt_euler = cam2frt_euler * [-1, -1, 1]
        cam2imu_trans = np.array(calib['translate'], dtype=np.float32)
        imu2car_quat_xyzw = np.array(calib['rotation'], dtype=np.float32)
        imu2car_trans = np.array(calib['translation'], dtype=np.float32)
        # kevin  对融合匹配度的测试
        # imu2car_trans = imu2car_trans * 1.2
        # ===
        cam_intrinsic = np.array(calib['intrinsic'], dtype=np.float32).reshape(3,3)
        cam_distort = calib['distort']

        # 计算相机到ego的旋转和平移 ==================================
        
        # front与imu的旋转欧拉角
        frt2imu_euler = -1 * np.pi / 2
        # 得到绕x轴旋转的旋转矩阵
        rot_euler = frt2imu_euler
        Rx = np.array([
                    [             1,                  0,                  0],
                    [             0,  np.cos(rot_euler), -np.sin(rot_euler)],
                    [             0,  np.sin(rot_euler),  np.cos(rot_euler)]])
        # 求单应性矩阵
        frt2imu_r = Rx

        # camera2imu单应性矩阵计算
        cam2frt_r = tu.euler2rotmat_YXZ_in(cam2frt_euler)
        cam2imu_r = frt2imu_r @ cam2frt_r
        cam2imu_mat = np.eye(4, dtype=np.float32)
        cam2imu_mat[:3,:3] = cam2imu_r
        cam2imu_mat[:3,3] = cam2imu_trans

        # imu2car单应性矩阵计算
        imu2car_r = tu.quaternion2rotmat_sci(imu2car_quat_xyzw)
        imu2car_mat = np.eye(4, dtype=np.float32)
        imu2car_mat[:3,:3] = imu2car_r
        imu2car_mat[:3,3] = imu2car_trans

        # car2camera单应性矩阵计算
        cam2car_mat = imu2car_mat @ cam2imu_mat
        print('extrinsic:\n', cam2car_mat)
        # =============================================================

        # 相机到ego
        sensor2ego_rotation = cam2car_mat[:3,:3]
        sensor2ego_translation = cam2car_mat[:3,3]
        sensor2ego_quat_xyzw = tu.rotmat2quaternion_sci(sensor2ego_rotation)
        q_ = sensor2ego_quat_xyzw
        sensor2ego_quat_wxyz = np.array([q_[3],q_[0],q_[1],q_[2]])
        # 四元数 wxyz
        calib_info["sensor2ego_rotation"] = sensor2ego_quat_wxyz
        calib_info["sensor2ego_translation"] = sensor2ego_translation
        
        
        # 相机到雷达平移套用相机到车的平移
        sensor2lidar_rotation =  sensor2ego_rotation
        sensor2lidar_translation = sensor2ego_translation
        calib_info["sensor2lidar_rotation"] = sensor2lidar_rotation
        calib_info["sensor2lidar_translation"] = sensor2lidar_translation
        
        
        # timestamp 在后面赋值同 lidar，也就是假设cam和lidar采样时间绝对同步。
        # ego2global_translation 在后面赋值同 lidar，也就是假设cam和lidar无运动差异。
        # ego2global_rotation 在后面赋值同 lidar，也就是假设cam和lidar无运动差异。

        # 相机内参
        tmp = calib["intrinsic"]
        camera_intrinsics = [tmp[:3], tmp[3:6], tmp[6:]]
        calib_info["camera_intrinsics"] = camera_intrinsics

        # 赋值
        cam_calib_info[channel] = calib_info
    return cam_calib_info


if __name__ == "__main__":
    
    # 数据集路径
    data_dir = './data/nuscenes_cidi_byd'
    pcd_dir_name = 'pcd.bin.car'
    label_dir_name = 'label.car'
    # 采集相机通道信息
    cams_channel_list = os.listdir(os.path.join(data_dir, 'image'))
    cams_channel_list.sort()
    print('{} cams channel'.format(len(cams_channel_list)), cams_channel_list)

    # pkl 生成 ==============================================================

    # 采集nuscenes信息
    nuscenes_infos = mmcv.load('./data/nuscenes/nuscenes_infos_train.pkl')
    # 采集相机通道标定信息
    cam_calib_info = get_cam_calib_data(data_dir, cams_channel_list)
    # 分配子集
    all_pcd_files = os.listdir(os.path.join(data_dir, pcd_dir_name))
    print('all_pcd_files ', len(all_pcd_files))
    # random.shuffle(all_pcd_files)  # 测试期保持不变
    train_files = all_pcd_files[:100]
    print('train_files ', len(train_files))
    valid_files = all_pcd_files[-10:]
    print('valid_files ', len(valid_files))
    # 生成pkl
    make_data(train_files,nuscenes_infos,cam_calib_info,cams_channel_list, 'nuscenes_infos_train.pkl')
    make_data(valid_files,nuscenes_infos,cam_calib_info,cams_channel_list, 'nuscenes_infos_val.pkl')
    exit()


    # 生成 gt ==================================================================
    dataset_name = 'NuScenesDataset'
    root_path = data_dir
    out_dir = data_dir
    info_prefix = 'nuscenes'
    load_augmented = False
    create_groundtruth_database(
            dataset_name,
            root_path,
            info_prefix,
            f"{out_dir}/{info_prefix}_infos_train.pkl",
            load_augmented=load_augmented,
        )   

    print('All is done!')


    # 验证写入信息
    # reload_infos = mmcv.load('./data/nuscenes_cidi/nuscenes_infos_train.pkl')
    # print(reload_infos)