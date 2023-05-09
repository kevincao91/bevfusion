import json
import shutil
import tempfile
import os
from os import path as osp
import time
from typing import Any, Dict

import mmcv
import numpy as np
import pyquaternion
import torch
from nuscenes.utils.data_classes import Box as NuScenesBox
from pyquaternion import Quaternion

from mmdet.datasets import DATASETS

from ..core.bbox import LiDARInstance3DBoxes
# kevin only 4 infer
from .custom_3d_kevin import Custom3DDataset
# ===
# from .custom_3d import Custom3DDataset


# kevin
from nuscenes.eval.detection.data_classes import DetectionMetrics, DetectionMetricDataList
from nuscenes.eval.detection.data_classes import DetectionMetricData

        
def center_distance(gt_box, pred_box):
    """
    L2 distance between the box centers (xy only).
    :param gt_box: GT annotation sample.
    :param pred_box: Predicted sample.
    :return: L2 distance.
    """
    return np.linalg.norm(
        np.array((pred_box["psr"]["position"]["x"], pred_box["psr"]["position"]["y"]))
         - np.array((gt_box["psr"]["position"]["x"], gt_box["psr"]["position"]["y"])))

def accumulate(gt_boxes, 
               pred_boxes,
               sample_token_list,
               class_name: str,
               dist_th: float):

    # ---------------------------------------------
    # Organize input and initialize accumulators.
    # ---------------------------------------------

    # Count the positives.
    all_gt_boxes = []
    for sample_token in sample_token_list:
        data = gt_boxes[sample_token]
        # copy sample_token
        for it in data:
            it['sample_token'] = sample_token
        # ===
        all_gt_boxes.extend(data)

    npos = len([1 for gt_box in all_gt_boxes if gt_box['obj_type'] == class_name])
    # print("Found {} GT bbox, \t from total {} GT bbox across {} samples data.".
    #         format(npos, len(all_gt_boxes), len(sample_token_list)))

    # For missing classes in the GT, return a data structure corresponding to no predictions.
    if npos == 0:
        # print("Skiped find PRD bbox.")
        return DetectionMetricData.no_predictions()

    # Organize the predictions in a single list.
    all_pred_boxes = []
    for sample_token in sample_token_list:
        data = pred_boxes[sample_token]
        # copy sample_token
        for it in data:
            it['sample_token'] = sample_token
        # ===
        all_pred_boxes.extend(data)
    pred_boxes_list = [pred_box for pred_box in all_pred_boxes if pred_box['obj_type'] == class_name]
    pred_confs = [pred_box['score'] for pred_box in pred_boxes_list]

    # print("Found {} PRD bbox, \t from total {} PRD bbox across {} samples data.".
    #         format(len(pred_confs), len(all_pred_boxes), len(sample_token_list)))

    # For missing classes in the PRD, return a data structure corresponding to no predictions.
    if len(pred_boxes_list) == 0:
        return DetectionMetricData.no_predictions()

    # Sort by confidence.
    sortind = [i for (v, i) in sorted((v, i) for (i, v) in enumerate(pred_confs))][::-1]

    # Do the actual matching.
    tp = []  # Accumulator of true positives.
    fp = []  # Accumulator of false positives.
    conf = []  # Accumulator of confidences.

    # match_data holds the extra metrics we calculate for each match.
    # match_data = {'trans_err': [],
    #               'vel_err': [],
    #               'scale_err': [],
    #               'orient_err': [],
    #               'attr_err': [],
    #               'conf': []}
    match_data = {'trans_err': np.ones(DetectionMetricData.nelem),
                  'vel_err': np.ones(DetectionMetricData.nelem),
                  'scale_err': np.ones(DetectionMetricData.nelem),
                  'orient_err': np.ones(DetectionMetricData.nelem),
                  'attr_err': np.ones(DetectionMetricData.nelem),
                  'conf': []}

    # ---------------------------------------------
    # Match and accumulate match data.
    # ---------------------------------------------

    taken = set()  # Initially no gt bounding box is matched.
    for ind in sortind:
        pred_box = pred_boxes_list[ind]
        min_dist = np.inf
        match_gt_idx = None

        for gt_idx, gt_box in enumerate(gt_boxes[pred_box['sample_token']]):

            # Find closest match among ground truth boxes
            if gt_box['obj_type'] == class_name and not (pred_box['sample_token'], gt_idx) in taken:
                this_distance = center_distance(gt_box, pred_box)
                if this_distance < min_dist:
                    min_dist = this_distance
                    match_gt_idx = gt_idx

        # If the closest match is close enough according to threshold we have a match!
        is_match = min_dist < dist_th

        if is_match:
            taken.add((pred_box['sample_token'], match_gt_idx))

            #  Update tp, fp and confs.
            tp.append(1)
            fp.append(0)
            conf.append(pred_box['score'])

            # Since it is a match, update match data also.
            '''
            gt_box_match = gt_boxes[pred_box.sample_token][match_gt_idx]

            match_data['trans_err'].append(center_distance(gt_box_match, pred_box))
            # match_data['vel_err'].append(velocity_l2(gt_box_match, pred_box))
            match_data['scale_err'].append(1 - scale_iou(gt_box_match, pred_box))

            # Barrier orientation is only determined up to 180 degree. (For cones orientation is discarded later)
            period = np.pi if class_name == 'barrier' else 2 * np.pi
            match_data['orient_err'].append(yaw_diff(gt_box_match, pred_box, period=period))

            # match_data['attr_err'].append(1 - attr_acc(gt_box_match, pred_box))
            '''
            match_data['conf'].append(pred_box['score'])

        else:
            # No match. Mark this as a false positive.
            tp.append(0)
            fp.append(1)
            conf.append(pred_box['score'])

    # Check if we have any matches. If not, just return a "no predictions" array.
    # if len(match_data['trans_err']) == 0:
    #     return DetectionMetricData.no_predictions()

    # ---------------------------------------------
    # Calculate and interpolate precision and recall
    # ---------------------------------------------

    # Accumulate.
    tp = np.cumsum(tp).astype(float)
    fp = np.cumsum(fp).astype(float)
    conf = np.array(conf)

    # Calculate precision and recall.
    prec = tp / (fp + tp)
    rec = tp / float(npos)

    rec_interp = np.linspace(0, 1, DetectionMetricData.nelem)  # 101 steps, from 0% to 100% recall.
    prec = np.interp(rec_interp, rec, prec, right=0)
    conf = np.interp(rec_interp, rec, conf, right=0)
    rec = rec_interp


    # ---------------------------------------------
    # Done. Instantiate MetricData and return
    # ---------------------------------------------
    return DetectionMetricData(recall=rec,
                               precision=prec,
                               confidence=conf,
                               trans_err=match_data['trans_err'],
                               vel_err=match_data['vel_err'],
                               scale_err=match_data['scale_err'],
                               orient_err=match_data['orient_err'],
                               attr_err=match_data['attr_err'])

def calc_ap(md, min_recall, min_precision):
    """ Calculated average precision. """

    assert 0 <= min_precision < 1
    assert 0 <= min_recall <= 1

    prec = np.copy(md.precision)
    prec = prec[round(100 * min_recall) + 1:]  # Clip low recalls. +1 to exclude the min recall bin.
    prec -= min_precision  # Clip low precision
    prec[prec < 0] = 0
    return float(np.mean(prec)) / (1.0 - min_precision)


# ========


@DATASETS.register_module()
class NuScenesDataset(Custom3DDataset):
    r"""NuScenes Dataset.

    This class serves as the API for experiments on the NuScenes Dataset.

    Please refer to `NuScenes Dataset <https://www.nuscenes.org/download>`_
    for data downloading.

    Args:
        ann_file (str): Path of annotation file.
        pipeline (list[dict], optional): Pipeline used for data processing.
            Defaults to None.
        dataset_root (str): Path of dataset root.
        classes (tuple[str], optional): Classes used in the dataset.
            Defaults to None.
        load_interval (int, optional): Interval of loading the dataset. It is
            used to uniformly sample the dataset. Defaults to 1.
        with_velocity (bool, optional): Whether include velocity prediction
            into the experiments. Defaults to True.
        modality (dict, optional): Modality to specify the sensor data used
            as input. Defaults to None.
        box_type_3d (str, optional): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'LiDAR' in this dataset. Available options includes.
            - 'LiDAR': Box in LiDAR coordinates.
            - 'Depth': Box in depth coordinates, usually for indoor dataset.
            - 'Camera': Box in camera coordinates.
        filter_empty_gt (bool, optional): Whether to filter empty GT.
            Defaults to True.
        test_mode (bool, optional): Whether the dataset is in test mode.
            Defaults to False.
        eval_version (bool, optional): Configuration version of evaluation.
            Defaults to  'detection_cvpr_2019'.
        use_valid_flag (bool): Whether to use `use_valid_flag` key in the info
            file as mask to filter gt_boxes and gt_names. Defaults to False.
    """
    NameMapping = {
        "movable_object.barrier": "barrier",
        "vehicle.bicycle": "bicycle",
        "vehicle.bus.bendy": "bus",
        "vehicle.bus.rigid": "bus",
        "vehicle.car": "car",
        "vehicle.construction": "construction_vehicle",
        "vehicle.motorcycle": "motorcycle",
        "human.pedestrian.adult": "pedestrian",
        "human.pedestrian.child": "pedestrian",
        "human.pedestrian.construction_worker": "pedestrian",
        "human.pedestrian.police_officer": "pedestrian",
        "movable_object.trafficcone": "traffic_cone",
        "vehicle.trailer": "trailer",
        "vehicle.truck": "truck",
    }
    DefaultAttribute = {
        "car": "vehicle.parked",
        "pedestrian": "pedestrian.moving",
        "trailer": "vehicle.parked",
        "truck": "vehicle.parked",
        "bus": "vehicle.moving",
        "motorcycle": "cycle.without_rider",
        "construction_vehicle": "vehicle.parked",
        "bicycle": "cycle.without_rider",
        "barrier": "",
        "traffic_cone": "",
    }
    AttrMapping = {
        "cycle.with_rider": 0,
        "cycle.without_rider": 1,
        "pedestrian.moving": 2,
        "pedestrian.standing": 3,
        "pedestrian.sitting_lying_down": 4,
        "vehicle.moving": 5,
        "vehicle.parked": 6,
        "vehicle.stopped": 7,
    }
    AttrMapping_rev = [
        "cycle.with_rider",
        "cycle.without_rider",
        "pedestrian.moving",
        "pedestrian.standing",
        "pedestrian.sitting_lying_down",
        "vehicle.moving",
        "vehicle.parked",
        "vehicle.stopped",
    ]
    # https://github.com/nutonomy/nuscenes-devkit/blob/57889ff20678577025326cfc24e57424a829be0a/python-sdk/nuscenes/eval/detection/evaluate.py#L222 # noqa
    ErrNameMapping = {
        "trans_err": "mATE",
        "scale_err": "mASE",
        "orient_err": "mAOE",
        "vel_err": "mAVE",
        "attr_err": "mAAE",
    }
    # CLASSES = (
    #     "car",
    #     "truck",
    #     "trailer",
    #     "bus",
    #     "construction_vehicle",
    #     "bicycle",
    #     "motorcycle",
    #     "pedestrian",
    #     "traffic_cone",
    #     "barrier",
    # )

    # kevin
    # 对应7类
    CLASSES = (
        "car",
        "truck",
        "pedestrian",
        "rider",
        "bus",
        "bicycle",
        "traffic_cone",
    )
    # 对应4类
    # CLASSES = (
    #     "car",
    #     "truck",
    #     "pedestrian",
    #     "rider",
    # )

    def __init__(
        self,
        ann_file,
        pipeline=None,
        dataset_root=None,
        object_classes=None,
        map_classes=None,
        load_interval=1,
        with_velocity=True,
        modality=None,
        box_type_3d="LiDAR",
        filter_empty_gt=True,
        test_mode=False,
        # kevin
        # eval_version="detection_cvpr_2019",
        eval_version="detection_cidi_2023",
        use_valid_flag=False,
    ) -> None:
        self.load_interval = load_interval
        self.use_valid_flag = use_valid_flag
        super().__init__(
            dataset_root=dataset_root,
            ann_file=ann_file,
            pipeline=pipeline,
            classes=object_classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
        )
        self.map_classes = map_classes

        self.with_velocity = with_velocity
        self.eval_version = eval_version
        from nuscenes.eval.detection.config import config_factory

        self.eval_detection_configs = config_factory(self.eval_version)
        if self.modality is None:
            self.modality = dict(
                use_camera=False,
                use_lidar=True,
                use_radar=False,
                use_map=False,
                use_external=False,
            )

    def get_cat_ids(self, idx):
        """Get category distribution of single scene.

        Args:
            idx (int): Index of the data_info.

        Returns:
            dict[list]: for each category, if the current scene
                contains such boxes, store a list containing idx,
                otherwise, store empty list.
        """
        info = self.data_infos[idx]
        if self.use_valid_flag:
            mask = info["valid_flag"]
            gt_names = set(info["gt_names"][mask])
        else:
            gt_names = set(info["gt_names"])

        cat_ids = []
        for name in gt_names:
            if name in self.CLASSES:
                cat_ids.append(self.cat2id[name])
        return cat_ids

    def load_annotations(self, ann_file):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations sorted by timestamps.
        """
        data = mmcv.load(ann_file)
        data_infos = list(sorted(data["infos"], key=lambda e: e["timestamp"]))
        data_infos = data_infos[:: self.load_interval]
        self.metadata = data["metadata"]
        self.version = self.metadata["version"]
        return data_infos

    def get_data_info(self, index: int) -> Dict[str, Any]:
        info = self.data_infos[index]

        data = dict(
            token=info["token"],
            sample_idx=info['token'],
            lidar_path=info["lidar_path"],
            sweeps=info["sweeps"],
            timestamp=info["timestamp"],
            location=info["location"],
        )

        # ego to global transform
        ego2global = np.eye(4).astype(np.float32)
        ego2global[:3, :3] = Quaternion(info["ego2global_rotation"]).rotation_matrix
        ego2global[:3, 3] = info["ego2global_translation"]
        data["ego2global"] = ego2global

        # lidar to ego transform
        lidar2ego = np.eye(4).astype(np.float32)
        lidar2ego[:3, :3] = Quaternion(info["lidar2ego_rotation"]).rotation_matrix
        lidar2ego[:3, 3] = info["lidar2ego_translation"]
        data["lidar2ego"] = lidar2ego

        if self.modality["use_camera"]:
            data["image_paths"] = []
            data["lidar2camera"] = []
            data["lidar2image"] = []
            data["camera2ego"] = []
            data["camera_intrinsics"] = []
            data["camera2lidar"] = []

            for _, camera_info in info["cams"].items():
                data["image_paths"].append(camera_info["data_path"])

                # lidar to camera transform
                lidar2camera_r = np.linalg.inv(camera_info["sensor2lidar_rotation"])
                lidar2camera_t = (
                    camera_info["sensor2lidar_translation"] @ lidar2camera_r.T
                )
                lidar2camera_rt = np.eye(4).astype(np.float32)
                lidar2camera_rt[:3, :3] = lidar2camera_r.T
                lidar2camera_rt[3, :3] = -lidar2camera_t
                data["lidar2camera"].append(lidar2camera_rt.T)

                # camera intrinsics
                camera_intrinsics = np.eye(4).astype(np.float32)
                camera_intrinsics[:3, :3] = camera_info["camera_intrinsics"]
                data["camera_intrinsics"].append(camera_intrinsics)

                # lidar to image transform
                lidar2image = camera_intrinsics @ lidar2camera_rt.T
                data["lidar2image"].append(lidar2image)

                # camera to ego transform
                camera2ego = np.eye(4).astype(np.float32)
                camera2ego[:3, :3] = Quaternion(
                    camera_info["sensor2ego_rotation"]
                ).rotation_matrix
                camera2ego[:3, 3] = camera_info["sensor2ego_translation"]
                data["camera2ego"].append(camera2ego)

                # camera to lidar transform
                camera2lidar = np.eye(4).astype(np.float32)
                camera2lidar[:3, :3] = camera_info["sensor2lidar_rotation"]
                camera2lidar[:3, 3] = camera_info["sensor2lidar_translation"]
                data["camera2lidar"].append(camera2lidar)

        annos = self.get_ann_info(index)
        data["ann_info"] = annos
        return data

    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: Annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): \
                    3D ground truth bboxes
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - gt_names (list[str]): Class names of ground truths.
        """
        info = self.data_infos[index]
        # filter out bbox containing no points
        if self.use_valid_flag:
            mask = info["valid_flag"]
        else:
            mask = info["num_lidar_pts"] > 0
        gt_bboxes_3d = info["gt_boxes"][mask]
        gt_names_3d = info["gt_names"][mask]
        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d)

        if self.with_velocity:
            gt_velocity = info["gt_velocity"][mask]
            nan_mask = np.isnan(gt_velocity[:, 0])
            gt_velocity[nan_mask] = [0.0, 0.0]
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)

        # the nuscenes box center is [0.5, 0.5, 0.5], we change it to be
        # the same as KITTI (0.5, 0.5, 0)
        # haotian: this is an important change: from 0.5, 0.5, 0.5 -> 0.5, 0.5, 0
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d, box_dim=gt_bboxes_3d.shape[-1], origin=(0.5, 0.5, 0)
        ).convert_to(self.box_mode_3d)
        # kevin
        # gt_bboxes_3d = LiDARInstance3DBoxes(
        #     gt_bboxes_3d, box_dim=gt_bboxes_3d.shape[-1], origin=(0.5, 0.5, 0.5)
        # )

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d,
        )
        return anns_results

    def _format_bbox(self, results, jsonfile_prefix=None):
        """Convert the results to the standard format.

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str): The prefix of the output jsonfile.
                You can specify the output directory/filename by
                modifying the jsonfile_prefix. Default: None.

        Returns:
            str: Path of the output json file.
        """
        # 标注中类别首字母都大写
        idx2name = self.gt_CLASSES
        # 采集subset信息
        sample_token_list = []
        sample_scene_token_list = []
        subset_infos = mmcv.load(self.ann_file)
        for it in subset_infos['infos']:
            lidar_path = it['lidar_path']
            sample_scene_token_str = '/'.join(lidar_path.split('/')[-3:])
            sample_scene_token_str = sample_scene_token_str.replace('.pcd.bin','')
            sample_scene_token_list.append(sample_scene_token_str)                                                                                                                                                    
            sample_token_str = lidar_path.split('/')[-1].replace('.pcd.bin','')
            sample_token_list.append(sample_token_str)

        print("Start to copy groundtrue file...")
        gt_cp_dir = osp.join(jsonfile_prefix, 'gt_data')
        mmcv.mkdir_or_exist(gt_cp_dir)
        for sample_token, sample_scene_token in zip(sample_token_list, sample_scene_token_list):
            src_ = osp.join(self.dataset_root ,sample_scene_token.replace('pcd.car.bin','label')+'.json')
            det_ = osp.join(gt_cp_dir ,sample_token+'.json')
            if not os.path.exists(det_):
                shutil.copy(src_, det_)
                # print('copy file to {}'.format(det_))

        print("Start to convert detection format...")
        mmcv.mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, "model_pred_data")
        mmcv.mkdir_or_exist(res_path)
        for sample_id, det in enumerate(mmcv.track_iter_progress(results)):
            
            # kevin
            bboxes = det["boxes_3d"].tensor.numpy()
            scores = det["scores_3d"].numpy()
            labels = det["labels_3d"].numpy()
            print('bboxes: ', bboxes.shape)
            
            # kevin =======
            indices = np.argsort(-scores)
            bboxes = bboxes[indices]
            scores = scores[indices]
            labels = labels[indices]


            # 结果持久化
            len_bboxes = bboxes.shape[0]
            res_list = []
            for ii in range(len_bboxes):
                box_data = list(map(float, bboxes[ii, :]))
                position = {
                    'x': box_data[0],
                    'y': box_data[1],
                    'z': box_data[2]
                    }
                scale = {
                    # 对于导出lable在car坐标系下的数据，需要xy互换
                    'x': box_data[4],
                    'y': box_data[3],
                    'z': box_data[5]
                    }
                rotation = {
                    'x': 0,
                    'y': 0,
                    # 数据生成时的转换  -rots - np.pi / 2  = yaw
                    # 需要回到原来的角度  rots = - yaw - np.pi / 2
                    'z': -1 * (box_data[6] + np.pi /2)
                    }
                psr = {
                    'position': position,
                    'scale': scale,
                    'rotation': rotation
                }
                res_data = {
                    'psr': psr,
                    "obj_type": idx2name[labels[ii]],
                    "obj_id":"",
                    "score": float(scores[ii])
                    }
                # print(res_data)
                res_list.append(res_data)

            if len_bboxes > 0:
                lidar_name = sample_token_list[sample_id]
                path_out = osp.join(jsonfile_prefix, 'model_pred_data', '{}.json'.format(lidar_name))
                mmcv.mkdir_or_exist(osp.dirname(path_out))
                with open(path_out,"w") as f:
                    json.dump(res_list,f,indent=4)
                    print("{} 写入完成...".format(path_out))
            # ===

        return res_path

    def format_results(self, results, jsonfile_prefix=None):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: Returns (result_files, tmp_dir), where `result_files` is a \
                dict containing the json filepaths, `tmp_dir` is the temporal \
                directory created for saving json files when \
                `jsonfile_prefix` is not specified.
        """
        assert isinstance(results, list), "results must be a list"
        assert len(results) == len(
            self
        ), "The length of results is not equal to the dataset len: {} != {}".format(
            len(results), len(self)
        )

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, "results")
        else:
            tmp_dir = None

        result_files = self._format_bbox(results, jsonfile_prefix)
        return result_files, tmp_dir

    def evaluate_map(self, results):
        thresholds = torch.tensor([0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65])

        num_classes = len(self.map_classes)
        num_thresholds = len(thresholds)

        tp = torch.zeros(num_classes, num_thresholds)
        fp = torch.zeros(num_classes, num_thresholds)
        fn = torch.zeros(num_classes, num_thresholds)

        for result in results:
            pred = result["masks_bev"]
            label = result["gt_masks_bev"]

            pred = pred.detach().reshape(num_classes, -1)
            label = label.detach().bool().reshape(num_classes, -1)

            pred = pred[:, :, None] >= thresholds
            label = label[:, :, None]

            tp += (pred & label).sum(dim=1)
            fp += (pred & ~label).sum(dim=1)
            fn += (~pred & label).sum(dim=1)

        ious = tp / (tp + fp + fn + 1e-7)

        metrics = {}
        for index, name in enumerate(self.map_classes):
            metrics[f"map/{name}/iou@max"] = ious[index].max().item()
            for threshold, iou in zip(thresholds, ious[index]):
                metrics[f"map/{name}/iou@{threshold.item():.2f}"] = iou.item()
        metrics["map/mean/iou@max"] = ious.max(dim=1).values.mean().item()
        return metrics

    def evaluate(
        self,
        results,
        metric="bbox",
        jsonfile_prefix=None,
        result_names=["pts_bbox"],
        **kwargs,
    ):
        """Evaluation in nuScenes protocol.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            dict[str, float]: Results of each evaluation metric.
        """
        # kevin
        jsonfile_prefix = './evaluate_tmp_data'
        # 转换内部class标签到标注的class标签，单词大写首字母
        gt_CLASSES = []
        for cls in self.CLASSES:
            if cls.find('_')>0:
                words = cls.split('_')
                words = [word.capitalize() for word in words]
                new_cls = '_'.join(words)
                gt_CLASSES.append(new_cls)
            else:
                gt_CLASSES.append(cls.capitalize())
        self.gt_CLASSES = gt_CLASSES
        # ===

        metrics = {}

        if "masks_bev" in results[0]:
            metrics.update(self.evaluate_map(results))

        if "boxes_3d" in results[0]:
            result_out_dir, tmp_dir = self.format_results(results, jsonfile_prefix)
            # kevin
            print('evaluate:  ', result_out_dir, tmp_dir)
            # ===

            self.eval_cfg = self.eval_detection_configs
            DIST_THS = self.eval_cfg.dist_ths
            print('DIST_THS : ', DIST_THS)

            subset_file = self.ann_file
            pred_dir = result_out_dir
            gt_dir = osp.join(jsonfile_prefix, 'gt_data')
            # 采集subset信息
            sample_token_list = []
            subset_infos = mmcv.load(subset_file)
            for it in subset_infos['infos']:
                lidar_path = it['lidar_path']                                                                                                                                                 
                sample_token_str = lidar_path.split('/')[-1].replace('.pcd.bin','')
                sample_token_list.append(sample_token_str)
            sample_token_list.sort()
            print('sample_token_list:',len(sample_token_list))

            # ===========================================
            start_time = time.time()

            # -----------------------------------
            # Step 1: Accumulate metric data for all classes and distance thresholds.
            # -----------------------------------
            print('load data...')

            pred_data = {}
            for sample_token in sample_token_list:
                pred_path = osp.join(pred_dir, '{}.json'.format(sample_token))
                with open(pred_path) as f:
                    frame_data = json.load(f)
                pred_data[sample_token] = frame_data

            gt_data = {}
            for sample_token in sample_token_list:
                gt_path = osp.join(gt_dir, '{}.json'.format(sample_token))
                with open(gt_path) as f:
                    frame_data = json.load(f)
                gt_data[sample_token] = frame_data

            print('Accumulating metric data...')
            metric_data_list = DetectionMetricDataList()
            for class_name in self.gt_CLASSES:
                for dist_th in DIST_THS:
                    # print('accumulate AP @ {} @ {}'.format(class_name, dist_th))
                    md = accumulate(gt_data, pred_data, sample_token_list, class_name, dist_th)
                    metric_data_list.set(class_name, dist_th, md)
            # print(metric_data_list.serialize())

            # -----------------------------------
            # Step 2: Calculate metrics from the data.
            # -----------------------------------
            print('Calculating metrics...')
            min_recall = self.eval_cfg.min_recall
            min_precision = self.eval_cfg.min_precision
            print('min_recall, min_precision: ', min_recall, min_precision)

            metrics = DetectionMetrics(self.eval_cfg)
            for class_name in self.gt_CLASSES:
                # Compute APs.
                for dist_th in DIST_THS:
                    metric_data = metric_data_list[(class_name, dist_th)]
                    ap = calc_ap(metric_data, min_recall, min_precision)
                    print('accumulate AP @ {} @ {} == {}'.format(class_name, dist_th, ap))
                    metrics.add_label_ap(class_name, dist_th, ap)
            
            # Compute evaluation time.
            metrics.add_runtime(time.time() - start_time)
            # =======================================================================

            # Render PR and TP curves.
            # self.render(metrics, metric_data_list)

            # Dump the metric data, meta and metrics to disk.
            output_dir = osp.join(jsonfile_prefix, 'evaluation')
            if not osp.exists(output_dir):
                os.mkdir(output_dir)
            print('Saving metrics to: %s' % output_dir)
            metrics_summary = metrics.serialize()
            with open(osp.join(output_dir, 'metrics_summary.json'), 'w') as f:
                json.dump(metrics_summary, f, indent=2)
            with open(osp.join(output_dir, 'metrics_details.json'), 'w') as f:
                json.dump(metric_data_list.serialize(), f, indent=2)

            # Print high-level metrics.
            print('mAP: %.4f' % (metrics_summary['mean_ap']))
            err_name_mapping = {
                'trans_err': 'mATE',
                'scale_err': 'mASE',
                'orient_err': 'mAOE',
                'vel_err': 'mAVE',
                'attr_err': 'mAAE'
            }
            for tp_name, tp_val in metrics_summary['tp_errors'].items():
                print('%s: %.4f' % (err_name_mapping[tp_name], tp_val))
            print('NDS: %.4f' % (metrics_summary['nd_score']))
            print('Eval time: %.1fs' % metrics_summary['eval_time'])

            # Print per-class metrics.
            '''
            print()
            print('Per-class results:')
            print('Object Class\tAP')
            class_aps = metrics_summary['mean_dist_aps']
            for class_name in class_aps.keys():
                print('%s\t%.3f'
                        % (class_name, class_aps[class_name]))
            # Print per-class metrics.
            print()
            print('Per-class results:')
            print('Object Class\tAP\tATE\tASE\tAOE\tAVE\tAAE')
            class_aps = metrics_summary['mean_dist_aps']
            class_tps = metrics_summary['label_tp_errors']
            for class_name in class_aps.keys():
                print('%s\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
                    % (class_name, class_aps[class_name],
                        class_tps[class_name]['trans_err'],
                        class_tps[class_name]['scale_err'],
                        class_tps[class_name]['orient_err'],
                        class_tps[class_name]['vel_err'],
                        class_tps[class_name]['attr_err']))
            '''
            # ===============

            # record metrics
            metrics = mmcv.load(osp.join(output_dir, "metrics_summary.json"))
            detail = dict()
            for name in self.gt_CLASSES:
                for k, v in metrics["label_aps"][name].items():
                    val = float("{:.4f}".format(v))
                    detail["object/{}_ap_dist_{}".format(name, k)] = val
                # for k, v in metrics["label_tp_errors"][name].items():
                #     val = float("{:.4f}".format(v))
                #     detail["object/{}_{}".format(name, k)] = val
                # for k, v in metrics["tp_errors"].items():
                #     val = float("{:.4f}".format(v))
                #     detail["object/{}".format(self.ErrNameMapping[k])] = val

            # detail["object/nds"] = metrics["nd_score"]
            detail["object/map"] = metrics["mean_ap"]

            # kevin
            # if tmp_dir is not None:
            #     tmp_dir.cleanup()
            source_path = '/home/cao.ke/Workspace/bevfusion/evaluate_tmp_data'
            target_path = '/home/cao.ke/Workspace/bevfusion/evaluate_tmp_data_save'
            subdir = os.listdir(target_path)
            cnt = len(subdir)+1
            target_dir = '/home/cao.ke/Workspace/bevfusion/evaluate_tmp_data_save/'+str(cnt)
            # os.mkdir(target_dir)
            shutil.copytree(source_path, target_dir)
            # ===

        return detail


def output_to_nusc_box(detection):
    """Convert the output to the box class in the nuScenes.

    Args:
        detection (dict): Detection results.

            - boxes_3d (:obj:`BaseInstance3DBoxes`): Detection bbox.
            - scores_3d (torch.Tensor): Detection scores.
            - labels_3d (torch.Tensor): Predicted box labels.

    Returns:
        list[:obj:`NuScenesBox`]: List of standard NuScenesBoxes.
    """
    box3d = detection["boxes_3d"]
    scores = detection["scores_3d"].numpy()
    labels = detection["labels_3d"].numpy()

    box_gravity_center = box3d.gravity_center.numpy()
    box_dims = box3d.dims.numpy()
    box_yaw = box3d.yaw.numpy()
    # TODO: check whether this is necessary
    # with dir_offset & dir_limit in the head
    box_yaw = -box_yaw - np.pi / 2

    box_list = []
    for i in range(len(box3d)):
        quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box_yaw[i])
        velocity = (*box3d.tensor[i, 7:9], 0.0)
        # velo_val = np.linalg.norm(box3d[i, 7:9])
        # velo_ori = box3d[i, 6]
        # velocity = (
        # velo_val * np.cos(velo_ori), velo_val * np.sin(velo_ori), 0.0)
        box = NuScenesBox(
            box_gravity_center[i],
            box_dims[i],
            quat,
            label=labels[i],
            score=scores[i],
            velocity=velocity,
        )
        box_list.append(box)
    return box_list


def lidar_nusc_box_to_global(
    info, boxes, classes, eval_configs, eval_version="detection_cvpr_2019"
):
    """Convert the box from ego to global coordinate.

    Args:
        info (dict): Info for a specific sample data, including the
            calibration information.
        boxes (list[:obj:`NuScenesBox`]): List of predicted NuScenesBoxes.
        classes (list[str]): Mapped classes in the evaluation.
        eval_configs : Evaluation configuration object.
        eval_version (str): Evaluation version.
            Default: 'detection_cvpr_2019'

    Returns:
        list: List of standard NuScenesBoxes in the global
            coordinate.
    """
    box_list = []
    for box in boxes:
        # Move box to ego vehicle coord system
        box.rotate(pyquaternion.Quaternion(info["lidar2ego_rotation"]))
        box.translate(np.array(info["lidar2ego_translation"]))

        # filter det in ego.
        cls_range_map = eval_configs.class_range
        radius = np.linalg.norm(box.center[:2], 2)
        det_range = cls_range_map[classes[box.label]]
        if radius > det_range:
            continue
        # Move box to global coord system
        box.rotate(pyquaternion.Quaternion(info["ego2global_rotation"]))
        box.translate(np.array(info["ego2global_translation"]))
        box_list.append(box)
    return box_list
