import argparse
import copy
import math
import os
import os.path as osp
import shutil
import time
import warnings
import numpy as np

import mmcv
import json

from nuscenes.eval.detection.data_classes import DetectionMetrics, DetectionMetricDataList

from mmdet3d.datasets.nuscenes_dataset_kevin import calc_ap, accumulate

def load_json_data(sample_token_list, data_dir):
    data = {}
    for sample_token in sample_token_list:
        pred_path = os.path.join(data_dir, '{}.json'.format(sample_token))
        with open(pred_path) as f:
            frame_data = json.load(f)
        data[sample_token] = frame_data
    return data


def filter_eval_boxes(data, max_dist):

    # Accumulators for number of filtered boxes.
    new_data = {}
    total, dist_filter = 0, 0
    for sample_token in data:
        new_data[sample_token] = []
        # Filter on distance first.
        total += len(data[sample_token])
        for box in data[sample_token]:
            obj_type = box['obj_type']
            x = float(box['psr']['position']['x'])
            y = float(box['psr']['position']['y'])
            center_dist = math.sqrt(x*x+y*y)
            if center_dist <= max_dist[obj_type]:
                new_data[sample_token].append(box)
        dist_filter += len(new_data[sample_token])
       
    print("=> Original number of boxes: %d" % total)
    print("=> After distance based filtering: %d" % dist_filter)

    return new_data


def parse_args():
    parser = argparse.ArgumentParser(description="CIDI eval a model")
    parser.add_argument("--data-dir", type=str, default='./data/nuscenes_cidi_h5_202211_split2scene')
    parser.add_argument("--split", type=str, default='val')
    parser.add_argument("--out-dir", type=str, default="viz4exp116")
    args = parser.parse_args()
    return args


def main():

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
    # 转换内部class标签到标注的class标签，单词大写首字母
    gt_CLASSES = []
    for cls in CLASSES:
        if cls.find('_')>0:
            words = cls.split('_')
            words = [word.capitalize() for word in words]
            new_cls = '_'.join(words)
            gt_CLASSES.append(new_cls)
        else:
            gt_CLASSES.append(cls.capitalize())
    # ===

    from nuscenes.eval.detection.config import config_factory
    eval_version = "detection_cidi_2023"
    eval_detection_configs = config_factory(eval_version)
    eval_cfg = eval_detection_configs
    DIST_THS = eval_cfg.dist_ths
    print('DIST_THS : ', DIST_THS)

    args = parse_args()
    print(args.data_dir)
    subset_file = osp.join(args.data_dir, 'nuscenes_infos_{}.pkl'.format(args.split))
    pred_dir = osp.join(args.out_dir, 'model_pred_data')
    gt_dst_dir = osp.join(args.out_dir, 'gt_data')
    
    if args.data_dir.find('split2scene') > 0:
        # 采集subset信息
        sample_token_list = []
        sample_scene_token_list = []
        subset_infos = mmcv.load(subset_file)
        for it in subset_infos['infos']:
            lidar_path = it['lidar_path']
            sample_scene_token_str = '/'.join(lidar_path.split('/')[-3:])
            sample_scene_token_str = sample_scene_token_str.replace('.pcd.bin','')
            sample_scene_token_list.append(sample_scene_token_str)                                                                                                                                                    
            sample_token_str = lidar_path.split('/')[-1].replace('.pcd.bin','')
            sample_token_list.append(sample_token_str)

        print("Start to copy groundtrue file...")
        mmcv.mkdir_or_exist(gt_dst_dir)
        for sample_token, sample_scene_token in zip(sample_token_list, sample_scene_token_list):
            src_ = osp.join(args.data_dir ,sample_scene_token.replace('pcd.car.bin','label')+'.json')
            det_ = osp.join(gt_dst_dir ,sample_token+'.json')
            if not os.path.exists(det_):
                shutil.copy(src_, det_)
                # print('copy file to {}'.format(det_))

    else:
        # 采集subset信息
        sample_token_list = []
        subset_infos = mmcv.load(subset_file)
        for it in subset_infos['infos']:
            lidar_path = it['lidar_path']
            sample_token_str = lidar_path.split('/')[-1].replace('.pcd.bin','')
            sample_token_list.append(sample_token_str)

        print("Start to copy groundtrue file...")
        gt_src_dir = osp.join(args.data_dir, 'label')
        mmcv.mkdir_or_exist(gt_dst_dir)
        for sample_token in sample_token_list:
            src_ = osp.join(gt_src_dir, sample_token+'.json')
            det_ = osp.join(gt_dst_dir, sample_token+'.json')
            if not os.path.exists(det_):
                shutil.copy(src_, det_)
                # print('copy file to {}'.format(det_))

    start_time = time.time()
    # -----------------------------------
    # Step 1: Accumulate metric data for all classes and distance thresholds.
    # -----------------------------------
    print('load data...')
    pred_data = load_json_data(sample_token_list, pred_dir)
    gt_data = load_json_data(sample_token_list, gt_dst_dir)

    # Filter boxes (distance, points per box, etc.).
    CLASS_RANGE = eval_cfg.class_range
    print('CLASS_RANGE : ', CLASS_RANGE)
    print('Filtering predictions')
    pred_data = filter_eval_boxes(pred_data, CLASS_RANGE)
    print('Filtering ground truth annotations')
    gt_data = filter_eval_boxes(gt_data, CLASS_RANGE)


    print('Accumulating metric data...')
    metric_data_list = DetectionMetricDataList()
    for class_name in gt_CLASSES:
        for dist_th in DIST_THS:
            print('accumulate AP @ {} @ {}'.format(class_name, dist_th))
            md = accumulate(gt_data, pred_data, sample_token_list, class_name, dist_th)
            # print('accumulate AP @ {} @ {} => {}'.format(class_name, dist_th, md))
            metric_data_list.set(class_name, dist_th, md)
    # print(metric_data_list)

    # -----------------------------------
    # Step 2: Calculate metrics from the data.
    # -----------------------------------
    print('Calculating metrics...')
    min_recall = eval_cfg.min_recall
    min_precision = eval_cfg.min_precision
    print('min_recall, min_precision: ', min_recall, min_precision)

    metrics = DetectionMetrics(eval_cfg)
    for class_name in gt_CLASSES:
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
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    output_dir = os.path.join(args.out_dir, 'evaluation')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    print('Saving metrics to: %s' % output_dir)
    metrics_summary = metrics.serialize()
    with open(os.path.join(output_dir, 'metrics_summary.json'), 'w') as f:
        json.dump(metrics_summary, f, indent=2)
    with open(os.path.join(output_dir, 'metrics_details.json'), 'w') as f:
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
    print()
    print('Per-class results:')
    print('Object Class\tAP')
    class_aps = metrics_summary['mean_dist_aps']
    for class_name in class_aps.keys():
        print('%s\t%.3f'
                % (class_name, class_aps[class_name]))

    # ===============

    # record metrics
    metrics = mmcv.load(os.path.join(output_dir, "metrics_summary.json"))
    detail = dict()
    for name in gt_CLASSES:
        for k, v in metrics["label_aps"][name].items():
            val = float("{:.4f}".format(v))
            detail["object/{}_ap_dist_{}".format(name, k)] = val

    detail["object/map"] = metrics["mean_ap"]


if __name__ == "__main__":
    # 预测结果阈值要设置为None
    main()
