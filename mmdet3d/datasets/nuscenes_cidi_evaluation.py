import argparse
import copy
import os
import warnings
import numpy as np

import mmcv
import json


class DetectionMetrics:
    """ Stores average precision and true positive metric results. Provides properties to summarize. """

    def __init__(self):
        self._label_aps = {}

    def add_label_ap(self, detection_name, dist_th, ap):
        if detection_name not in self._label_aps:
            # self._label_aps[detection_name] = {0.5: None, 1.0: None, 2.0: None, 4.0: None}
            self._label_aps[detection_name] = {}
        self._label_aps[detection_name][dist_th] = ap

    def get_label_ap(self, detection_name: str, dist_th: float) -> float:
        return self._label_aps[detection_name][dist_th]

    @property
    def mean_dist_aps(self):
        """ Calculates the mean over distance thresholds for each label. """
        return {class_name: np.mean(list(d.values())) for class_name, d in self._label_aps.items()}

    @property
    def mean_ap(self) -> float:
        """ Calculates the mean AP by averaging over distance thresholds and classes. """
        return float(np.mean(list(self.mean_dist_aps.values())))

    def serialize(self):
        return {
            'label_aps': self._label_aps,
            'mean_dist_aps': self.mean_dist_aps,
            'mean_ap': self.mean_ap,
        }


class DetectionMetricDataList:
    """ This stores a set of MetricData in a dict indexed by (name, match-distance). """

    def __init__(self):
        self.md = {}

    def __getitem__(self, key):
        return self.md[key]

    def get_class_data(self, detection_name: str):
        """ Get all the MetricData entries for a certain detection_name. """
        return [(md, dist_th) for (name, dist_th), md in self.md.items() if name == detection_name]

    def get_dist_data(self, dist_th: float):
        """ Get all the MetricData entries for a certain match_distance. """
        return [(md, detection_name) for (detection_name, dist), md in self.md.items() if dist == dist_th]

    def set(self, detection_name, match_distance, data):
        """ Sets the MetricData entry for a certain detection_name and match_distance. """
        self.md[(detection_name, match_distance)] = data

    def serialize(self):
        res = {}
        for key, value in self.md.items():
            if value:
                res[key[0] + ':' + str(key[1]) + ':recall'] = list(map(float, value['recall']))
                res[key[0] + ':' + str(key[1]) + ':precision'] = list(map(float, value['precision']))
                res[key[0] + ':' + str(key[1]) + ':confidence'] = list(map(float, value['confidence']))
            else:
                res[key[0] + ':' + str(key[1]) + ':recall'] = []
                res[key[0] + ':' + str(key[1]) + ':precision'] = []
                res[key[0] + ':' + str(key[1]) + ':confidence'] = []
        return res


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
    print("Found {} bbox of class {} out of {} total GT bbox across {} samples data.".
            format(npos, class_name, len(all_gt_boxes), len(sample_token_list)))

    # For missing classes in the GT, return a data structure corresponding to no predictions.
    if npos == 0:
        return None

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

    print("Found {} PRED of class {} out of {} total across {} samples.".
            format(len(pred_confs), class_name, len(all_pred_boxes), len(sample_token_list)))

    # Sort by confidence.
    sortind = [i for (v, i) in sorted((v, i) for (i, v) in enumerate(pred_confs))][::-1]

    # Do the actual matching.
    tp = []  # Accumulator of true positives.
    fp = []  # Accumulator of false positives.
    conf = []  # Accumulator of confidences.

    # match_data holds the extra metrics we calculate for each match.
    match_data = {'trans_err': [],
                  'vel_err': [],
                  'scale_err': [],
                  'orient_err': [],
                  'attr_err': [],
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
            match_data['conf'].append(pred_box['score'])

        else:
            # No match. Mark this as a false positive.
            tp.append(0)
            fp.append(1)
            conf.append(pred_box['score'])

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

    rec_interp = np.linspace(0, 1, 101)  # 101 steps, from 0% to 100% recall.
    prec = np.interp(rec_interp, rec, prec, right=0)
    conf = np.interp(rec_interp, rec, conf, right=0)
    rec = rec_interp


    # ---------------------------------------------
    # Done. Instantiate MetricData and return
    # ---------------------------------------------
    return {'recall': rec,
            'precision': prec,
            'confidence': conf}

def calc_ap(md, min_recall, min_precision):
    """ Calculated average precision. """

    assert 0 <= min_precision < 1
    assert 0 <= min_recall <= 1

    if not md:
        return 0.0

    prec = np.copy(md['precision'])
    prec = prec[round(100 * min_recall) + 1:]  # Clip low recalls. +1 to exclude the min recall bin.
    prec -= min_precision  # Clip low precision
    prec[prec < 0] = 0
    return float(np.mean(prec)) / (1.0 - min_precision)



def parse_args():
    parser = argparse.ArgumentParser(description="CIDI eval a model")
    parser.add_argument("--data-dir", type=str, default='/home/cao.ke/Workspace/bevfusion/data/nuscenes_cidi_bkjw_1')
    parser.add_argument("--split", type=str, default='val')
    parser.add_argument("--out-dir", type=str, default="viz4exp100")
    args = parser.parse_args()
    return args


def main():

    CLASSES = [
        # 对应10类
        # - car
        # - truck
        # - construction_vehicle
        # - bus
        # - trailer
        # - barrier
        # - motorcycle
        # - bicycle
        # - pedestrian
        # - traffic_cone
        # 对应3类
        'Car',
        'Truck',
        'Pedestrian'
        # ==
    ]
    DIST_THS = [
        0.5,
        1.0,
        2.0,
        4.0
    ]

    args = parse_args()
    print(args.data_dir)
    label_dir = os.path.join(args.data_dir, 'label')
    subset_file = os.path.join(args.data_dir, 'nuscenes_infos_{}.pkl'.format(args.split))
    pred_dir = os.path.join(args.out_dir, 'model_pred_data')
    # 采集subset信息
    sample_token_list = []
    subset_infos = mmcv.load(subset_file)
    for it in subset_infos['infos']:
        lidar_path = it['lidar_path']
        sample_token_str = lidar_path.split('/')[-1].replace('.pcd.bin','')
        sample_token_list.append(sample_token_str)
    sample_token_list.sort()

    # -----------------------------------
    # Step 1: Accumulate metric data for all classes and distance thresholds.
    # -----------------------------------
    print('load data...')

    pred_data = {}
    for sample_token in sample_token_list:
        pred_path = os.path.join(pred_dir, '{}.json'.format(sample_token))
        with open(pred_path) as f:
            frame_data = json.load(f)
        pred_data[sample_token] = frame_data

    gt_data = {}
    for sample_token in sample_token_list:
        gt_path = os.path.join(label_dir, '{}.json'.format(sample_token))
        with open(gt_path) as f:
            frame_data = json.load(f)
        gt_data[sample_token] = frame_data

    print('Accumulating metric data...')
    metric_data_list = DetectionMetricDataList()
    for class_name in CLASSES:
        for dist_th in DIST_THS:
            md = accumulate(gt_data, pred_data, sample_token_list, class_name, dist_th)
            metric_data_list.set(class_name, dist_th, md)
    # print(metric_data_list)

    # -----------------------------------
    # Step 2: Calculate metrics from the data.
    # -----------------------------------
    print('Calculating metrics...')
    min_recall = 0.1
    min_precision = 0.1
    metrics = DetectionMetrics()
    for class_name in CLASSES:
        # Compute APs.
        for dist_th in DIST_THS:
            metric_data = metric_data_list[(class_name, dist_th)]
            ap = calc_ap(metric_data, min_recall, min_precision)
            metrics.add_label_ap(class_name, dist_th, ap)
    # =======================================================================

    # Render PR and TP curves.
    # self.render(metrics, metric_data_list)

    # Dump the metric data, meta and metrics to disk.
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
    for name in CLASSES:
        for k, v in metrics["label_aps"][name].items():
            val = float("{:.4f}".format(v))
            detail["object/{}_ap_dist_{}".format(name, k)] = val

    detail["object/map"] = metrics["mean_ap"]


if __name__ == "__main__":
    main()
