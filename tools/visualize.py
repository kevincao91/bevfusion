import argparse
import copy
import os

import mmcv
import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDistributedDataParallel
from mmcv.runner import load_checkpoint
from torchpack import distributed as dist
from torchpack.utils.config import configs
# kevin
# from torchpack.utils.tqdm import tqdm
from tqdm import tqdm
import json
import time

from mmdet3d.core import LiDARInstance3DBoxes
from mmdet3d.core.utils import visualize_camera, visualize_lidar, visualize_map
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model


def recursive_eval(obj, globals=None):
    if globals is None:
        globals = copy.deepcopy(obj)

    if isinstance(obj, dict):
        for key in obj:
            obj[key] = recursive_eval(obj[key], globals)
    elif isinstance(obj, list):
        for k, val in enumerate(obj):
            obj[k] = recursive_eval(val, globals)
    elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
        obj = eval(obj[2:-1], globals)
        obj = recursive_eval(obj, globals)

    return obj


def main() -> None:
    dist.init()

    parser = argparse.ArgumentParser()
    parser.add_argument("config", metavar="FILE")
    parser.add_argument("--mode", type=str, default="gt", choices=["gt", "pred"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--bbox-classes", nargs="+", type=int, default=None)
    parser.add_argument("--bbox-score", type=float, default=None)   #  测试map=>None, 可视化=>0.5, 筛选推理=>0.3
    parser.add_argument("--map-score", type=float, default=0.5)
    parser.add_argument("--out-dir", type=str, default="viz")
    args, opts = parser.parse_known_args()

    configs.load(args.config, recursive=True)
    configs.update(opts)

    cfg = Config(recursive_eval(configs), filename=args.config)

    # kevin
    # print(cfg)
    # exit()
    # ===

    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
    torch.cuda.set_device(dist.local_rank())

    # build the dataloader
    dataset = build_dataset(cfg.data[args.split])
    dataflow = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=True,
        shuffle=False,
    )

    # build the model and load checkpoint
    if args.mode == "pred":
        model = build_model(cfg.model)
        load_checkpoint(model, args.checkpoint, map_location="cpu")

        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
        )
        model.eval()

    # kevin
    point_list = []
    time_list = []
    cla_score_map = {
        'Car':0.50,
        'Truck':0.30,
        'Pedestrian':0.35,
        'Rider':0.35,
        'Bus':0.35,
        'Bicycle':0.35,
        'Traffic_Cone':0.35,
        }
    idx2name = {
        0:'Car',
        1:'Truck',
        2:'Pedestrian',
        3:'Rider',
        4:'Bus',
        5:'Bicycle',
        6:'Traffic_Cone',
    }
    score_dict = {
        0:cla_score_map['Car'],
        1:cla_score_map['Truck'],
        2:cla_score_map['Pedestrian'],
        3:cla_score_map['Rider'],
        4:cla_score_map['Bus'],
        5:cla_score_map['Bicycle'],
        6:cla_score_map['Traffic_Cone'],
        }
    # ==

    for data in tqdm(dataflow):
        metas = data["metas"].data[0][0]
        # name = "{}-{}".format(metas["timestamp"], metas["token"])

        # kevin
        lidar_path = metas['lidar_path']
        lidar_name = os.path.split(lidar_path)[-1][:-8]
        name = lidar_name
        # print('data["points"].data[0][0].size(0): ', data["points"].data[0][0].size(0))
        point_list.append(data["points"].data[0][0].size(0))
        # if len(point_list)<50:
        #     continue
        # elif len(point_list)>100:
        #     break
        # ==

        if args.mode == "pred":
            with torch.inference_mode():
                # kevin
                tin = time.time()
                outputs = model(**data)
                torch.cuda.synchronize()
                tit = time.time() - tin
                # print(tit)
                time_list.append(tit)
                # ===

        if args.mode == "gt" and "gt_bboxes_3d" in data:
            bboxes = data["gt_bboxes_3d"].data[0][0].tensor.numpy()
            labels = data["gt_labels_3d"].data[0][0].numpy()

            if args.bbox_classes is not None:
                indices = np.isin(labels, args.bbox_classes)
                bboxes = bboxes[indices]
                labels = labels[indices]

            bboxes[..., 2] -= bboxes[..., 5] / 2
            bboxes = LiDARInstance3DBoxes(bboxes, box_dim=9)
        elif args.mode == "pred" and "boxes_3d" in outputs[0]:
            bboxes = outputs[0]["boxes_3d"].tensor.numpy()
            scores = outputs[0]["scores_3d"].numpy()
            labels = outputs[0]["labels_3d"].numpy()

            if args.bbox_classes is not None:
                indices = np.isin(labels, args.bbox_classes)
                bboxes = bboxes[indices]
                scores = scores[indices]
                labels = labels[indices]

            if args.bbox_score is not None:
                indices = scores >= args.bbox_score
                bboxes = bboxes[indices]
                scores = scores[indices]
                labels = labels[indices]

            # kevin =======
            if args.bbox_score is None:
                indices = np.argsort(-scores)
                # indices = indices[:16]    # 取分数前16个  可视化时的可选项目，只显示前几个结果
                bboxes = bboxes[indices]
                scores = scores[indices]
                labels = labels[indices]

            # class bbox_score
            # print('----------')
            # print('labels:',labels)
            # print('scores:',scores)
            # cls_msk = []
            # for lb, sc in zip(labels, scores):
            #     if sc > score_dict[lb]:
            #         cls_msk.append(True)
            #     else:
            #         cls_msk.append(False)
            # if len(cls_msk):
            #     bboxes = bboxes[cls_msk]
            #     scores = scores[cls_msk]
            #     labels = labels[cls_msk]
            # print('labels:',labels)
            # print('scores:',scores)
            # print('----------')

            print('bboxes: ', bboxes.shape)
            
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
                path_out = os.path.join(args.out_dir, 'model_pred_data', '{}.json'.format(lidar_name))
                print(path_out)
                mmcv.mkdir_or_exist(os.path.dirname(path_out))
                with open(path_out,"w") as f:
                    json.dump(res_list,f,indent=4)
                    print("{} 写入完成...".format(path_out))
            
            # exit(0)
            # ===========



            bboxes[..., 2] -= bboxes[..., 5] / 2
            bboxes = LiDARInstance3DBoxes(bboxes, box_dim=9)
        else:
            bboxes = None
            labels = None

        if args.mode == "gt" and "gt_masks_bev" in data:
            masks = data["gt_masks_bev"].data[0].numpy()
            masks = masks.astype(np.bool)
        elif args.mode == "pred" and "masks_bev" in outputs[0]:
            masks = outputs[0]["masks_bev"].numpy()
            masks = masks >= args.map_score
        else:
            masks = None

        # if "img" in data:
        #     for k, image_path in enumerate(metas["filename"]):
        #         image = mmcv.imread(image_path)
        #         visualize_camera(
        #             os.path.join(args.out_dir, f"camera-{k}", f"{name}.png"),
        #             image,
        #             bboxes=bboxes,
        #             labels=labels,
        #             transform=metas["lidar2image"][k],
        #             classes=cfg.object_classes,
        #         )

        # if "points" in data:
        #     lidar = data["points"].data[0][0].numpy()
        #     # kevin
        #     name = lidar_name
        #     # ===
        #     visualize_lidar(
        #         os.path.join(args.out_dir, "lidar", f"{name}.png"),
        #         lidar,
        #         bboxes=bboxes,
        #         labels=labels,
        #         xlim=[cfg.point_cloud_range[d] for d in [0, 3]],
        #         ylim=[cfg.point_cloud_range[d] for d in [1, 4]],
        #         classes=cfg.object_classes,
        #     )

        # if masks is not None:
        #     visualize_map(
        #         os.path.join(args.out_dir, "map", f"{name}.png"),
        #         masks,
        #         classes=cfg.map_classes,
        #     )

    # kevin
    for po,ti in zip(point_list, time_list):
        print(po, ti)

    points = np.array(point_list[1:])
    times = np.array(time_list[1:])
    print('[1:]  np.average(points) {} , np.average(times) {} ms, len {}'.format(np.average(points), 1000*np.average(times), times.shape))
    points = np.array(point_list[2:])
    times = np.array(time_list[2:])
    print('[2:]  np.average(points) {} , np.average(times) {} ms, len {}'.format(np.average(points), 1000*np.average(times), times.shape))
    
    out_path = os.path.join(args.out_dir, "infer_time.txt")
    with open(out_path, 'w') as f:
        for po,ti in zip(point_list, time_list):
            f.write('{} {}\n'.format(po, ti))
    # ==
    

if __name__ == "__main__":
    main()
