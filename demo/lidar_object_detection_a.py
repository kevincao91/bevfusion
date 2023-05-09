import argparse
import os
import json
import time

from tqdm import tqdm
import numpy as np

import mmcv
from mmcv import Config
from mmcv.runner import load_checkpoint

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from mmdet3d.core import LiDARInstance3DBoxes
from mmdet3d.core.utils import visualize_camera, visualize_lidar
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model

from utils.config import configs
from utils.bbox_head import BboxesHead
from utils.utils import recursive_eval


idx2name = {
    0:'Car',
    1:'Truck',
    2:'Pedestrian',
    3:'Rider',
    4:'Bus',
    5:'Bicycle',
    6:'Traffic_Cone',
}


def main() -> None:

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", metavar="FILE", default='demo/configs/nuscenes/det/transfusion/secfpn/lidar/voxelnet_kevin_ab.yaml')
    parser.add_argument("--mode", type=str, default="pred", choices=["gt", "pred"])
    parser.add_argument("--checkpoint", type=str, default='runs/exp121_runs_cidi-h5-202211_lidar-only_without-sweep/epoch_50.pth')
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--bbox-classes", nargs="+", type=int, default=None)
    parser.add_argument("--bbox-score", type=float, default=None)   #  测试map=>None, 可视化=>0.5, 筛选推理=>0.3
    parser.add_argument("--out-dir", type=str, default="demo/outputs")
    args, opts = parser.parse_known_args()

    configs.load(args.config, recursive=True)
    configs.update(opts)

    cfg = Config(recursive_eval(configs), filename=args.config)

    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
    torch.cuda.set_device(1)
    print(torch.cuda.current_device())
    print(torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # build the dataloader
    dataset = build_dataset(cfg.data[args.split])
    dataflow = DataLoader(dataset,batch_size=1,pin_memory=False)

    # build the model and load checkpoint
    model = build_model(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location="cpu")
    model = model.to(device)
    model.eval()

    # ===
    head = cfg.model.heads.object
    num_proposals = head.num_proposals
    num_classes = head.num_classes
    loss_cls = head.loss_cls
    bbox_coder = head.bbox_coder
    bbh = BboxesHead(num_proposals, num_classes, loss_cls, bbox_coder)
    # ===

    for points, metas in tqdm(dataflow):

        # kevin
        lidar_path = metas[0]['lidar_path'][0]
        lidar_name = os.path.split(lidar_path)[-1][:-8]
        name = lidar_name
        
        points_cpu = points[0].squeeze()

        # points_np = points_cpu.detach().numpy()
        # savefile = name + '_point.bin'
        # points_np.tofile(savefile)
        # exit()

        points_cuda = points_cpu.to(device)

        # with torch.inference_mode():   # for torch 1.9
        with torch.no_grad():         
            # kevin
            tin = time.time()
            # print('model in', tin)
            out_list = model(points_cuda)
            torch.cuda.synchronize()
            tit = time.time() - tin
            print('model out', tit)

        # out_list
        # [center,height,dim,rot,vel,heatmap, query_heatmap_score, dense_heatmap, self_query_labels]
        # [     0     1,    2  3 ,  4,   5,            6,             7         ,                 8]
        # torch.Size([1, 2, 200])
        # torch.Size([1, 1, 200])
        # torch.Size([1, 3, 200])
        # torch.Size([1, 2, 200])
        # torch.Size([1, 2, 200])
        # torch.Size([1, 7, 200])
        # torch.Size([1, 7, 200])
        # torch.Size([1, 7, 180, 180])
        # torch.Size([1, 200])

        # ===
        pred_dict = [[out_list]]
        outputs = [{}]
        bboxes = bbh.get_bboxes(pred_dict)
        for k, (boxes, scores, labels) in enumerate(bboxes):
            outputs[k].update(
                {
                    "boxes_3d": boxes.to("cpu"),
                    "scores_3d": scores.cpu(),
                    "labels_3d": labels.cpu(),
                }
            )
        # ===

        if "boxes_3d" in outputs[0]:
            bboxes = outputs[0]["boxes_3d"].detach().numpy()
            scores = outputs[0]["scores_3d"].detach().numpy()
            labels = outputs[0]["labels_3d"].detach().numpy()

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

            # kevin =======  分数项为空时
            if args.bbox_score is None:
                indices = np.argsort(-scores)
                indices = indices[:20]    # 取分数前20个  可视化时的可选项目，只显示前几个结果
                bboxes = bboxes[indices]
                scores = scores[indices]
                labels = labels[indices]
            # ===
            print('bboxes after fliter: ', bboxes.shape)

          
            # 结果持久化 ====
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
                res_list.append(res_data)

            if len_bboxes > 0:
                path_out = os.path.join(args.out_dir, 'model_pred_data', '{}.json'.format(lidar_name))
                mmcv.mkdir_or_exist(os.path.dirname(path_out))
                with open(path_out,"w") as f:
                    json.dump(res_list,f,indent=4)
                    print("{} 写入完成...".format(path_out))
            # ===========

            bboxes[..., 2] -= bboxes[..., 5] / 2
            bboxes = LiDARInstance3DBoxes(bboxes, box_dim=9)
        else:
            bboxes = None
            labels = None

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
        lidar = points_cpu.numpy()
        visualize_lidar(
            os.path.join(args.out_dir, "lidar", f"{name}.png"),
            lidar,
            bboxes=bboxes,
            labels=labels,
            xlim=[cfg.point_cloud_range[d] for d in [0, 3]],
            ylim=[cfg.point_cloud_range[d] for d in [1, 4]],
            classes=cfg.object_classes,
        )

    

if __name__ == "__main__":
    main()
