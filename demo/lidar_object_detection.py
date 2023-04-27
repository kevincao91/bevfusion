import argparse
import copy
import os

import mmcv
import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDistributedDataParallel
from mmcv.runner import load_checkpoint
from utils.config import configs
# kevin
# from torchpack.utils.tqdm import tqdm
from tqdm import tqdm
import json
import time
import torch.nn.functional as F

from mmdet3d.core import LiDARInstance3DBoxes
from mmdet3d.core.utils import visualize_camera, visualize_lidar, visualize_map
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model

from mmdet3d.core.bbox.coders.transfusion_bbox_coder_kevin import TransFusionBBoxCoder
from mmdet3d.ops.voxel.voxelize_kevin import Voxelization
from mmdet3d.models.backbones.dense_encoder import DenseEncoder

from mmcv.runner import force_fp32


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


class BboxesHead():
    def __init__(
        self,
        num_proposals=128,
        num_classes=4,
        # loss
        loss_cls=dict(type="GaussianFocalLoss", reduction="mean"),
        # others
        bbox_coder=None,
    ):
        self.num_classes = num_classes
        self.num_proposals = num_proposals

        self.use_sigmoid_cls = loss_cls.get("use_sigmoid", False)
        if not self.use_sigmoid_cls:
            self.num_classes += 1

        # ===
        print('bbox_coder:', bbox_coder)
        # self.bbox_coder = build_bbox_coder(bbox_coder)
        pc_range = torch.tensor(bbox_coder['pc_range'])
        out_size_factor = bbox_coder['out_size_factor']
        voxel_size = torch.tensor(bbox_coder['voxel_size'])
        post_center_range = torch.tensor(bbox_coder['post_center_range'])
        score_threshold = bbox_coder['score_threshold']
        code_size = bbox_coder['code_size']
        self.bbox_coder = TransFusionBBoxCoder(pc_range, out_size_factor, voxel_size, post_center_range, score_threshold, code_size)
        # ===

    def get_bboxes(self, preds_dicts):
        # kevin onnx 
        # dict-->list
        # [center,height,dim,rot,vel,heatmap, query_heatmap_score, dense_heatmap, self_query_labels]
        # [     0     1,    2  3 ,  4,   5,            6,             7         ,                 8]
        # rets = []
        
        self_query_labels = preds_dicts[0][0][8]

        preds_dict = preds_dicts[0]
        batch_size = preds_dict[0][5].shape[0]
        batch_score = preds_dict[0][5][..., -self.num_proposals :].sigmoid()
        # if self.loss_iou.loss_weight != 0:
        #    batch_score = torch.sqrt(batch_score * preds_dict[0]['iou'][..., -self.num_proposals:].sigmoid())
        one_hot = F.one_hot(
            self_query_labels, num_classes=self.num_classes
        ).permute(0, 2, 1)
        batch_score = batch_score * preds_dict[0][6] * one_hot

        batch_center = preds_dict[0][0][..., -self.num_proposals :]
        batch_height = preds_dict[0][1][..., -self.num_proposals :]
        batch_dim = preds_dict[0][2][..., -self.num_proposals :]
        batch_rot = preds_dict[0][3][..., -self.num_proposals :]
        batch_vel = None
        # if "vel" in preds_dict[0]:
        batch_vel = preds_dict[0][4][..., -self.num_proposals :]

        temp = self.bbox_coder.decode(
            batch_score,
            batch_rot,
            batch_dim,
            batch_center,
            batch_height,
            batch_vel,
            filter=True,
        )


        # for i in range(batch_size):
        boxes3d = temp[0]["bboxes"]
        scores = temp[0]["scores"]
        labels = temp[0]["labels"]

        ret = dict(bboxes=boxes3d, scores=scores, labels=labels)
        ret_layer= [ret]
        # ===
        rets = [ret_layer]
        # ===
        
        assert len(rets) == 1
        assert len(rets[0]) == 1
        
        # kevin
        res = [
            [
                rets[0][0]["bboxes"],
                rets[0][0]["scores"],
                rets[0][0]["labels"].int(),
            ]
        ]
        return res


class LidarEncoder():
    def __init__(self, encoders):
        self.voxelize_model = Voxelization(**encoders["lidar"]["voxelize"])
        self.sparse_shape = encoders.lidar.backbone.sparse_shape

        from spconv.pytorch.utils import PointToVoxel

        self.gen = PointToVoxel(vsize_xyz=[0.075, 0.075, 0.2],
                                coors_range_xyz=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0],
                                num_point_features=5,
                                max_num_voxels=160000,
                                max_num_points_per_voxel=10)


    def sparse2dense(self, voxel_features, coors, batch_size):
        coors = coors.int()
        
        # input_sp_tensor = SparseConvTensor(
        #     voxel_features, coors, self.sparse_shape, batch_size
        # )
        # input_ds_tensor = input_sp_tensor.dense()
        # kevin only 4 infer
        features = voxel_features
        indices = coors
        if indices.dtype != torch.int32:
            indices.int()
        spatial_shape = self.sparse_shape
        batch_size = batch_size.cpu()

        # def dense
        output_shape = [1, 1440,1440, 41,5]
        indices = indices.long()
        # res = scatter_nd(indices, features, output_shape)
        # def scatter_nd(indices, updates, shape):
        # ret = torch.zeros(*shape, dtype=updates.dtype, device=updates.device)
        updates = features
        ret = torch.zeros(output_shape, dtype=updates.dtype, device=updates.device)
        ndim = indices.shape[-1]
        output_shape = list(indices.shape[:-1])+output_shape[indices.shape[-1] :]   # (n,5)
        flatted_indices = indices.view(-1, ndim)
        for idx, coor in enumerate(flatted_indices):
            b = coor[0]
            x = coor[1]
            y = coor[2]
            z = coor[3]
            ret[b,x,y,z,:] = updates[idx]
        # slices = [flatted_indices[:, i] for i in range(ndim)]
        # slices += [Ellipsis]
        # ret[slices] = updates.view(*output_shape)
        res = ret
        # ===
        ndim = len(spatial_shape)
        # trans_params = list(range(0, ndim + 1))
        # trans_params.insert(1, ndim + 1)
        res = res.permute([0,4,1,2,3]).contiguous()
        
        return res

    def extract_lidar_features(self, x):
        feats_list, coords_list, sizes_list = [], [], []

        f, c, n = self.voxelize_model(x)

        k = 0
        feats_list.append(f)
        coords_list.append(F.pad(c, (1, 0), mode="constant", value=float(k)))
        if n is not None:
            sizes_list.append(n)
        # ===

        feats = torch.cat(feats_list, dim=0)
        coords = torch.cat(coords_list, dim=0)
        sizes = torch.cat(sizes_list, dim=0)
        
        # if self.voxelize_reduce:
        feats = feats.sum(dim=1, keepdim=False) / sizes.type_as(feats).view(
            -1, 1
        )
        feats = feats.contiguous()
        # ===

        batch_size = coords[-1, 0] + 1
        x = self.sparse2dense(feats, coords, batch_size)

        return x
    
    def extract_lidar_features_v2(self, x):
        
        feats_list, coords_list, sizes_list = [], [], []

        f, c, n = self.gen(x)
        index = [2,1,0]
        c = c[:,index]

        k = 0
        feats_list.append(f)
        coords_list.append(F.pad(c, (1, 0), mode="constant", value=float(k)))
        if n is not None:
            sizes_list.append(n)
        # ===

        feats = torch.cat(feats_list, dim=0)
        coords = torch.cat(coords_list, dim=0)
        sizes = torch.cat(sizes_list, dim=0)
        
        # if self.voxelize_reduce:
        feats = feats.sum(dim=1, keepdim=False) / sizes.type_as(feats).view(
            -1, 1
        )
        feats = feats.contiguous()
        # ===

        batch_size = coords[-1, 0] + 1
        x = self.sparse2dense(feats, coords, batch_size)

        return x



def main() -> None:

    parser = argparse.ArgumentParser()
    # parser.add_argument("--config", metavar="FILE", default='demo/configs/nuscenes/det/transfusion/secfpn/lidar/voxelnet_0p075.yaml')
    parser.add_argument("--config", metavar="FILE", default='demo/configs/nuscenes/det/transfusion/secfpn/lidar/voxelnet_kevin.yaml')
    parser.add_argument("--mode", type=str, default="pred", choices=["gt", "pred"])
    parser.add_argument("--checkpoint", type=str, default='exp121_runs_cidi-h5-202211_lidar-only_without-sweep/epoch_50.pth')
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--bbox-classes", nargs="+", type=int, default=None)
    parser.add_argument("--bbox-score", type=float, default=None)   #  测试map=>None, 可视化=>0.5, 筛选推理=>0.3
    parser.add_argument("--map-score", type=float, default=0.5)
    parser.add_argument("--out-dir", type=str, default="demo/outputs")
    args, opts = parser.parse_known_args()

    configs.load(args.config, recursive=True)
    configs.update(opts)

    cfg = Config(recursive_eval(configs), filename=args.config)

    # kevin
    # print(cfg)
    # exit()
    # ===

    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
    torch.cuda.set_device(1)
    print(torch.cuda.current_device())
    print(torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # build the dataloader
    dataset = build_dataset(cfg.data[args.split])
    # dataflow = build_dataloader(
    #     dataset,
    #     samples_per_gpu=1,
    #     workers_per_gpu=cfg.data.workers_per_gpu,
    #     dist=True,
    #     shuffle=False,
    # )
    from torch.utils.data import DataLoader
    dataflow = DataLoader(dataset,batch_size=1,pin_memory=False)

    # build the model and load checkpoint
    if args.mode == "pred":
        model = build_model(cfg.model)
        load_checkpoint(model, args.checkpoint, map_location="cpu")

        # model = MMDistributedDataParallel(
        #     model.cuda(),
        #     device_ids=[torch.cuda.current_device()],
        #     broadcast_buffers=False,
        # )
        model = model.to(device)
        model.eval()

    # kevin
    point_list = []
    time_list = []
    cla_score_map = {
        'Car':0.50,
        'Truck':0.40,
        'Pedestrian':0.40,
        'Rider':0.30,
        'Bus':0.30,
        'Bicycle':0.30,
        'Traffic_Cone':0.50,
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

    # 
    head = cfg.model.heads.object
    print(head)
    num_proposals = head.num_proposals
    num_classes = head.num_classes
    loss_cls = head.loss_cls
    bbox_coder = head.bbox_coder
    bbh = BboxesHead(num_proposals, num_classes, loss_cls, bbox_coder)

    encoders = cfg.model.encoders
    # encoders.lidar.backbone.pop('type')
    lie = LidarEncoder(encoders)

    # ===

    for points, metas in tqdm(dataflow):

        # kevin
        lidar_path = metas[0]['lidar_path'][0]
        lidar_name = os.path.split(lidar_path)[-1][:-8]
        name = lidar_name
        
        point = points[0].squeeze()
        if point.size(0) > 60000:
            point_cpu = point[:60000, :]
        else:
            point_cpu = torch.zeros((60000,5))
            point_cpu[:point.size(0), :] = point

        # point_cuda = point_cpu.to(device)
        # print(point_cuda.size())
        # print(point_cpu.size())
        # point_np = point_cpu.detach().numpy()
        # point_np.tofile(lidar_name+'_point.bin')

        # tin = time.time()
        # print('extract_lidar_features in', tin)
        # lidar_features = lie.extract_lidar_features(point_cpu)
        lidar_features = lie.extract_lidar_features_v2(point_cpu)
        lidar_features = lidar_features.to(device)
        # tit = time.time() - tin
        # print('extract_lidar_features out', tit)
        
        # print(lidar_features.size())
        # lidar_features = lidar_features.detach().cpu().numpy()
        # lidar_features.tofile(lidar_name+'_voxel.bin')

        # print(lidar_features.size())
        # print(lidar_features[0,:,747,755,17])


        # with torch.inference_mode():   # for torch 1.9
        with torch.no_grad():         
            # kevin
            tin = time.time()
            print('model in', tin)
            # outputs = model(**data)
            out_list = model(lidar_features)
            torch.cuda.synchronize()
            tit = time.time() - tin
            print('model out', tit)
            time_list.append(tit)
        # ===
        # pred_dict
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


        # out_list = [center,height,dim,rot,vel,heatmap,query_heatmap_score,dense_heatmap,self_query_labels]
        pred_dict = [[out_list]]
        # copy from 
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


        if args.mode == "pred" and "boxes_3d" in outputs[0]:
            # bboxes = outputs[0]["boxes_3d"].tensor.numpy()
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

            # kevin =======  
            if args.bbox_score is None:
                indices = np.argsort(-scores)
                indices = indices[:20]    # 取分数前20个  可视化时的可选项目，只显示前几个结果
                bboxes = bboxes[indices]
                scores = scores[indices]
                labels = labels[indices]
            # ===

            # class bbox_score
            # print('----------')
            # print('labels:',labels)
            # print('scores:',scores)
            if args.bbox_score is not None:
                cls_msk = []
                for lb, sc in zip(labels, scores):
                    if sc > score_dict[lb]:
                        cls_msk.append(True)
                    else:
                        cls_msk.append(False)
                if len(cls_msk):
                    bboxes = bboxes[cls_msk]
                    scores = scores[cls_msk]
                    labels = labels[cls_msk]
            # print('labels:',labels)
            # print('scores:',scores)
            # print('----------')

            print('bboxes after fliter: ', bboxes.shape)
            
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
        lidar = point_cpu.numpy()
        # kevin
        name = lidar_name
        # ===
        visualize_lidar(
            os.path.join(args.out_dir, "lidar", f"{name}.png"),
            lidar,
            bboxes=bboxes,
            labels=labels,
            xlim=[cfg.point_cloud_range[d] for d in [0, 3]],
            ylim=[cfg.point_cloud_range[d] for d in [1, 4]],
            classes=cfg.object_classes,
        )


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
