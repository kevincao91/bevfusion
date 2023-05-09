import time
from typing import Any, Dict

import torch
from torch import nn
from torch.nn import functional as F

from mmcv.runner import force_fp32

from mmdet3d.models.builder import (
    build_backbone,
    build_fuser,
    build_head,
    build_neck,
)
from mmdet3d.ops import Voxelization
from mmdet3d.models import FUSIONMODELS


from .base import Base3DFusionModel

__all__ = ["BEVFusion"]


@FUSIONMODELS.register_module()
class BEVFusion(Base3DFusionModel):
    def __init__(
        self,
        encoders: Dict[str, Any],
        fuser: Dict[str, Any],
        decoder: Dict[str, Any],
        heads: Dict[str, Any],
        **kwargs,
    ) -> None:
        super().__init__()

        self.encoders = nn.ModuleDict()
        voxelize_module = Voxelization(**encoders["lidar"]["voxelize"])

        self.encoders['lidar'] = nn.ModuleDict(
            {
                "voxelize": voxelize_module,
                "backbone": build_backbone(encoders["lidar"]["backbone"]),
            }
        )

        self.voxelize_reduce = encoders["lidar"].get("voxelize_reduce", True)

        if fuser is not None:
            self.fuser = build_fuser(fuser)
        else:
            self.fuser = None

        self.decoder = nn.ModuleDict(
            {
                "backbone": build_backbone(decoder["backbone"]),
                "neck": build_neck(decoder["neck"]),
            }
        )
        self.heads = nn.ModuleDict()
        for name in heads:
            if heads[name] is not None:
                self.heads[name] = build_head(heads[name])

        if "loss_scale" in kwargs:
            self.loss_scale = kwargs["loss_scale"]
        else:
            self.loss_scale = dict()
            for name in heads:
                if heads[name] is not None:
                    self.loss_scale[name] = 1.0

        self.f_pad = F.pad
        # ===

    def extract_lidar_features(self, x) -> torch.Tensor:
        feats, coords, sizes = self.voxelize(x)

        batch_size = coords[-1, 0] + 1
        x = self.encoders['lidar']["backbone"](feats, coords, batch_size)

        return x

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        feats_list, coords_list, sizes_list = [], [], []

        ret = self.encoders['lidar']["voxelize"](points)
        f, c, n = ret

        k = 0
        feats_list.append(f)
        coords_list.append(self.f_pad(c, (1, 0), mode="constant", value=float(k)))
        if n is not None:
            sizes_list.append(n)

        feats = torch.cat(feats_list, dim=0)
        coords = torch.cat(coords_list, dim=0)
        sizes = torch.cat(sizes_list, dim=0)
        if self.voxelize_reduce:
            feats = feats.sum(dim=1, keepdim=False) / sizes.type_as(feats).view(
                -1, 1
            )
            feats = feats.contiguous()

        return feats, coords, sizes

    def forward(self, points):
        outputs = self.forward_single(points)
        return outputs

    def forward_single(self, points):  # points  feature
        
        # kevin
        # tin = time.time()
        # print('encoders in', tin)
        feature = self.extract_lidar_features(points)
        features = [feature]
        # torch.cuda.synchronize()
        # tit = time.time() - tin
        # print('encoders out', tit)

        if not self.training:
            # avoid OOM
            features = features[::-1]

        if self.fuser is not None:
            x = self.fuser(features)
        else:
            assert len(features) == 1, features
            x = features[0]

        # kevin
        # tin = time.time()
        # print('decoder in', tin)
        x0, x1 = self.decoder["backbone"](x)
        single_x = self.decoder["neck"](x0, x1)
        # torch.cuda.synchronize()
        # tit = time.time() - tin
        # print('decoder out', tit)
        
        for type, head in self.heads.items():
            if type == "object":
                # kevin
                # tin = time.time()
                # print('head in', tin)
                pred_dict = head(single_x)
                # torch.cuda.synchronize()
                # tit = time.time() - tin
                # print('head out', tit)

                return pred_dict
            else:
                raise ValueError(f"unsupported head: {type}")

