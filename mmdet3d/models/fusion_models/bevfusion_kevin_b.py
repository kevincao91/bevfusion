import time
from typing import Any, Dict

import torch
from torch import nn

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


    def forward(self, points):
        outputs = self.forward_single(points)
        return outputs

    def forward_single(self, feature):  # points  feature
        # kevin
        # tin = time.time()
        # print('encoders in', tin)
        feature = self.encoders['lidar']["backbone"](feature)
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

