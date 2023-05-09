import copy

import numpy as np
import torch
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_conv_layer
from mmcv.runner import force_fp32
from torch import nn

from mmdet3d.core import (
    PseudoSampler,
    circle_nms,
    draw_heatmap_gaussian,
    gaussian_radius,
    xywhr2xyxyr,
)
from mmdet3d.models.builder import HEADS, build_loss
from mmdet3d.models.utils import FFN, PositionEmbeddingLearned, TransformerDecoderLayer
from mmdet3d.ops.iou3d.iou3d_utils import nms_gpu
from mmdet.core import (
    AssignResult,
    build_assigner,
    build_bbox_coder,
    build_sampler,
    multi_apply,
)

from ....core.bbox import LiDARInstance3DBoxes

from typing import List

__all__ = ["TransFusionHead"]


def clip_sigmoid(x, eps=1e-4):
    y = torch.clamp(x.sigmoid_(), min=eps, max=1 - eps)
    return y


@HEADS.register_module()
class TransFusionHead(nn.Module):
    def __init__(
        self,
        num_proposals=128,
        auxiliary=True,
        in_channels=128 * 3,
        hidden_channel=128,
        num_classes=4,
        # config for Transformer
        num_decoder_layers=3,
        num_heads=8,
        nms_kernel_size=1,
        ffn_channel=256,
        dropout=0.1,
        bn_momentum=0.1,
        activation="relu",
        # config for FFN
        common_heads=dict(),
        num_heatmap_convs=2,
        conv_cfg=dict(type="Conv1d"),
        norm_cfg=dict(type="BN1d"),
        bias="auto",
        # loss
        loss_cls=dict(type="GaussianFocalLoss", reduction="mean"),
        loss_iou=dict(
            type="VarifocalLoss", use_sigmoid=True, iou_weighted=True, reduction="mean"
        ),
        loss_bbox=dict(type="L1Loss", reduction="mean"),
        loss_heatmap=dict(type="GaussianFocalLoss", reduction="mean"),
        # others
        train_cfg=None,
        test_cfg=None,
        bbox_coder=None,
    ):
        super(TransFusionHead, self).__init__()

        self.fp16_enabled = False

        self.num_classes = num_classes
        self.num_proposals = num_proposals
        self.auxiliary = auxiliary
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.num_decoder_layers = num_decoder_layers
        self.bn_momentum = bn_momentum
        self.nms_kernel_size = nms_kernel_size
        self.train_cfg = train_cfg
        # self.test_cfg = test_cfg
        # kevin onnx
        grid_size = test_cfg['grid_size']
        out_size_factor = test_cfg['out_size_factor']
        # ===

        self.use_sigmoid_cls = loss_cls.get("use_sigmoid", False)
        if not self.use_sigmoid_cls:
            self.num_classes += 1
        # kevin onnx
        # self.loss_cls = build_loss(loss_cls)
        # self.loss_bbox = build_loss(loss_bbox)
        # self.loss_iou = build_loss(loss_iou)
        # self.loss_heatmap = build_loss(loss_heatmap)
        # ===
        
        self.sampling = False

        # a shared convolution
        self.shared_conv = build_conv_layer(
            dict(type="Conv2d"),
            in_channels,
            hidden_channel,
            kernel_size=3,
            padding=1,
            bias=bias,
        )

        layers = []
        layers.append(
            ConvModule(
                hidden_channel,
                hidden_channel,
                kernel_size=3,
                padding=1,
                bias=bias,
                conv_cfg=dict(type="Conv2d"),
                norm_cfg=dict(type="BN2d"),
            )
        )
        layers.append(
            build_conv_layer(
                dict(type="Conv2d"),
                hidden_channel,
                num_classes,
                kernel_size=3,
                padding=1,
                bias=bias,
            )
        )
        self.heatmap_head = nn.Sequential(*layers)
        self.class_encoding = nn.Conv1d(num_classes, hidden_channel, 1)

        # transformer decoder layers for object query with LiDAR feature
        self.decoder = nn.ModuleList()
        for i in range(self.num_decoder_layers):
            self.decoder.append(
                TransformerDecoderLayer(
                    hidden_channel,
                    num_heads,
                    ffn_channel,
                    dropout,
                    activation,
                    self_posembed=PositionEmbeddingLearned(2, hidden_channel),
                    cross_posembed=PositionEmbeddingLearned(2, hidden_channel),
                )
            )

        # Prediction Head
        self.prediction_heads = nn.ModuleList()
        for i in range(self.num_decoder_layers):
            heads = copy.deepcopy(common_heads)
            heads.update(dict(heatmap=(self.num_classes, num_heatmap_convs)))
            self.prediction_heads.append(
                FFN(
                    hidden_channel,
                    heads,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    bias=bias,
                )
            )

        self.init_weights()
        self._init_assigner_sampler()

        # Position Embedding for Cross-Attention, which is re-used during training
        x_size = grid_size[0] // out_size_factor
        y_size = grid_size[1] // out_size_factor
        self.bev_pos = self.create_2D_grid(x_size, y_size)

        self.img_feat_pos = None
        self.img_feat_collapsed_pos = None

    def create_2D_grid(self, x_size, y_size):
        meshgrid = [[0, x_size - 1, x_size], [0, y_size - 1, y_size]]
        # NOTE: modified
        batch_x, batch_y = torch.meshgrid(
            *[torch.linspace(it[0], it[1], it[2]) for it in meshgrid]
        )
        batch_x = batch_x + 0.5
        batch_y = batch_y + 0.5
        coord_base = torch.cat([batch_x[None], batch_y[None]], dim=0)[None]
        coord_base = coord_base.view(1, 2, -1).permute(0, 2, 1)
        return coord_base

    def init_weights(self):
        # initialize transformer
        for m in self.decoder.parameters():
            if m.dim() > 1:
                nn.init.xavier_uniform_(m)
        if hasattr(self, "query"):
            nn.init.xavier_normal_(self.query)
        self.init_bn_momentum()

    def init_bn_momentum(self):
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.momentum = self.bn_momentum

    def _init_assigner_sampler(self):
        """Initialize the target assigner and sampler of the head."""
        if self.train_cfg is None:
            return

        if self.sampling:
            self.bbox_sampler = build_sampler(self.train_cfg.sampler)
        else:
            self.bbox_sampler = PseudoSampler()
        if isinstance(self.train_cfg.assigner, dict):
            self.bbox_assigner = build_assigner(self.train_cfg.assigner)
        elif isinstance(self.train_cfg.assigner, list):
            self.bbox_assigner = [
                build_assigner(res) for res in self.train_cfg.assigner
            ]

    # def forward_single(self, inputs, img_inputs, metas):
    def forward_single(self, inputs):
        """Forward function for CenterPoint.
        Args:
            inputs (torch.Tensor): Input feature map with the shape of
                [B, 512, 128(H), 128(W)]. (consistent with L748)
        Returns:
            list[dict]: Output results for tasks.
        """
        batch_size = inputs.shape[0]
        lidar_feat = self.shared_conv(inputs)

        #################################
        # image to BEV
        #################################
        lidar_feat_flatten = lidar_feat.view(
            batch_size, lidar_feat.shape[1], -1
        )  # [BS, C, H*W]

        # kevin
        # print('TransFusionHead self.bev_pos'.center(20,'='))
        # print(self.bev_pos.size())
        # ===

        bev_pos = self.bev_pos.repeat(batch_size, 1, 1).to(lidar_feat.device)

        # kevin
        # print('TransFusionHead bev_pos'.center(20,'='))
        # print(bev_pos.size())
        # ===

        #################################
        # image guided query initialization
        #################################
        dense_heatmap = self.heatmap_head(lidar_feat)
        # dense_heatmap_img = None
        heatmap = dense_heatmap.detach().sigmoid()
        # kevin
        # print('TransFusionHead heatmap'.center(20,'='))
        # print(heatmap.size())
        # print(heatmap)
        # ===
        padding = self.nms_kernel_size // 2
        local_max = torch.zeros_like(heatmap)
        # equals to nms radius = voxel_size * out_size_factor * kenel_size
        local_max_inner = F.max_pool2d(
            heatmap, kernel_size=self.nms_kernel_size, stride=1, padding=0
        )
        local_max[:, :, padding:(-padding), padding:(-padding)] = local_max_inner


        # kevin
        # print('TransFusionHead local_max'.center(20,'='))
        # print(local_max.size())
        # print(local_max)
        # ===

        ## for Pedestrian & Traffic_cone in nuScenes
        local_max[:, 2] = heatmap[:, 2]
        local_max[:, 3] = heatmap[:, 3]
        local_max[:, 5] = heatmap[:, 5]
        local_max[:, 6] = heatmap[:, 6]

        '''
        local_max[
            :,
            2,
        ] = F.max_pool2d(heatmap[:, 2], kernel_size=1, stride=1, padding=0)
        local_max[
            :,
            3,
        ] = F.max_pool2d(heatmap[:, 3], kernel_size=1, stride=1, padding=0)
        local_max[
            :,
            5,
        ] = F.max_pool2d(heatmap[:, 5], kernel_size=1, stride=1, padding=0)
        local_max[
            :,
            6,
        ] = F.max_pool2d(heatmap[:, 6], kernel_size=1, stride=1, padding=0)
        '''
        # =====

        heatmap = heatmap * (heatmap == local_max)  # Kevin 这里的操作像是非极大值抑制，只突出局部最大值，非局部最大值置零，在每个类别特征层面操作
        heatmap = heatmap.view(batch_size, heatmap.shape[1], -1)
        # kevin
        # print('TransFusionHead heatmap'.center(20,'='))
        # print(heatmap.size())
        # print(heatmap)
        # ===

        # top #num_proposals among all classes
        # top_proposals = heatmap.view(batch_size, -1).argsort(dim=-1, descending=True)[
        #     ..., : self.num_proposals
        # ]
        # torch.Size([1, 200])

        # kevin onnx
        v, inds = heatmap.view(batch_size, -1).sort(dim=-1, descending=True)
        top_proposals = inds[
            ..., : self.num_proposals
        ]
        # ===

        # kevin
        # print('TransFusionHead top_proposals'.center(20,'='))
        # print(top_proposals.size())
        # print(top_proposals)
        # ===

        top_proposals_class = top_proposals // heatmap.shape[-1]
        top_proposals_index = top_proposals % heatmap.shape[-1]
        query_feat = lidar_feat_flatten.gather(
            index=top_proposals_index[:, None, :].expand(
                -1, lidar_feat_flatten.shape[1], -1
            ),
            dim=-1,
        )
        out_self_query_labels = top_proposals_class
        # kevin
        # print('TransFusionHead out_self_query_labels'.center(20,'='))
        # print(out_self_query_labels.size())
        # print(out_self_query_labels)
        # ===

        # add category embedding
        one_hot = F.one_hot(top_proposals_class, num_classes=self.num_classes).permute(
            0, 2, 1
        )
        # kevin
        # print('TransFusionHead one_hot'.center(20,'='))
        # print(one_hot.size())  # torch.Size([1, 7, 200])
        # print(one_hot)
        # ===

        query_cat_encoding = self.class_encoding(one_hot.float())
        query_feat += query_cat_encoding    # 特征和类别相加 尺度不变
        
        # kevin
        # print('TransFusionHead query_feat'.center(20,'='))
        # print(query_feat.size())  # torch.Size([1, 128, 200])
        # print(query_feat)
        # ===

        query_pos = bev_pos.gather(
            index=top_proposals_index[:, None, :]
            .permute(0, 2, 1)
            .expand(-1, -1, bev_pos.shape[-1]),
            dim=1,
        )

        # kevin
        # print('TransFusionHead query_pos'.center(20,'='))
        # print(query_pos.size())  # torch.Size([1, 128, 200])
        # print(query_pos)
        # ===

        #################################
        # transformer decoder layer (LiDAR feature as K,V)
        #################################
        # ret_dicts = []
        # for i in range(self.num_decoder_layers):
        for i, decoder in enumerate(self.decoder):
            prefix = "last_" if (i == self.num_decoder_layers - 1) else f"{i}head_"

            # Transformer Decoder Layer
            # :param query: B C Pq    :param query_pos: B Pq 3/6
            query_feat = self.decoder[0](
                query_feat, lidar_feat_flatten, query_pos, bev_pos
            )

            # kevin
            # print('TransFusionHead decoder_layers {}  query_feat'.format(i).center(20,'='))
            # print(query_feat.size())  # torch.Size([1, 128, 200])
            # print(query_pos)
            # ===

            # Prediction
            # [center,height,dim,rot,vel,heatmap]
            # [     0     1,    2  3 ,  4,   5, ]
            res_layer = self.prediction_heads[0](query_feat)
            res_layer[0] = res_layer[0] + query_pos.permute(0, 2, 1)
            first_res_layer = res_layer
            ret_dicts = [res_layer]

            # for next level positional embedding
            query_pos = res_layer[0].detach().clone().permute(0, 2, 1)

        #################################


        query_heatmap_score = heatmap.gather(
            index=top_proposals_index[:, None, :].expand(-1, self.num_classes, -1),
            dim=-1,
        )  # [bs, num_classes, num_proposals]


        # if self.auxiliary is False:
        #     # only return the results of last decoder layer
        #     return [ret_dicts[-1]]

        # return all the layer's results for auxiliary superivison
        # new_res = {}
        # for key in ret_dicts[0].keys():
        #     # # [center,height,dim,rot,vel,heatmap, query_heatmap_score, dense_heatmap]
        #     if key not in ["dense_heatmap", "dense_heatmap_old", "query_heatmap_score"]:
        #         new_res[key] = torch.cat(
        #             [ret_dict[key] for ret_dict in ret_dicts], dim=-1
        #         )
        #     else:
        #         new_res[key] = ret_dicts[0][key]
        new_res = ret_dicts[0] # batch_size==1 skip!
        
        # kevin onnx 
        # dict-->list
        # [center,height,dim,rot,vel,heatmap, query_heatmap_score, dense_heatmap]
        return new_res[0],new_res[1],new_res[2],new_res[3],new_res[4],new_res[5],query_heatmap_score,dense_heatmap,out_self_query_labels


    def forward(self, feats):
    # def forward(self, feats, metas=None):
        """Forward pass.
        Args:
            feats (list[torch.Tensor]): Multi-level features, e.g.,
                features produced by FPN.
        Returns:
            tuple(list[dict]): Output results. first index by level, second index by layer
        """
        if isinstance(feats, torch.Tensor):
            feats = [feats]
        # kevin onnx
        # res = multi_apply(self.forward_single, feats, [None], [metas])
        # print('===> res:', res)
        feats = feats[0]
        res = self.forward_single(feats)
        return res


    