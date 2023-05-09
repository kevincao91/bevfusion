import torch
from torch import nn as nn

from mmcv.cnn import build_conv_layer, build_norm_layer
from mmdet.models import BACKBONES


@BACKBONES.register_module()
class DenseEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        sparse_shape,
        order=("conv", "norm", "act"),
        norm_cfg=dict(type="BN3d", eps=1e-3, momentum=0.01),
        base_channels=16,
        output_channels=128,
        encoder_channels=((16,), (32, 32, 32), (64, 64, 64), (64, 64, 64)),
        encoder_paddings=((1,), (1, 1, 1), (1, 1, 1), ((0, 1, 1), 1, 1)),
        block_type="conv_module",
    ):
        super().__init__()
        assert block_type in ["conv_module", "basicblock"]
        self.sparse_shape = sparse_shape
        self.in_channels = in_channels
        self.order = order
        self.base_channels = base_channels
        self.output_channels = output_channels
        self.encoder_channels = encoder_channels
        self.encoder_paddings = encoder_paddings
        self.stage_num = len(self.encoder_channels)
        self.fp16_enabled = False
        # Spconv init all weight on its own

        assert isinstance(order, (list, tuple)) and len(order) == 3
        assert set(order) == {"conv", "norm", "act"}

        encoder_out_channels = self.make_encoder_layers(
            self.make_dense_convmodule, norm_cfg, self.base_channels, block_type=block_type
        )

        self.conv_out = self.make_dense_convmodule(
            encoder_out_channels,
            self.output_channels,
            kernel_size=(1, 1, 3),
            stride=(1, 1, 2),
            norm_cfg=norm_cfg,
            padding=0,
            conv_type="Conv3d",
        )


    def forward(self, voxel_features, coors, batch_size):
        
        coors = coors.int()
        
        features = voxel_features
        indices = coors
        if indices.dtype != torch.int32:
            indices.int()
        spatial_shape = self.sparse_shape
        batch_size = batch_size.cpu()

        # make dense features ===
        output_shape = [1, 1440,1440, 41,5]
        indices = indices.long()
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
        res = ret
        # ======

        ndim = len(spatial_shape)
        res = res.permute([0,4,1,2,3]).contiguous()
        
        input_ds_tensor = res
        x =input_ds_tensor

        encode_features = []
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)
            encode_features.append(x)

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        spatial_features = self.conv_out(encode_features[-1])

        N, C, H, W, D = spatial_features.shape
        spatial_features = spatial_features.permute(0, 1, 4, 2, 3).contiguous()
        spatial_features = spatial_features.view(N, C * D, H, W)

        return spatial_features


    def make_dense_convmodule(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        conv_type="Conv3d",
        norm_cfg=None,
        order=("conv", "norm", "act"),
    ):

        assert isinstance(order, tuple) and len(order) <= 3
        assert set(order) | {"conv", "norm", "act"} == {"conv", "norm", "act"}

        conv_cfg = dict(type=conv_type)

        layers = list()
        for layer in order:
            if layer == "conv":
                layers.append(
                    build_conv_layer(
                        conv_cfg,
                        in_channels,
                        out_channels,
                        kernel_size,
                        stride=stride,
                        padding=padding,
                        bias=False,
                    )
                )
            elif layer == "norm":
                layers.append(build_norm_layer(norm_cfg, out_channels)[1])
            elif layer == "act":
                layers.append(nn.ReLU(inplace=True))

        layers = nn.Sequential(*layers)
        return layers


    def make_encoder_layers(
        self,
        make_block,
        norm_cfg,
        in_channels,
        block_type="conv_module",
        conv_cfg=dict(type="Conv3d"),
    ):

        assert block_type in ["conv_module", "basicblock"]
        self.encoder_layers = nn.Sequential()

        for i, blocks in enumerate(self.encoder_channels):
            blocks_list = []
            for j, out_channels in enumerate(tuple(blocks)):
                padding = tuple(self.encoder_paddings[i])[j]
                # each stage started with a spconv layer
                # except the first stage
                if block_type == "basicblock":
                    if j == 0 and i == 0:
                        blocks_list.append(
                            make_block(
                                5,
                                out_channels,
                                3,
                                norm_cfg=norm_cfg,
                                stride=1,
                                padding=padding,
                                conv_type="Conv3d",
                            )
                        )
                    elif j == len(blocks) - 1 and i != len(self.encoder_channels) - 1:
                        blocks_list.append(
                            make_block(
                                in_channels,
                                out_channels,
                                3,
                                norm_cfg=norm_cfg,
                                stride=2,
                                padding=padding,
                                conv_type="Conv3d",
                            )
                        )
                    else:
                        blocks_list.append(
                            DenseBasicBlock(
                                out_channels,
                                out_channels,
                                norm_cfg=norm_cfg,
                                conv_cfg=conv_cfg,
                            )
                        )
                else:
                    blocks_list.append(
                        make_block(
                            in_channels,
                            out_channels,
                            3,
                            norm_cfg=norm_cfg,
                            padding=padding,
                            conv_type="Conv3d",
                        )
                    )
                in_channels = out_channels
            stage_name = f"encoder_layer{i + 1}"
            stage_layers = nn.Sequential(*blocks_list)
            self.encoder_layers.add_module(stage_name, stage_layers)
        return out_channels


from mmcv.runner import BaseModule
class DenseBasicBlock(BaseModule):

    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 dcn=None,
                 plugins=None,
                 init_cfg=None):
        super(DenseBasicBlock, self).__init__(init_cfg)
        assert dcn is None, 'Not implemented yet.'
        assert plugins is None, 'Not implemented yet.'

        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False)
        
        self.bn1 = nn.BatchNorm3d(planes, eps=norm_cfg['eps'], momentum=norm_cfg['momentum'])

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.with_cp = with_cp

    def forward(self, x):
        identity = x

        assert x.dim() == 5, f"x.dim()={x.dim()}"

        out = self.conv1(x)
        out = self.bn1(out)
        out += identity
        out = self.relu(out)

        return out
    


def scatter_nd(indices, updates, shape):

    ret = torch.zeros(shape, dtype=updates.dtype, device=updates.device)
    ndim = indices.shape[-1]
    output_shape = list(indices.shape[:-1]) + shape[indices.shape[-1] :]  # (n,5)
    flatted_indices = indices.view(-1, ndim)
    slices = [flatted_indices[:, i] for i in range(ndim)]
    slices += [Ellipsis]
    ret[slices] = updates.view(*output_shape)
    return ret
