# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
from torch import nn
from torch.autograd import Function
from torch.nn.modules.utils import _pair

from .voxel_layer import hard_voxelize


class _Voxelization(Function):
    @staticmethod
    def forward(
        ctx, points, voxels, coors, num_points_per_voxel, voxel_size, coors_range, max_points=35, max_voxels=20000, deterministic=True
    ):

        voxel_num = hard_voxelize(
            points,
            voxels,
            coors,
            num_points_per_voxel,
            voxel_size,
            coors_range,
            max_points,
            max_voxels,
            3,
            deterministic,
        )
        return voxel_num

    @staticmethod 
    def symbolic(g, points, voxels, coors, num_points_per_voxel, voxel_size, coors_range, max_points=35, max_voxels=20000, deterministic=True
    ):
        """Symbolic function for creating onnx op.""" 
        return g.op( 
            'Kevin::voxelGenerator', 
            points, 
            voxels,
            coors,
            num_points_per_voxel,
            voxel_size, 
            coors_range,
            max_points,
            max_voxels,
            deterministic) 


voxelization = _Voxelization.apply


class Voxelization(nn.Module):
    def __init__(
        self, voxel_size, point_cloud_range, max_num_points, max_voxels=20000, deterministic=True
    ):
        super(Voxelization, self).__init__()

        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.max_num_points = max_num_points
        if isinstance(max_voxels, tuple):
            self.max_voxels = max_voxels
        else:
            self.max_voxels = _pair(max_voxels)
        self.deterministic = deterministic

        point_cloud_range = torch.tensor(point_cloud_range, dtype=torch.float32)
        # [0, -40, -3, 70.4, 40, 1]
        voxel_size = torch.tensor(voxel_size, dtype=torch.float32)
        grid_size = (point_cloud_range[3:] - point_cloud_range[:3]) / voxel_size
        grid_size = torch.round(grid_size).long()
        input_feat_shape = grid_size[:2]
        self.grid_size = grid_size
        # the origin shape is as [x-len, y-len, z-len]
        # [w, h, d] -> [d, h, w] removed
        self.pcd_shape = [*input_feat_shape, 1]#[::-1]
        # kevin
        print('Voxelization'.center(20,'=')) # [tensor(1440), tensor(1440), 1]
        print(self.__repr__()) # True


    def forward(self, points):
        """
        Args:
            input: NC points
        """
        max_voxels = self.max_voxels[1]

        voxels = points.new_zeros(size=(max_voxels, self.max_num_points, points.size(1)))
        coors = points.new_zeros(size=(max_voxels, 3), dtype=torch.int)
        num_points_per_voxel = points.new_zeros(size=(max_voxels,), dtype=torch.int)

        return voxels, coors, num_points_per_voxel

        #  points, voxels, coors, num_points_per_voxel, voxel_size, coors_range, max_points=35, max_voxels=20000, deterministic=True
        voxel_num = voxelization(
            points,
            voxels,
            coors,
            num_points_per_voxel,
            self.voxel_size,
            self.point_cloud_range,
            self.max_num_points,
            max_voxels,
            self.deterministic,
        )
    
        # select the valid voxels
        voxels_out = voxels[:voxel_num]
        coors_out = coors[:voxel_num]
        num_points_per_voxel_out = num_points_per_voxel[:voxel_num]
        return voxels_out, coors_out, num_points_per_voxel_out


    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "voxel_size=" + str(self.voxel_size)
        tmpstr += ", point_cloud_range=" + str(self.point_cloud_range)
        tmpstr += ", max_num_points=" + str(self.max_num_points)
        tmpstr += ", max_voxels=" + str(self.max_voxels)
        tmpstr += ", deterministic=" + str(self.deterministic)
        tmpstr += ")"
        return tmpstr
    

