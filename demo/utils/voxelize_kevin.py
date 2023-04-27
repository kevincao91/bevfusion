# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
from torch import nn
from torch.nn.modules.utils import _pair



def dynamic_voxelize_kernel(points, coors, voxel_size, coors_range, grid_size,
                            num_points:int, NDim:int=3):
    failed = False
    # // int coor[NDim];
    coor = -1*torch.ones(NDim)

    for i in range(num_points):
        failed = False
        for j in range(NDim):
            c = torch.floor((points[i][j] - coors_range[j]) / voxel_size[j])
            # // necessary to rm points out of range
            if c < 0 or c >= grid_size[j]:
                failed = True
                break
            coor[j] = c

        for k in range(NDim):
            if failed:
                coors[i][k] = -1
            else:
                coors[i][k] = coor[k]


def hard_voxelize_kernel(points, voxels, coors, num_points_per_voxel, coor_to_voxelidx,
                          voxel_size, coors_range, grid_size,
                          max_points:int, max_voxels:int,
                          num_points:int, num_features:int,
                          NDim:int=3):
  
    # // declare a temp coors
    temp_coors = torch.zeros(num_points, NDim, dtype=torch.long)

    # // First use dynamic voxelization to get coors,
    # // then check max points/voxels constraints
    dynamic_voxelize_kernel(points, temp_coors,
                            voxel_size, coors_range, grid_size,
                            num_points)

    coor = temp_coors

    voxel_num = torch.tensor(0)
    for i in range(num_points):
        # // T_int* coor = temp_coors.data_ptr<int>() + i * NDim;

        if coor[i][0] == -1:
            continue

        voxelidx = coor_to_voxelidx[coor[i][0]][coor[i][1]][coor[i][2]]

        # // record voxel
        if voxelidx == -1:
            voxelidx = voxel_num.clone()
            if max_voxels != -1 and voxel_num >= max_voxels:
                continue
            voxel_num += 1

            coor_to_voxelidx[coor[i][0]][coor[i][1]][coor[i][2]] = voxelidx

            for k in range(NDim):
                coors[voxelidx][k] = coor[i][k]


        # // put points into voxel
        num = num_points_per_voxel[voxelidx]
        if max_points == -1 or num < max_points:
            for k in range(num_features):
                voxels[voxelidx][num][k] = points[i][k]
            num_points_per_voxel[voxelidx] += 1

    return voxel_num


def hard_voxelize_cpu(points, voxels, coors, num_points_per_voxel,
                      voxel_size, coors_range, max_points:int, max_voxels:int, NDim:int=3):
    
    grid_size = torch.zeros(NDim, dtype=torch.int64)
    num_points = points.size(0)
    num_features = points.size(1)

    for i in range(NDim):
        grid_size[i] = torch.round((coors_range[NDim+i] - coors_range[i]) / voxel_size[i])

    # // coors, num_points_per_voxel, coor_to_voxelidx are int Tensor
    # // printf("cpu coor_to_voxelidx size: [%d, %d, %d]\n", grid_size[2],
    # // grid_size[1], grid_size[0]);
    # coor_to_voxelidx = -1*torch.ones(grid_size[2], grid_size[1], grid_size[0],dtype=torch.int)
    coor_to_voxelidx_ = -1*torch.ones(grid_size[0], grid_size[1], grid_size[2])
    coor_to_voxelidx = coor_to_voxelidx_.long()
   
    voxel_num = hard_voxelize_kernel(
                points, voxels, coors, num_points_per_voxel, coor_to_voxelidx, 
                voxel_size, coors_range, grid_size, max_points, max_voxels, num_points,
                num_features)

    return voxel_num


class Voxelization(nn.Module):
    def __init__(
        self, voxel_size, point_cloud_range, max_num_points, max_voxels=20000, deterministic=True
    ):
        super(Voxelization, self).__init__()
        """
        Args:
            voxel_size (list): list [x, y, z] size of three dimension
            point_cloud_range (list):
                [x_min, y_min, z_min, x_max, y_max, z_max]
            max_num_points (int): max number of points per voxel
            max_voxels (tuple or int): max number of voxels in
                (training, testing) time
            deterministic: bool. whether to invoke the non-deterministic
                version of hard-voxelization implementations. non-deterministic
                version is considerablly fast but is not deterministic. only
                affects hard voxelization. default True. for more information
                of this argument and the implementation insights, please refer
                to the following links:
                https://github.com/open-mmlab/mmdetection3d/issues/894
                https://github.com/open-mmlab/mmdetection3d/pull/904
                it is an experimental feature and we will appreciate it if
                you could share with us the failing cases.
        """
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
        print('Voxelization (from voxelize_kevin)'.center(20,'=')) # [tensor(1440), tensor(1440), 1]
        print(self.__repr__()) # True


    def forward(self, input):
        """
        Args:
            input: NC points
        """
        if self.training:
            max_voxels = self.max_voxels[0]
        else:
            max_voxels = self.max_voxels[1]

        # copy from ops/voxel/voxelize.py
        points = input
        voxel_size = torch.tensor(self.voxel_size)
        coors_range = torch.tensor(self.point_cloud_range)
        max_points = self.max_num_points
        max_voxels = max_voxels
        
        assert max_points != -1 and max_voxels != -1

        voxels = points.new_zeros(size=(max_voxels, max_points, points.size(1)))
        coors = points.new_zeros(size=(max_voxels, 3))
        num_points_per_voxel = points.new_zeros(size=(max_voxels,), dtype=torch.int)

        # kevin
        # print('_Voxelization'.center(20,'='), voxels.shape, coors.shape, num_points_per_voxel.shape) # [tensor(1440), tensor(1440), 1]
        # torch.Size([120000, 10, 5]) torch.Size([120000, 3]) torch.Size([120000])

        voxel_num = hard_voxelize_cpu(
            points,
            voxels,
            coors,
            num_points_per_voxel,
            voxel_size,
            coors_range,
            max_points,
            max_voxels,
        )
        # kevin
        # print('_Voxelization'.center(20,'='), voxel_num) # 41631
        # with open("CPU.txt", 'w') as f:
        #     for i in range(coors.size(0)):
        #         f.write(str(coors[i].detach().cpu().numpy())+'\n')
        # CPU
        # for i in range(76,100,1):
        #     print(i,coors[i])
        #     print(i,voxels[i])

        # exit()

        # select the valid voxels
        voxels_out = voxels[:voxel_num]
        coors_out = coors[:voxel_num]
        num_points_per_voxel_out = num_points_per_voxel[:voxel_num]
        return voxels_out, coors_out, num_points_per_voxel_out

        # ===



    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "voxel_size=" + str(self.voxel_size)
        tmpstr += ", point_cloud_range=" + str(self.point_cloud_range)
        tmpstr += ", max_num_points=" + str(self.max_num_points)
        tmpstr += ", max_voxels=" + str(self.max_voxels)
        tmpstr += ", deterministic=" + str(self.deterministic)
        tmpstr += ")"
        return tmpstr
