import argparse
import copy
import time
import onnx  # 注意这里导入onnx时必须在torch导入之前，否则可能会出现segmentation fault，别人说的
import torch

from torchpack.utils.config import configs
from mmcv import Config

import argparse
import copy

import torch
from torch import nn

from mmcv import Config
from mmcv.runner import load_checkpoint
from demo.utils.config import configs

from typing import Any, Dict, List
from torch.nn import functional as F

from torch.autograd import Function
from torch.nn.modules.utils import _pair
from mmdet3d.ops.voxel.voxel_layer import hard_voxelize

from mmdet3d.models import build_model

import onnxruntime
import numpy as np
from onnxruntime.datasets import get_example

def my_check_model(in_model):
    # 我们可以使用异常处理的方法进行检验
    try:
        # 当我们的模型不可用时，将会报出异常
        onnx.checker.check_model(in_model)
    except onnx.checker.ValidationError as e:
        print("The model is invalid: %s"%e)
    else:
        # 模型可用时，将不会报出异常，并会输出“The model is valid!”
        print("The model is valid!")

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

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


class _Voxelization(Function):
    @staticmethod
    def forward(
        ctx, points, voxel_size, coors_range, max_points:int=35, max_voxels:int=20000, deterministic:bool=True
    ):
        voxels = points.new_zeros(size=(max_voxels, max_points, points.size(1)))
        coors = points.new_zeros(size=(max_voxels, 3), dtype=torch.int)
        num_points_per_voxel = points.new_zeros(size=(max_voxels,), dtype=torch.int)

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

        # select the valid voxels
        voxels_out = voxels[:voxel_num]
        coors_out = coors[:voxel_num]
        num_points_per_voxel_out = num_points_per_voxel[:voxel_num]
        return voxels_out, coors_out, num_points_per_voxel_out

    @staticmethod 
    def symbolic(g, points, voxel_size, coors_range, max_points:int=35, max_voxels:int=20000, deterministic:bool=True):
        """Symbolic function for creating onnx op.""" 
        return g.op( 
            'Kevin::voxelGenerator', 
            points, 
            voxel_size, 
            coors_range,
            max_points,
            max_voxels,
            deterministic) 


voxelization = _Voxelization.apply


class Voxelization(nn.Module):
    def __init__(
        self, voxel_size:float, point_cloud_range:List[float], max_num_points:int, max_voxels:int=20000, deterministic:bool=True
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

    def forward(self, input):
        """
        Args:
            input: NC points
        """
        max_voxels = self.max_voxels[1]

        return voxelization(
            input,
            self.voxel_size,
            self.point_cloud_range,
            self.max_num_points,
            max_voxels,
            self.deterministic,
        )

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "voxel_size=" + str(self.voxel_size)
        tmpstr += ", point_cloud_range=" + str(self.point_cloud_range)
        tmpstr += ", max_num_points=" + str(self.max_num_points)
        tmpstr += ", max_voxels=" + str(self.max_voxels)
        tmpstr += ", deterministic=" + str(self.deterministic)
        tmpstr += ")"
        return tmpstr


class MyModel(torch.nn.Module): 
    def __init__(self,
                 encoders: Dict[str, Any]):
        super().__init__()
        self.Voxelization = Voxelization(**encoders["lidar"]["voxelize"])

    def forward(self, points):
        return self.Voxelization(points)

def get_torch_model():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", metavar="FILE", default='demo/configs/nuscenes/det/transfusion/secfpn/lidar/voxelnet_kevin.yaml')
    parser.add_argument("--checkpoint", type=str, default='exp121_runs_cidi-h5-202211_lidar-only_without-sweep/epoch_50.pth')
    args, opts = parser.parse_known_args()

    configs.load(args.config, recursive=True)
    configs.update(opts)

    cfg = Config(recursive_eval(configs), filename=args.config)

    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark

    # build the model and load checkpoint
    model = MyModel(encoders=cfg.model.encoders)
    # load_checkpoint(model, args.checkpoint, map_location="cpu")

    model = model.to(device)
    model.eval()
    return model

def get_torch_infer():

    point_cuda = point_cpu.to(device)
    print(point_cuda)
    print(point_cuda.size())

    # 得到torch模型的输出
    with torch.no_grad():
        for i in range(1):
            tin = time.time()
            torch_out = torch_model(point_cuda)
            print(time.time()-tin)
        
    print(torch_out[0].size(),torch_out[1].size(),torch_out[2].size())

    with open('out_torch.txt', 'w') as f:
        f.write(str(torch_out))
    
    return torch_out

def torch2onnx():

    # model_jit = torch.jit.script(torch_model)
    # torch.jit.save(model_jit, model_name+'_jit.pt', _extra_files=None)
    
    # model_jit = torch.jit.load(model_name+'_jit.pt')
    # print(model_jit.graph)
    # print('jit model finished'.center(40,'-'))

    output_model_name = model_name + '_ori.onnx'
    input_names = ["input"]  # 输入节点的名称，这里也可以给一个list，list中名称分别对应每一层可学习的参数，便于后续查询
    output_names = ['output']  # 输出节点的名称
    point_cuda = point_cpu.to(device)
    dummy_input = point_cuda

    torch.onnx.export(
        torch_model,
        dummy_input,
        output_model_name,  # 输出文件的名称
        export_params=True, # 如果指定为True或默认, 参数也会被导出. 如果你要导出一个没训练过的就设为 False
        verbose=True,  # 是否以字符串的形式显示计算图
        training=False, # 在训练模式下导出模型。目前，ONNX导出的模型只是为了做推断，所以你通常不需要将其设置为True
        input_names=input_names,
        output_names=output_names,
        do_constant_folding=False,  # 是否压缩常量
        opset_version=13,
        #dynamic_axes={"input": {0: "batch_size", 2: "h"}, "output": {0: "batch_size"}, },  # 设置动态维度，此处指明input节点的第0维度可变，命名为batch_size
        #dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}, }
        )
    
    onnx_model = onnx.load(output_model_name)
    my_check_model(onnx_model)
    graph = onnx.helper.printable_graph(onnx_model.graph)  # 输出onnx的计算图
    with open('onnx_ori.txt', 'w') as f:
        f.write(str(graph))

    print('onnx export finished'.center(40,'-'))

    
    import onnxoptimizer
    passes = ["extract_constant_to_initializer", "eliminate_unused_initializer"]
    onnx_op_model = onnxoptimizer.optimize(onnx_model, passes)
    my_check_model(onnx_op_model)
    graph = onnx.helper.printable_graph(onnx_op_model.graph)  # 输出onnx的计算图
    output_model_name = model_name + '_op.onnx'
    onnx.save(onnx_op_model, output_model_name)
    with open('onnx_op.txt', 'w') as f:
        f.write(str(graph))
    print('onnx onnxoptimizer finished'.center(40,'-'))

    return output_model_name

    from onnxsim import simplify
    # convert model
    model_sim, check = simplify(onnx_op_model)
    assert check, "Simplified ONNX model could not be validated"
    output_model_name = model_name + '_sim.onnx'
    onnx.save(model_sim, output_model_name)
    with open('onnx_sim.txt', 'w') as f:
        f.write(str(graph))
    print('onnxsim simplify finished'.center(40,'-'))

    return model_sim

def get_onnx_infer():

    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device != 'cpu' else ['CPUExecutionProvider']
    print(providers)

    ort_session = onnxruntime.InferenceSession(onnx_model_name, providers=providers)
    print(ort_session.get_providers())
    ort_session.set_providers(['CUDAExecutionProvider'], [{'device_id': 2}])

    ort_inputs = {ort_session.get_inputs()[0].name:to_numpy(point_cpu)}

    for i in range(10):
        tin = time.time()
        ort_output = ort_session.run(None,ort_inputs)
        print(time.time()-tin)

    with open('out_onnx.txt', 'w') as f:
        f.write(str(ort_output))
   
    return ort_output
        




if __name__ == "__main__":

    torch.cuda.set_device(0)
    print(torch.cuda.current_device())
    print(torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    bin_path = '1667440143.700219870_point.bin'
    pc_np = np.fromfile(bin_path, dtype=np.float32)
    pc_np = np.reshape(pc_np, (-1, 5))
    point_cpu = torch.from_numpy(pc_np)

    print(point_cpu)
    print(point_cpu.size())

    model_name = "test"
    
    print('torch model get'.center(40,'-'))
    torch_model = get_torch_model()
    print('torch model get finished'.center(40,'-'))
    
    print('torch model strat'.center(40,'-'))
    torch_out = get_torch_infer()
    print('torch model finished'.center(40,'-'))


    print('onnx model get'.center(40,'-'))
    onnx_model_name = torch2onnx()
    print('onnx model get finished'.center(40,'-'))

    

    # onnx_model_name = model_name + '_op.onnx'
    # print('onnx model strat'.center(40,'-'))
    # onnx_out = get_onnx_infer()
    # print('onnx model finished'.center(40,'-'))

    # 判断输出结果是否一致，小数点后3位一致即可
    # res = np.testing.assert_almost_equal(to_numpy(torch_out), onnx_out, decimal=3)
    # print(res)
