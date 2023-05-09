
## 基础
为了实现BEVFusion代码中激光检测模型的终端部署，进行了部分工作。

首先原始模型在3090下可以实现50ms以下的检测速度，并有提升效果，所以开始尝试部署工作。


模型首先需要转到onnx,有两个部分无法装换，[算子表](https://github.com/onnx/onnx/blob/main/docs/Operators.md)

- 点云体素化 

- 稀疏卷积

## 第一次尝试
将体素化部分用torch代码实现，稀疏卷积替换为普通3D卷积

通过修改，可以全模型转onnx

导出模型结果异常大，运行时间约300s，运行结果不正确

放弃该方案

## 的二次尝试
将体素化操作提出模型，然后通过点云预处理补偿，这部分可以通过调用C的库实现

通过修改，部分模型转出onnx

导出模型结果正常，运行时间约600ms，运行结果正确

这个时间任然无法接受


## 第三次尝试
在bevfusion的基础上

将体素化和稀疏卷积通过自定义算子实现


现广泛使用的2个库 

-有Facebook的官方实现稀疏卷积[库](https://github.com/facebookresearch/SparseConvNet)

-重庆大学的稀疏卷积作者的[库](https://github.com/traveller59/spconv)

一般建议用后者

该仓库提供了点云体素化、稀疏卷积算子的python接口和C++库

所以这样有了制作自定义算子的基础

自定义算子需要两部分：实现和对应的接口

参考[openmmlab的教程](https://www.zhihu.com/column/c_1497987564452114432)




## 后续的工作

-制作onnx体素化和稀疏卷积算子的自定义算子的符号函数，实现通过源代码实现或调用spconv库

-理解libspconv示例中体素化和稀疏卷积算子的操作过程

-制作TensorRT体素化和稀疏卷积算子的hpp和cpp文件，在这两个文件中分别声明和实现插件



体素化算子内容相对单一一点，可以先走通体素化算子的流程，再走通稀疏卷积的流程

具体的

1. 先用单一体素化算子最小Python的简单示例代码运行完

2. 然后添加onnx导出代码，添加符号函数

3. 导出onnx模型后

4. 参考cuda实现参考代码，在TensorRT实现自定义算子

5. 添加TensorRT导出代码

6. 运行TensorRT模型

7. 核对原始模型和onnx模型结果




### 体素化

Python的简单示例代码：

Python的示例代码在[仓库](https://github.com/traveller59/spconv)
example/voxel_gen.py
具体参考函数 main_cuda()


实际模型构建：

在[仓库](https://github.com/mit-han-lab/bevfusion)
mmdet3d/ops/voxel/voxelize.py


cuda实现参考代码：

cuda的示例代码在[仓库](https://github.com/traveller59/spconv)
example/libspconv/main.cu
具体参考函数  Point2VoxelGPU3D::point_to_voxel_hash_static()


### 稀疏卷积

Python的简单示例代码

在[仓库](https://github.com/traveller59/spconv)
example/mnist/mnist_sparse.py
用2D模型作参考


实际模型构建：

在[仓库](https://github.com/mit-han-lab/bevfusion)
mmdet3d/models/backbones/sparse_encoder.py

cuda实现参考代码：

cuda的示例代码在 example/libspconv/main.cu
具体参考函数  Point2VoxelGPU3D::point_to_voxel_hash_static()  之后的代码
具体这部分我没看懂，所以没有更多的提示


## 其他提示

稀疏卷积作者的[库](https://github.com/traveller59/spconv)

感觉可以挖出一些宝藏

好好分析这个工程，特别是example

祝好运！