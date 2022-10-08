import numpy as np


lidar_path = 'data/nuscenes/samples/LIDAR_TOP/n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151603547590.pcd.bin'
points = np.fromfile(lidar_path, dtype=np.float32)
load_dim = 5
points = points.reshape(-1, load_dim)
print(points.shape)

x_list = points[:,0]
y_list = points[:,1]
z_list = points[:,2]
i_list = points[:,3]
r_list = points[:,4]
print('max x:', max(x_list),', min x:',min(x_list))
print('max y:', max(y_list),', min y:',min(y_list))
print('max z:', max(z_list),', min z:',min(z_list))
print('max i:', max(i_list),', min i:',min(i_list))
print('max r:', max(r_list),', min r:',min(r_list))


lidar_path = 'data/nuscenes_cidi/pcd.bin/1639641206.399883986.pcd.bin'
points = np.fromfile(lidar_path, dtype=np.float32)
load_dim = 5
points = points.reshape(-1, load_dim)
print(points.shape)

x_list = points[:,0]
y_list = points[:,1]
z_list = points[:,2]
i_list = points[:,3]
r_list = points[:,4]
print('max x:', max(x_list),', min x:',min(x_list))
print('max y:', max(y_list),', min y:',min(y_list))
print('max z:', max(z_list),', min z:',min(z_list))
print('max i:', max(i_list),', min i:',min(i_list))
print('max r:', max(r_list),', min r:',min(r_list))