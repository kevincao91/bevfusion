import numpy as np
# import mayavi.mlab
from scipy.spatial.transform import Rotation as R
from pyquaternion import Quaternion as Q

# 点云显示
def show_pc(pc):
    # 取值
    x = pc[:, 0]  # x position of point
    y = pc[:, 1]  # y position of point
    z = pc[:, 2]  # z position of point
    if pc.shape[-1] >= 4:
        r = pc[:, 3]  # reflectance value of point
    d = np.sqrt(x ** 2 + y ** 2)  # Map Distance from sensor

    # vals = 'height'
    vals = 'distance'
    if vals == "height":
        col = z
    else:
        col = d

    fig = mayavi.mlab.figure(bgcolor=(0, 0, 0), size=(640, 500))
    mayavi.mlab.points3d(x, y, z,
                        col,  # Values used for Color
                        mode="point",
                        colormap='spectral',  # 'bone', 'copper', 'gnuplot'
                        # color=(0, 1, 0),   # Used a fixed (r,g,b) instead
                        figure=fig,
                        scale_factor=0.25
                        )

    mayavi.mlab.show(stop=True)



# 逆畸变
def dedistort(p, D):
    # r_sq
    pp = p.copy()
    # print(p)
    for i in range(p.shape[1]):
        rr = p[0, i] * p[0, i] + p[1, i] * p[1, i]
        # radial_0
        radial = (lambda x: p[:, i] * (1 + D[0] * x + D[1] * x * x + D[4] * x * x * x))(rr)
        pp[0, i] = 2 * D[2] * p[0, i] * p[1, i] + D[3] * (rr + 2 * p[0, i] * p[0, i]) + radial[0]
        pp[1, i] = 2 * D[3] * p[0, i] * p[1, i] + D[2] * (rr + 2 * p[1, i] * p[1, i]) + radial[1]

    return pp


# 旋转矩阵到四元数
def rotmat2quaternion_sci(rm):
    # scipy算法实现
    # 函数输出是[x,y,z,w]
    r = R.from_matrix(rm)
    quat_ = r.as_quat()
    return quat_


# 四元数到旋转矩阵
def quaternion2rotmat_sci(quat_xyzw):
    # scipy算法实现
    # 函数输入需求是[x,y,z,w]
    r = R.from_quat(quat_xyzw)
    rm = r.as_matrix()
    return rm

# 四元数到旋转矩阵
def quaternion2rotmat(quat_list, mode='Ham'):
    # print(mode.center(21, '='))

    qq = np.array(quat_list)
    m4q = sum(qq*qq)
    assert abs(m4q-1)<1e-3, 'The modulus of the quaternion {} != 1'.format(quat_list)
    
    if mode == 'Ham':
        # Extract the values from quat_list[w, x, y, z]
        w, x, y, z = quat_list
        # First row of the rotation matrix
        r00 = 1 - 2 * (y**2 + z**2)
        r01 = 2 * (x * y - z * w)
        r02 = 2 * (x * z + y * w)
        # Second row of the rotation matrix
        r10 = 2 * (x * y + z * w)
        r11 = 1 - 2 * (x**2 + z**2)
        r12 = 2 * (y * z - x * w)
        # Third row of the rotation matrix
        r20 = 2 * (x * z - y * w)
        r21 = 2 * (y * z + x * w)
        r22 = 1 - 2 * (x**2 + y**2)
        # 3x3 rotation matrix
        rotmat = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]], dtype=np.float32)
    elif mode=='JPL':
        # Extract the values from quat_list[x, y, z, w]
        x, y, z, w = quat_list
        # First row of the rotation matrix
        r00 = 1 - 2 * (y**2 + z**2)
        r01 = 2 * (x * y + z * w)
        r02 = 2 * (x * z - y * w)
        # Second row of the rotation matrix
        r10 = 2 * (x * y - z * w)
        r11 = 1 - 2 * (x**2 + z**2)
        r12 = 2 * (y * z - x * w)
        # Third row of the rotation matrix
        r20 = 2 * (x * z + y * w)
        r21 = 2 * (y * z + x * w)
        r22 = 1 - 2 * (x**2 + y**2)
        # 3x3 rotation matrix
        rotmat = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]], dtype=np.float32)
    else:
        assert 1==2, "mode 不存在！"

    return rotmat


# 欧拉角到旋转矩阵
def euler2rotmat_zxy_ex(euler_list):
    # 得到对应旋转顺序是Z-X-Y的外旋（左乘）的旋转矩阵

    # 得到绕z轴旋转的旋转矩阵
    rot_euler = euler_list[0]
    Rz = np.array([
                  [   np.cos(rot_euler),  -np.sin(rot_euler),                  0],
                  [   np.sin(rot_euler),   np.cos(rot_euler),                  0],
                  [                   0,                   0,                  1]])
    # 得到绕x轴旋转的旋转矩阵
    rot_euler = euler_list[1]
    Rx = np.array([
                  [                   1,                   0,                  0],
                  [                   0,   np.cos(rot_euler), -np.sin(rot_euler)],
                  [                   0,   np.sin(rot_euler),  np.cos(rot_euler)]])
    # 得到绕y轴旋转的旋转矩阵
    rot_euler = euler_list[2]
    Ry = np.array([
                  [   np.cos(rot_euler),                   0,  np.sin(rot_euler)],
                  [                   0,                   1,                  0],
                  [  -np.sin(rot_euler),                   0,  np.cos(rot_euler)]])
    rotmat = Ry @ Rx @ Rz
    
    return rotmat

# 欧拉角到旋转矩阵
def euler2rotmat_YXZ_in(euler_list):
    # 得到对应旋转顺序是Y-X-Z的内旋（右乘）的旋转矩阵
    # 得到绕y轴旋转的旋转矩阵
    rot_euler = euler_list[0]
    Ry = np.array([
                  [   np.cos(rot_euler),                   0,  np.sin(rot_euler)],
                  [                   0,                   1,                  0],
                  [  -np.sin(rot_euler),                   0,  np.cos(rot_euler)]])
    # 得到绕x轴旋转的旋转矩阵
    rot_euler = euler_list[1]
    Rx = np.array([
                  [                   1,                   0,                  0],
                  [                   0,   np.cos(rot_euler), -np.sin(rot_euler)],
                  [                   0,   np.sin(rot_euler),  np.cos(rot_euler)]])
    # 得到绕z轴旋转的旋转矩阵
    rot_euler = euler_list[2]
    Rz = np.array([
                  [   np.cos(rot_euler),  -np.sin(rot_euler),                  0],
                  [   np.sin(rot_euler),   np.cos(rot_euler),                  0],
                  [                   0,                   0,                  1]])

    rotmat = Ry @ Rx @ Rz
    
    return rotmat


if __name__=="__main__":
    
    # ===========================================================
    # alpha beta gamma
    euler_list_ = [10, 20, 30]
    euler_list_ = np.array(euler_list_, dtype=np.float32)
    euler_list_ = euler_list_ * np.pi / 180.0    # 角度到弧度
    
    # scipy算法验证
    # 'zxy' 为外旋（左乘），先绕z轴alpha,再绕x轴beta,再绕y轴gamma  => rotmat = Ry(gamma) @ Rx(beta) @ Rz(alpha)
    r = R.from_euler('zxy', euler_list_, degrees=False)
    rm_0 = r.as_matrix()
    print('sci_res : \n', rm_0)
    
   
    # 我的算法
    rm_1 = euler2rotmat_zxy_ex(euler_list_)
    print('my__res : \n', rm_1)

    # gamma beta alpha
    euler_list_ = [30, 20, 10]
    euler_list_ = np.array(euler_list_, dtype=np.float32)
    euler_list_ = euler_list_ * np.pi / 180.0    # 角度到弧度
    
    # 'zxy' 外旋（左乘）== 'YXZ' 内旋（右乘），每个轴转动角度一致

    # scipy算法验证
    # 'YXZ' 为内旋（右乘），先绕y轴gamma,再绕x轴beta,再绕z轴alpha  => rotmat = Ry(gamma) @ Rx(beta) @ Rz(alpha)
    r = R.from_euler('YXZ', euler_list_, degrees=False)
    rm_0 = r.as_matrix()
    print('sci_res : \n', rm_0)
    # 我的算法
    rm_1 = euler2rotmat_YXZ_in(euler_list_)
    print('my__res : \n', rm_1)

    # pyquaternion算法验证
    # 欧拉角到旋转矩阵
    rm_1 = Q(axis=(0.0, 0.0, 1.0), radians=np.pi/2)
    rm_1 = rm_1.rotation_matrix
    print('pyq_res (绕z旋转90度): \n', rm_1)


    # ===========================================================
    # w,x,y,z
    quat_list_wxyz = [0.707, 0.0, 0.0, -0.707]

    # scipy算法验证
    # 四元数到旋转矩阵   函数输入是[x,y,z,w]
    q_ = quat_list_wxyz
    quat_list_xyzw = q_[1],q_[2],q_[3],q_[0]
    r = R.from_quat(quat_list_xyzw)
    rm_0 = r.as_matrix()
    print('sci_res : \n', rm_0)

    # pyquaternion算法验证
    # 四元数到旋转矩阵   函数输入是[w,x,y,z]
    rm_1 = Q(quat_list_wxyz).rotation_matrix
    print('pyq_res : \n', rm_1)

    # 我的算法
    rm_2 = quaternion2rotmat(quat_list_wxyz, mode='Ham')
    print('my__res : \n', rm_2)

    # 我的算法    JPL与Ham互为转置
    rm_2 = quaternion2rotmat(quat_list_xyzw, mode='JPL')
    print('my__res : \n', rm_2)

    rm_0 = np.array([   [-0.09201737, -0.99573094, -0.00725821],
                        [-0.06323759,  0.01311801, -0.9979123 ],
                        [ 0.99374735, -0.09136628, -0.06417471]])

    quat_0 = rotmat2quaternion_sci(rm_0)
    print('sci_res : \n', quat_0)

