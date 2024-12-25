import math
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import make_interp_spline
import json
import matplotlib.pyplot as plt

def calculate_transformation_matrix(cam_points, base_points):
    """
    计算从相机坐标系到基座坐标系的转移矩阵，考虑尺度差异。
    
    Args:
        cam_points: List of [x, y, z, rx, ry, rz] in camera frame (单位：像素).
        base_points: List of [x, y, z, rx, ry, rz] in base frame (单位：米).
        
    Returns:
        T_cam2base: 4x4 transformation matrix.
        scale: Scale factor s.
    """
    # 提取位置 (x, y, z)
    cam_positions = np.array([p[:3] for p in cam_points])
    base_positions = np.array([p[:3] for p in base_points])

    # Step 1: 计算比例因子 s
    cam_distances = np.linalg.norm(cam_positions[:, None, :] - cam_positions[None, :, :], axis=2)
    base_distances = np.linalg.norm(base_positions[:, None, :] - base_positions[None, :, :], axis=2)

    # 计算点对之间的平均距离，避免零距离
    mean_cam_distance = np.mean(cam_distances[cam_distances > 0])
    mean_base_distance = np.mean(base_distances[base_distances > 0])
    scale = mean_base_distance / mean_cam_distance

    # 缩放相机点到与基座点相同的单位
    cam_positions_scaled = cam_positions * scale

    # Step 2: 计算旋转和平移
    # Compute centroids
    centroid_cam = np.mean(cam_positions_scaled, axis=0)
    centroid_base = np.mean(base_positions, axis=0)

    # Center the points
    cam_centered = cam_positions_scaled - centroid_cam
    base_centered = base_positions - centroid_base

    # Step 3: 计算旋转部分
    H = cam_centered.T @ base_centered
    U, _, Vt = np.linalg.svd(H)
    R_AB = Vt.T @ U.T
    if np.linalg.det(R_AB) < 0:
        Vt[-1, :] *= -1
        R_AB = Vt.T @ U.T

    # Ensure R is a proper rotation matrix
    # if np.linalg.det(R) < 0:
    #     Vt[-1, :] *= -1
    #     R = Vt.T @ U.T

    # Compute translation
    t = centroid_base - R_AB @ centroid_cam

    # Construct the 4x4 transformation matrix
    T_cam2base = np.eye(4)
    T_cam2base[:3, :3] = R_AB
    T_cam2base[:3, 3] = t

    return T_cam2base, scale

def create_splines(cam_points, base_points, plot=False):
    """
    创建三条样条拟合曲线。

    Args:
        A (list or np.ndarray): 数据的 X 轴值。
        X (list or np.ndarray): 数据的 Y 轴值（曲线 1）。
        Y (list or np.ndarray): 数据的 Y 轴值（曲线 2）。
        Z (list or np.ndarray): 数据的 Y 轴值（曲线 3）。

    Returns:
        splines (tuple): 包含三个样条函数 (spline_X, spline_Y, spline_Z)。
    """
    # 检查输入数据长度是否一致
    X = [pose[3] for pose in base_points]
    Y = [pose[4] for pose in base_points]
    Z = [pose[5] for pose in base_points]
    A = [pose[5] for pose in cam_points]



    if not (len(A) == len(X) == len(Y) == len(Z)):
        raise ValueError("A, X, Y, Z 的长度必须相同！")

    # 创建样条拟合函数
    spline_X = make_interp_spline(A, X)
    spline_Y = make_interp_spline(A, Y)
    spline_Z = make_interp_spline(A, Z)

    if plot:
        plot_custom_axes(A, X, Y, Z)

    return [spline_X, spline_Y, spline_Z]

def plot_custom_axes(A, X, Y, Z):
    """
    绘制以 A 为 X 轴，X, Y, Z 为 Y 轴的数据曲线，并使用样条拟合。

    Args:
        A (list or np.ndarray): 数据的 X 轴值。
        X (list or np.ndarray): 数据的 Y 轴值（曲线 1）。
        Y (list or np.ndarray): 数据的 Y 轴值（曲线 2）。
        Z (list or np.ndarray): 数据的 Y 轴值（曲线 3）。
    """
    # 检查输入数据长度是否一致
    if not (len(A) == len(X) == len(Y) == len(Z)):
        raise ValueError("A, X, Y, Z 的长度必须相同！")

    # 创建样条拟合
    A_fine = np.linspace(min(A), max(A), 500)  # 更密集的 X 轴
    spline_X = make_interp_spline(A, X)(A_fine)
    spline_Y = make_interp_spline(A, Y)(A_fine)
    spline_Z = make_interp_spline(A, Z)(A_fine)

    # 创建图表
    plt.figure(figsize=(10, 6))

    # 绘制原始曲线
    plt.plot(A, X, label="X-Axis Data", marker='o', linestyle='--', alpha=0.7)
    plt.plot(A, Y, label="Y-Axis Data", marker='s', linestyle='--', alpha=0.7)
    plt.plot(A, Z, label="Z-Axis Data", marker='^', linestyle='--', alpha=0.7)

    # 绘制样条拟合曲线
    plt.plot(A_fine, spline_X, label="X-Axis Spline", linewidth=2)
    plt.plot(A_fine, spline_Y, label="Y-Axis Spline", linewidth=2)
    plt.plot(A_fine, spline_Z, label="Z-Axis Spline", linewidth=2)

    # 添加标题和标签
    plt.title("Custom Axes Plot with Spline Fit", fontsize=16)
    plt.xlabel("A (Custom X-Axis)", fontsize=14)
    plt.ylabel("Values (Y-Axis)", fontsize=14)

    # 显示图例
    plt.legend(fontsize=12)

    # 显示网格
    plt.grid(True, linestyle='--', alpha=0.7)

    # 显示图表
    plt.show()

def get_rxryrz(splines, x):
    """
    使用样条函数计算给定 x 值处的结果。

    Args:
        splines (tuple): 包含三个样条函数 (spline_X, spline_Y, spline_Z)。
        x (float or np.ndarray): 输入的 X 值。

    Returns:
        values (tuple): 包含三个曲线的计算值 (y_X, y_Y, y_Z)。
    """
    spline_X, spline_Y, spline_Z = splines
    y_X = spline_X(x)
    y_Y = spline_Y(x)
    y_Z = spline_Z(x)
    return y_X, y_Y, y_Z

def transform_cam_to_base(T_cam2base, scale, splines, cam_point):
    """
    将单个相机系点转换到基座系，包括位置和旋转。

    Args:
        T_cam2base: 4x4 transformation matrix.
        scale: Scale factor (pixels to meters).
        cam_point: [x, y, z, rx, ry, rz] in camera frame, where rotation is in axis-angle (Rodrigues) form.

    Returns:
        base_point: [x, y, z, rx, ry, rz] in base frame.
    """
    # 提取位置并转换
    cam_position = np.array(cam_point[:3])  # 提取位置部分
    cam_position_scaled = cam_position * scale  # 应用比例因子
    cam_position_homogeneous = np.append(cam_position_scaled, 1)  # 转为齐次坐标
    base_position_homogeneous = T_cam2base @ cam_position_homogeneous  # 转换到基座系
    base_position = base_position_homogeneous[:3]  # 提取非齐次坐标

    # 提取旋转并转换
    # cam_rotation = R.from_rotvec(cam_point[3:])  # 相机系的旋转 (轴角形式)
    # T_cam_rotation = T_cam2base[:3, :3]  # 仅提取旋转矩阵部分
    # base_rotation_matrix = T_cam_rotation @ cam_rotation.as_matrix()  # 变换旋转矩阵
    # base_rotation = R.from_matrix(base_rotation_matrix).as_rotvec()  # 转换回轴角形式

    y_X, y_Y, y_Z = get_rxryrz(splines, cam_point[-1])
    base_rotation = [float(y_X), float(y_Y), float(y_Z)]

    return [*base_position, *base_rotation]

def calculate_transformation(cam_points_data, base_points_data):
    T_cam2base, scale = calculate_transformation_matrix(cam_points_data, base_points_data)
    splines = create_splines(cam_points_data, base_points_data, True)
    return T_cam2base, scale, splines

def transfer_campoints_to_base(arm="right"):
    file_name = f"recorded_positions_{arm}.json"
    with open(file_name, "r") as file:
        cam_points = json.load(file)

    with open(f"robot_positions_{arm}.json", "r") as file:
        base_points = json.load(file)

    with open("curve_points.json", "r") as file:
        cam_points_test = json.load(file)

    # cam_points.reverse()
    # base_points.reverse()

    print("Cam points:", cam_points)
    print("Base points:", base_points)

    T_cam2base, scale, splines = calculate_transformation(cam_points, base_points)

    base_points_transformed = []
    for point in cam_points_test:
        base_points_transformed.append(transform_cam_to_base(T_cam2base, scale, splines, point))
    
    with open("base_points_transformed.json", "w") as file:
        json.dump(base_points_transformed, file, indent=4)

    print(f'Saved to base_points_transformed.json')
        
# Uncommand this line to test.
# transfer_campoints_to_base()
