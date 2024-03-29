import numpy as np
import matplotlib.pyplot as plt

# 从17中热图提取2D姿态点
def extract_keypoints_from_heatmaps(heatmaps, threshold=0.1):
    """
    从热力图中提取关节点坐标。
    :param heatmaps: 热力图数组，形状为 (batch_size, num_joints, height, width)
    :param threshold: 阈值，用于过滤低置信度的关节点
    :return: 2D关节点坐标列表，形状为 (num_joints, 2)
    """
    all_keypoints = []
    # 假设heatmaps的形状为 (1, num_joints, height, width)
    # 我们只关心第一个元素（第一个图像）的热力图
    heatmaps = heatmaps[0]
    for heatmap in heatmaps:
        y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        confidence = heatmap[y, x]
        if confidence > threshold:
            all_keypoints.append((x, y))
        else:
            all_keypoints.append((None, None))  # 低置信度的关节点
    return all_keypoints



# 假设的骨骼长度
bone_lengths = {
    # 躯干
    (5, 11): 18,  # 左肩到左髋
    (6, 12): 18,  # 右肩到右髋
    (11, 12): 15,  # 左髋到右髋

    # 上肢
    (5, 7): 15,  # 左肩到左肘
    (7, 9): 15,  # 左肘到左手腕
    (6, 8): 15,  # 右肩到右肘
    (8, 10): 15,  # 右肘到右手腕

    # 下肢
    (11, 13): 20,  # 左髋到左膝
    (13, 15): 20,  # 左膝到左脚踝
    (12, 14): 20,  # 右髋到右膝
    (14, 16): 20,  # 右膝到右脚踝
}


# 估算3D姿态点
def estimate_3d_pose(keypoints_2d, bone_lengths, reference_point_id=0, depth_offset=0.5):
    """
    从2D关节点坐标估计3D姿态。
    :param keypoints_2d: 2D关节点坐标，形状为 (num_joints, 2)
    :param bone_lengths: 骨骼长度字典，键为关节点对，值为长度
    :param reference_point_id: 参考点的索引
    :param depth_offset: 深度偏移量
    :return: 3D关节点坐标，形状为 (num_joints, 3)
    """
    num_joints = len(keypoints_2d)
    keypoints_3d = np.full((num_joints, 3), np.nan)  # 使用NaN初始化3D坐标

    # 假设参考点的Z坐标为0
    keypoints_3d[reference_point_id, :2] = keypoints_2d[reference_point_id]
    keypoints_3d[reference_point_id, 2] = 0

    for i in range(num_joints):
        if i != reference_point_id and keypoints_2d[i] != (None, None):
            distance_2d = np.linalg.norm(np.array(keypoints_2d[i]) - np.array(keypoints_2d[reference_point_id]))
            bone_length = bone_lengths.get((reference_point_id, i), 0)
            depth = np.sqrt(max(bone_length**2 - distance_2d**2, 0)) + depth_offset
            keypoints_3d[i, :2] = keypoints_2d[i]
            keypoints_3d[i, 2] = depth

    return keypoints_3d


def draw_3d_pose(keypoints_3d):
    # 将 keypoints_3d 转换为 NumPy 数组
    keypoints_3d = np.array(keypoints_3d)
    # 交换Y和Z
    keypoints_3d[:, [1, 2]] = keypoints_3d[:, [2, 1]]
    # 反转Z轴的值
    keypoints_3d[:, 2] = -keypoints_3d[:, 2]

    # 定义骨骼连接
    # skeleton = [(0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6), (6, 7), (0, 8), (8, 9), (9, 10),
    #             (10, 11), (8, 12), (12, 13), (13, 14), (0, 15), (15, 16)]
    skeleton = [
        (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
        (5, 11), (6, 12), (5, 6), (5, 7), (6, 8),
        (7, 9), (8, 10), (1, 2), (0, 1), (0, 2),
        (1, 3), (2, 4), (3, 5), (4, 6)
    ]

    # 创建3D图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 绘制关键点
    ax.scatter(keypoints_3d[:, 0], keypoints_3d[:, 1], keypoints_3d[:, 2])

    # 绘制骨骼连接
    for bone in skeleton:
        # 只有当两个关键点都不是NaN时才绘制
        if not np.any(np.isnan(keypoints_3d[bone[0]])) and not np.any(np.isnan(keypoints_3d[bone[1]])):
            x_coords = [keypoints_3d[bone[0], 0], keypoints_3d[bone[1], 0]]
            y_coords = [keypoints_3d[bone[0], 1], keypoints_3d[bone[1], 1]]
            z_coords = [keypoints_3d[bone[0], 2], keypoints_3d[bone[1], 2]]
            ax.plot(x_coords, y_coords, z_coords, 'gray')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()

def draw_3d(outputs):
    # 2D转3D
    # keypoints_2d形状为 (17, height, width)
    keypoints_2d = extract_keypoints_from_heatmaps(outputs)
    keypoints_3d = estimate_3d_pose(keypoints_2d, bone_lengths)
    draw_3d_pose(keypoints_3d)
    return keypoints_3d
