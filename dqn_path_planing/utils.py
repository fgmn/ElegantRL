import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm

def plot_route(tttt, important_points, gdf, filename):
    """
    绘制tttt记录的状态路径并保存到文件。

    参数:
    - tttt: 记录状态索引的列表
    - important_points: 状态对应的坐标数组
    - gdf: GeoDataFrame 对象，用于绘制地理数据
    - route: 预定路线的索引数组
    - filename: 保存绘图的文件名
    """

    # 确保保存路径存在
    save_dir = 'route'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 绘制GeoDataFrame
    ax = gdf.plot(color='lightgrey', figsize=(10, 8), legend=True, label='GeoData')
    
    # 绘制实际路径
    tttt_x = important_points[np.array(tttt).flatten(), 0]
    tttt_y = important_points[np.array(tttt).flatten(), 1]
    ax.plot(tttt_x, tttt_y, 'b.-', linewidth=2)

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Visualize Route')
    plt.legend()
    plt.grid(True)

    # 保存绘图
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path)
    plt.close()