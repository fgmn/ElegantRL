import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import summary
import os
from datetime import datetime
from utils import plot_route

# 获取当前时间戳并格式化为字符串
current_time = datetime.now().strftime("%Y%m%d-%H%M%S")

# 创建带有时间戳的日志目录
log_dir = f"logs/qlearning/{current_time}"
os.makedirs(log_dir, exist_ok=True)

# 创建一个 summary writer
writer = summary.create_file_writer(log_dir)

# Read the shapefile
gdf = gpd.read_file('dqn_path_planing\snnu.shp')

# Display the first few rows of the GeoDataFrame
# print(gdf.head())

# 提取所有坐标信息
all_coordinates = []

for geom in gdf.geometry:
    if geom.geom_type == 'Polygon':
        # 如果几何类型是多边形
        for x, y in np.array(geom.exterior.coords):
            all_coordinates.append((x, y))
    elif geom.geom_type == 'MultiPolygon':
        # 如果几何类型是多重多边形
        for polygon in geom.geoms:
            for x, y in np.array(polygon.exterior.coords):
                all_coordinates.append((x, y))
    elif geom.geom_type == 'LineString':
        # 如果几何类型是线串
        for x, y in np.array(geom.coords):
            all_coordinates.append((x, y))
    elif geom.geom_type == 'Point':
        # 如果几何类型是点
        x, y = geom.x, geom.y
        all_coordinates.append((x, y))

# 将坐标转换为 NumPy 数组
coordinate_points = np.array(all_coordinates)
print(len(coordinate_points))
# 删除 NaN 值并去重
important_points = np.unique(coordinate_points[~np.isnan(coordinate_points).any(axis=1)], axis=0)

# 打印重要点
print(len(important_points))

# 提取 x 和 y 坐标
x = coordinate_points[:, 0]  # 获取所有 x 坐标
y = coordinate_points[:, 1]  # 获取所有 y 坐标
# 计算 x 和 y 的最小值和最大值
x_min, x_max = np.min(x), np.max(x)
y_min, y_max = np.min(y), np.max(y)

# 打印 x 和 y 的范围
print(f"x 的范围: 最小值 = {x_min}, 最大值 = {x_max}")
print(f"y 的范围: 最小值 = {y_min}, 最大值 = {y_max}")

# 定义任务的起始状态和终止状态
task = {
    'initialState': np.array([108.889147, 34.159450]),
    'terminalState': np.array([108.942065, 34.204787])
}

# 初始化 inflection_point 和 inflection_point_index
num_points = important_points.shape[0]
inflection_point = {i: [] for i in range(num_points)}
inflection_point_index = {i: [] for i in range(num_points)}
# 初始化一维数组用于存储距离
distance_from_start = np.zeros(num_points)
distance_to_end = np.zeros(num_points)

# 计算每个重要点与起始状态和终止状态的距离
for i in range(num_points):
    distance_from_start[i] = np.linalg.norm(important_points[i] - task['initialState'])
    distance_to_end[i] = np.linalg.norm(important_points[i] - task['terminalState'])

# # 定义一个精度误差范围
# tolerance = 1e-6


# 提取连接关系
for geom in gdf.geometry:
    if geom.geom_type == 'LineString':
        # 对于每个 LineString，提取起点和终点
        coords = list(geom.coords)
        for start, end in zip(coords[:-1], coords[1:]):
            n1 = np.where((important_points[:, 0] == start[0]) & (important_points[:, 1] == start[1]))[0]
            n2 = np.where((important_points[:, 0] == end[0]) & (important_points[:, 1] == end[1]))[0]
            if len(n1) > 0 and len(n2) > 0:
                n1 = n1[0]
                n2 = n2[0]

                # 更新 inflection_point 和 inflection_point_index
                inflection_point[n1].append((x[i + 1], y[i + 1]))
                inflection_point_index[n1].append(n2)

                inflection_point[n2].append((x[i], y[i]))
                inflection_point_index[n2].append(n1)

# # 打印结果
# print("Inflection Points:")
# for key, value in inflection_point.items():
#     print(f"{key}: {value}")

# print("\nInflection Point Index:")
# for key, value in inflection_point_index.items():
#     print(f"{key}: {value}")

# 找到 distance_from_start 的最小值及其索引
start_path = np.min(distance_from_start)
inst = np.where(distance_from_start == start_path)[0][0]  # 索引

# 找到 distance_to_end 的最小值及其索引
end_path = np.min(distance_to_end)
tast = np.where(distance_to_end == end_path)[0][0]        # 索引
inst = 35
tast = 9000

# 打印结果
print(f"Start Path Minimum Value: {start_path}, Index: {inst}")
print(f"End Path Minimum Value: {end_path}, Index: {tast}")

# Determine the maximum number of neighbors any node has
max_neighbors = max(len(neighbors) for neighbors in inflection_point_index.values())

# Initialize Q-table, walking path, and distance weight
Q = np.zeros((num_points, max_neighbors))
distance_weight = np.zeros((num_points, max_neighbors))
# walking_path = np.zeros((num_points, max_neighbors))

distance = np.full(num_points, 1e6)
distance[inst] = 0  # Convergence check
distance_old = distance.copy()  
parent = np.full(num_points, -1, dtype=int)
count = 0
# delta_dis = []
# steps = []
# le_experience_route = []

# Print the size of the Q-table
q_table_size = Q.shape
print(f"Size of Q-table: {q_table_size}")

alpha = 0.3
gamma = 0.9
max_steps = 10000
episodes = 2000

for episode in tqdm(range(1, episodes + 1), desc="Episodes", leave=True):
    length_of_experience_route = 0
    # print(f"Episode: {episode}")

    # 当前状态设为初始状态
    current_state = inst

    tttt = [inst]  # 记录初始状态的索引
    episode_reward = 0
    # while True: # 采样一条从起点到终点的轨迹
    for step in tqdm(range(max_steps), desc=f"Episode {episode}", leave=False):

        neighbor_c = inflection_point_index[current_state]
        # print(neighbor_c)

        action_index = 0
        ran = np.random.rand()
        parameter = max(0.3 - 0.00001 * episode, 0.01)
        if ran <= parameter:
            rand_choose = np.random.permutation(neighbor_c)  # Randomly permute neighbors
            n_state = rand_choose[0]
        else:
            # Select the action (neighbor) with the maximum Q-value
            q_values = Q[current_state, :len(neighbor_c)]
            action_index = np.argmax(q_values)  # Find the index of the maximum Q-value
            n_state = neighbor_c[action_index]  # Determine the next state based on the action

        # Calculate the distance
        dis = np.linalg.norm(important_points[current_state] - important_points[n_state])
        distance_weight[current_state, action_index] = dis
        distance_weight[n_state, np.where(inflection_point_index[n_state] == current_state)] = dis

        if distance[n_state] >= distance[current_state] + dis:
            parent[n_state] = current_state
            distance[n_state] = distance[current_state] + dis

        deta = np.linalg.norm(important_points[current_state] - important_points[tast]) - \
            np.linalg.norm(important_points[n_state] - important_points[tast])
        cache = inflection_point_index[n_state]

        reward2 = (0.8 * (deta / abs(deta)) if deta != 0 else 0)

        if n_state == inst:
            reward = -100 + reward2
        elif len(cache) == 1 and n_state != tast:
            reward = -500
        elif n_state == tast:
            reward = 5000 + reward2
        else:
            reward = -1 + reward2
        episode_reward += reward

        # Calculate the maximum Q-value for the next state
        q_max_next_state = np.max(Q[n_state])

        # Update the Q-value for the current state and chosen action
        Q[current_state, action_index] += alpha * (reward + gamma * q_max_next_state - Q[current_state, action_index])

        # lengthof_experience_route = sum(np.linalg.norm(np.diff(important_points[tttt], axis=0), axis=1))
        tttt.append(n_state)

        if n_state == tast:
            break

        current_state = n_state
    if episode % 10 == 0:
        plot_route(tttt, important_points, gdf, 'ep' + str(episode))
    # Calculate the difference between the current and old distance
    d_distance = np.sum(np.abs(distance - distance_old))

    # Convergence check
    if d_distance < 0.001:
        if count > 25:  # Check for convergence after multiple small changes
            print(f"Convergence reached in episode: {episode}")
            break
        else:
            count += 1
    else:
        distance_old = distance.copy()
        count = 0

    # 使用 TensorBoard 记录每个指标
    with writer.as_default():
        summary.scalar("Delta Distance", d_distance, step=episode)
        summary.scalar("Steps", len(np.unique(tttt)), step=episode)
        # summary.scalar("Experience Route Length", lengthof_experience_route, step=episode)
        summary.scalar("Total Reward", episode_reward, step=episode)
    # delta_dis.append(d_distance)
    # steps.append(len(np.unique(tttt)))
    # le_experience_route.append(start_path + lengthof_experience_route + end_path)

# Initialize
segment_distance = []

# Determine route feasibility
if distance[tast] == 1e6:
    route = []
else:
    route = [tast]  # Initialize route with the target node

    # Backtrack to build the route from `tast` to `inst`
    while route[0] != inst:
        # Find the parent of the current node in the route
        parent_node = parent[route[0]]
        route.insert(0, parent_node)
        # print(type(route[0]))
        # Calculate segment distance
        action_index = np.where(inflection_point_index[route[1]] == route[0])[0]
        segment_distance.append(distance_weight[route[1], action_index])


# Reshape route for plotting
route = np.array(route).reshape(-1, 1)

# Calculate length of the planned route
lengthof_plan_route = sum(segment_distance)

# 绘制GeoDataFrame
ax = gdf.plot(color='lightgrey', figsize=(10, 8), legend=True, label='GeoData')
# gdf.plot()
# plt.show()

# Plot all segments and the main route
# plt.figure(figsize=(10, 6))

# # Plot walking paths
# for i in range(walking_path.shape[0]):
#     for j in range(1, np.sum(walking_path[i] != 0)):
#         px = [important_points[walking_path[i, 0], 0], important_points[walking_path[i, j], 0]]
#         py = [important_points[walking_path[i, 0], 1], important_points[walking_path[i, j], 1]]
#         plt.plot(px, py, 'b-')

# Plot the main route
route_x = important_points[route.flatten(), 0]
route_y = important_points[route.flatten(), 1]
ax.plot(route_x, route_y, 'm.-', linewidth=2)

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Planned Route and GeoData Segments')
plt.grid(True)
plt.show()

# # 创建一个新的图形窗口
# plt.figure(figsize=(12, 8))

# # 绘制 delta_dis 的变化图
# plt.subplot(3, 1, 1)
# plt.plot(delta_dis, marker='o', linestyle='-', color='b', label='Delta Dis')
# plt.title('Convergence and Route Characteristics Over Episodes')
# plt.ylabel('Delta Dis')
# plt.grid(True)
# plt.legend()

# # 绘制 steps 的变化图
# plt.subplot(3, 1, 2)
# plt.plot(steps, marker='o', linestyle='-', color='g', label='Steps')
# plt.ylabel('Steps')
# plt.grid(True)
# plt.legend()

# # 绘制 le_experience_route 的变化图
# plt.subplot(3, 1, 3)
# plt.plot(le_experience_route, marker='o', linestyle='-', color='r', label='Experience Route Length')
# plt.xlabel('Episode')
# plt.ylabel('Experience Route Length')
# plt.grid(True)
# plt.legend()

# # 显示图形
# plt.tight_layout()
# plt.show()