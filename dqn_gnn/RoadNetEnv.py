import os
import numpy as np
import numpy.random as rd
import pandas as pd
import geopandas as gpd

Array = np.ndarray

class RoadNetEnv:
    def __init__(self, gdffile='dqn_path_planing\snnu.shp'):
        self.num_points = None
        self.inflection_point = None
        self.inflection_point_index = None
        self.max_neighbors = None
        self.load_data_from_disk(gdffile)

        # reset()
        self.ori_node = 2036
        self.cur_node = None
        self.tar_node = None
        self.cumulative_returns = 0
        self.reset()

        # environment information
        self.env_name = 'RoadNetEnv'
        self.if_discrete = True
        self.state_dim = 4 # cur_node tar_node
        self.action_dim = self.max_neighbors
        self.max_step = 12345

    def reset(self):
        self.tar_node = np.random.randint(0, self.max_neighbors)
        self.cur_node = self.ori_node
        self.reward = 0
        self.cumulative_returns = 0
        return self.get_state()

    def step(self, action: Array) -> (Array, Array, bool, dict):

        neighbor_c = self.inflection_point_index[self.cur_node]
        if action >= len(neighbor_c):
            next_node = self.cur_node
        else:
            next_node = neighbor_c[action]

        """reward"""
        deta = np.linalg.norm(self.points[self.cur_node] - self.points[self.tar_node]) - \
            np.linalg.norm(self.points[next_node] - self.points[self.tar_node])
        cache = self.inflection_point_index[next_node]

        reward2 = (8 * (deta / abs(deta)) if deta != 0 else 0)

        if next_node == self.ori_node:
            reward = -100 + reward2
        elif len(cache) == 1 and next_node != self.tar_node:
            reward = -500
        elif next_node == self.tar_node:
            reward = 5000 + reward2
        else:
            reward = -1 + reward2
        self.cumulative_returns += reward

        """done"""
        done = (next_node == self.tar_node)

        """next_state"""
        self.cur_node = next_node
        next_state = self.get_state()

        return next_state, reward, done, None

    def get_state(self) -> Array:
        return np.hstack((self.points[self.cur_node], self.points[self.tar_node]))
    
    def load_data_from_disk(self, gdffile):
        # Read the shapefile
        gdf = gpd.read_file(gdffile)

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
        # 删除 NaN 值并去重
        self.points = np.unique(coordinate_points[~np.isnan(coordinate_points).any(axis=1)], axis=0)

        # 提取 x 和 y 坐标
        x = coordinate_points[:, 0]  # 获取所有 x 坐标
        y = coordinate_points[:, 1]  # 获取所有 y 坐标

        # 初始化 inflection_point 和 inflection_point_index
        self.num_points = self.points.shape[0]
        self.inflection_point = {i: [] for i in range(self.num_points)}
        self.inflection_point_index = {i: [] for i in range(self.num_points)}

        # 提取连接关系
        for geom in gdf.geometry:
            if geom.geom_type == 'LineString':
                # 对于每个 LineString，提取起点和终点
                coords = list(geom.coords)
                for start, end in zip(coords[:-1], coords[1:]):
                    n1 = np.where((self.points[:, 0] == start[0]) & (self.points[:, 1] == start[1]))[0]
                    n2 = np.where((self.points[:, 0] == end[0]) & (self.points[:, 1] == end[1]))[0]
                    if len(n1) > 0 and len(n2) > 0:
                        n1 = n1[0]
                        n2 = n2[0]

                        # 更新 inflection_point 和 inflection_point_index
                        self.inflection_point[n1].append((end[0], end[1]))
                        self.inflection_point_index[n1].append(n2)

                        self.inflection_point[n2].append((start[0], start[1]))
                        self.inflection_point_index[n2].append(n1)

        # Determine the maximum number of neighbors any node has
        self.max_neighbors = max(len(neighbors) for neighbors in self.inflection_point_index.values())
