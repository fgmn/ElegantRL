import os
import numpy as np
import numpy.random as rd
import pandas as pd
import networkx as nx
import math
import matplotlib.pyplot as plt
import torch
import random
from torch_geometric.data import Data

Array = np.ndarray


class Obs_data:
    def __init__(self, adj_list, node_state, pos_encoding):
        self.adj_list = adj_list
        self.node_state = node_state
        self.pos_encoding = pos_encoding

class GridNetEnv:
    def __init__(self, filename="grid_graph.adjlist"):

        self.load_data_from_disk(filename)

        # environment information
        self.env_name = 'GridNetEnv'
        self.if_discrete = True
        self.state_dim = 40 # cur_node tar_node
        self.action_dim = 4
        self.max_step = self.num_points
        
        # reset()
        self.ori_node = 0
        self.cur_node = None
        self.tar_node = None
        self.node_state = None
        self.cumulative_returns = 0
        self.reset()


    def reset(self, full_reset=False):
        self.tar_node = self.num_points - 1
        if full_reset: self.full_reset()
        self.cur_node = self.ori_node
        self.reward = 0
        self.cumulative_returns = 0

        # 初始化node_state
        # 设置起点状态为[1, 0, 0, 0]，终点状态为[0, 0, 0, 1]
        # 其余节点初始化为[0, 1, 0, 0]，表示从未访问过
        self.node_state = np.zeros((self.num_points, 4))
        
        self.node_state[self.ori_node] = [1, 0, 0, 0]
        self.node_state[self.tar_node] = [0, 0, 0, 1]

        for i in range(self.num_points):
            if i != self.ori_node and i != self.tar_node:
                self.node_state[i] = [0, 1, 0, 0]

        # return self.get_state(), self.get_action_mask()
        obs_edges, obs_nodes, obs_pos = self.get_obs()
        return obs_edges, obs_nodes, obs_pos, self.get_action_mask()
    
    def full_reset(self):
        '''
        重置起点和终点
        '''
        # self.ori_node = random.randint(0, self.num_points-1)
        # self.tar_node = random.randint(0, self.num_points-1)
        pass

    def step(self, action: Array) -> (Array, Array, bool, dict):

        neighbor_c = self.adj_list[self.cur_node]
        # if action >= len(neighbor_c):
        #     next_node = self.cur_node
        #     reward = -500
        # else:
        next_node = neighbor_c[action]
        # 更新node_state
        self.node_state[next_node] = [0, 0, 1, 0]

        """reward"""
        deta = np.linalg.norm(np.array(self.pos[self.cur_node]) - np.array(self.pos[self.tar_node])) - \
            np.linalg.norm(np.array(self.pos[next_node]) - np.array(self.pos[self.tar_node]))
        cache = self.adj_list[next_node]

        # reward2 = (1.2 * (deta / abs(deta)) if deta != 0 else 0)
        reward2 = 0

        # if next_node == self.ori_node:
        #     reward = -2 + reward2
        # elif len(cache) == 1 and next_node != self.tar_node:
        #     reward = -5
        # elif 
        if next_node == self.tar_node:
            reward = 100 + reward2
        else:
            reward = -1 + reward2
        self.cumulative_returns += reward

        """done"""
        done = (next_node == self.tar_node)

        """next_state"""
        self.cur_node = next_node
        # next_state = self.get_state()
        next_obs_edges, next_obs_nodes, next_obs_pos = self.get_obs()
        next_action_mask = self.get_action_mask()

        # return next_state, reward, done, None, next_action_mask
        return next_obs_edges, next_obs_nodes, next_obs_pos, reward, done, None, next_action_mask

    def one_hot_encoding(self, pos, max_val_1=9, max_val_2=9):
        """
        pos: 长度为2的nparray数组
        max_val_1: pos[0]的最大可能值
        max_val_2: pos[1]的最大可能值
        """
        # 创建 pos[0] 的 one-hot 编码，长度为 max_val_1 + 1
        one_hot_1 = np.zeros(max_val_1 + 1)
        one_hot_1[pos[0]] = 1

        # 创建 pos[1] 的 one-hot 编码，长度为 max_val_2 + 1
        one_hot_2 = np.zeros(max_val_2 + 1)
        one_hot_2[pos[1]] = 1

        # 将两个 one-hot 编码拼接在一起
        one_hot_combined = np.concatenate((one_hot_1, one_hot_2))

        return one_hot_combined

    def get_state(self):
        """
        Retrieves the current state representation and action mask for the environment.

        Returns:
        - state (numpy.ndarray): The current state, combining positions of the current and target nodes.
        - action_mask (numpy.ndarray): A boolean array indicating valid actions from the current state.
        """
        # Ensure positions are stored as numpy arrays for consistent operations
        current_position = np.array(self.pos[self.cur_node])
        target_position = np.array(self.pos[self.tar_node])

        # Combine positions into a single state array
        # state = np.hstack((current_position, target_position))
        # 使用 one-hot 向量
        delta = target_position - current_position

        vec1 = self.one_hot_encoding(current_position)
        vec2 = self.one_hot_encoding(delta)
        state = np.hstack((vec1, vec2))
        return state
    

    def get_obs(self):
        '''
        获取观测：邻接表，节点访问状态，原来的状态即位置的one-hot
        已经访问过的节点属性设为1
        '''
        # 使用torch_geometric存储邻接表
        # edges = [(src, dest) for src, nbrs in self.adj_list.items() for dest in nbrs]
        # edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        # obs_graph = Data(x=self.node_state, edge_index=edge_index)

        obs_edges = np.array(self.edges).T
        # print("NumPy array shape:", obs_edges.shape)


        # adj_list = self.adj_list
        
        # 节点访问状态
        obs_nodes = self.node_state    # np.arrary

        # 位置的one-hot编码
        obs_pos = self.get_state() # np.arrary

        # 创建观测对象
        # obs = Obs_data(adj_list, node_state, pos_encoding)
        
        # return obs
        # 目前obs_edges是固定的
        return obs_edges, obs_nodes, obs_pos

    def get_action_mask(self) -> Array:
        array = np.full(self.action_dim, False, dtype=bool)
        array[:len(self.adj_list[self.cur_node])] = True
        return array

    def parse_node(self, node):
        """Converts node coordinates in format '(x, y)' to a single index."""
        coords = node.strip('()').split(',')
        x, y = int(coords[0].strip()), int(coords[1].strip())
        node = x * self.num_cols + y
        if node not in self.pos:
            self.pos[node] = (x, y)
        return node
    
    def load_data_from_disk(self, filename):
        # Read the shapefile
        G = nx.read_adjlist(filename, nodetype=str, delimiter=":")

        self.num_points = G.number_of_nodes()
        self.num_cols = math.ceil(np.sqrt(self.num_points))
        self.num_edges = G.number_of_edges()

        self.adj_list = {i: [] for i in range(self.num_points)} # 邻接表
        self.pos = {}

        for u, v in G.edges():
            u = self.parse_node(u)
            v = self.parse_node(v)
            self.adj_list[u].append(v)
            self.adj_list[v].append(u)

        # self.edges = [(src, dest) for src, nbrs in self.adj_list.items() for dest in nbrs]
        self.edges = set()
        for src, nbrs in self.adj_list.items():
            for dest in nbrs:
                # 保证边的存储顺序，使得第一个节点总是编号较小的
                ordered_edge = tuple(sorted((src, dest)))
                self.edges.add(ordered_edge)
        self.edges = list(self.edges)


    def visualize_best_actions(self, actor):    # to be fixed
        device = next(actor.parameters()).device

        G = nx.Graph()
        for u in range(self.num_points):
            for v in self.adj_list[u]:
                G.add_edge(u, v)

        # 绘制整个图
        nx.draw(G, self.pos, node_color='lightblue', with_labels=False, edge_color='gray', node_size=30)    # node_size=300

        # 为每个状态绘制最佳动作的箭头
        for node in range(self.num_points):
            # best_action_index = np.argmax(actions[:len(adj_list[state])])
            # best_neighbor = adj_list[state][best_action_index] if adj_list[state] else state

            # state = np.hstack((self.pos[node], self.pos[self.tar_node]))
            target_position = np.array(self.pos[self.tar_node])
            current_position = np.array(self.pos[node])
            delta = target_position - current_position
            vec1 = self.one_hot_encoding(current_position)
            vec2 = self.one_hot_encoding(delta)
            state = np.hstack((vec1, vec2))
            tensor_state = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            tensor_action = actor(tensor_state)
            print(f"Node {node}: Actions -> {tensor_action.cpu().numpy().flatten()}")

            if tensor_action.dim() == 1:
                tensor_action = tensor_action.unsqueeze(0)
            # tensor_action = tensor_action.argmax(dim=1)
            x = len(self.adj_list[node])
            if tensor_action.dim() == 1:
                # 如果tensor_action是一维的
                tensor_action = tensor_action[:x].argmax()
            elif tensor_action.dim() > 1:
                # 如果tensor_action是多维的，例如在批处理或多代理情况下
                tensor_action = tensor_action[:, :x].argmax(dim=1)
            action = tensor_action.detach().cpu().numpy()[0]  # not need detach(), because using torch.no_grad() outside
            best_neighbor = self.adj_list[node][action]
            print(self.adj_list[node])
            # Print the action decision
            # print(f"Node {node}: Best Action -> Move to Node {best_neighbor}")

            # 绘制箭头从当前状态指向最佳邻居
            if best_neighbor != node:  # 防止自环
                nx.draw_networkx_edges(
                    G, self.pos, 
                    edgelist=[(node, best_neighbor)], 
                    width=0.2, 
                    edge_color='red', 
                    style='solid',  # 使用实线
                    arrows=True, 
                    arrowstyle='-|>',  # 箭头样式
                    arrowsize=2      # 箭头大小
                )

        # 显示图
        plt.title('Best Actions Visualization')
        plt.savefig('Best Actions Visualization')
        # plt.show()