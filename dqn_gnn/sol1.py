import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import math
from tensorflow import summary
from datetime import datetime
import os
import seaborn as sns

# 获取当前时间戳并格式化为字符串
current_time = datetime.now().strftime("%Y%m%d-%H%M%S")

# 创建带有时间戳的日志目录
log_dir = f"logs/qlearning/{current_time}"
os.makedirs(log_dir, exist_ok=True)

# 创建一个 summary writer
writer = summary.create_file_writer(log_dir)


def read_graph_from_file(filename):
    # Read the graph using a colon as the delimiter between nodes and their adjacency lists
    G = nx.read_adjlist(filename, nodetype=str, delimiter=":")
    return G

filename = "grid_graph.adjlist"
G = read_graph_from_file(filename)

num_points = G.number_of_nodes()
num_cols = math.ceil(np.sqrt(num_points))

adj_list = {i: [] for i in range(num_points)}

def parse_node(node, num_cols=num_cols):
    """Converts node coordinates in format '(x, y)' to a single index."""
    coords = node.strip('()').split(',')
    x, y = int(coords[0].strip()), int(coords[1].strip())
    return x * num_cols + y

def get_position(index, num_cols=num_cols, return_nparray: bool=True):
    """Converts a single index back to node coordinates in format '(x, y)'."""
    x = index // num_cols
    y = index % num_cols
    if return_nparray:
        return np.array([x, y])
    else:
        return (x, y)

for u, v in G.edges():
    u = parse_node(u)
    v = parse_node(v)
    adj_list[u].append(v)
    adj_list[v].append(u)

inst = 0
tast = num_points - 1

########################## Q-LEARNING TRAIN ###########################
# Initialize Q-table
max_neighbors = 4
Q = np.zeros((num_points, max_neighbors))

alpha = 0.3
gamma = 0.9
max_steps = int(num_points)
episodes = 2000

for episode in tqdm(range(1, episodes + 1), desc="Episodes", leave=True):
    # 当前状态设为初始状态
    current_state = inst

    episode_reward = 0
    # 采样一条从起点到终点的轨迹
    for step in tqdm(range(max_steps), desc=f"Episode {episode}", leave=False):

        neighbor_c = adj_list[current_state]
        # print(neighbor_c)

        ran = np.random.rand()
        parameter = max(0.5 - 0.00001 * episode, 0.01)

        if ran <= parameter:
            action_index = np.random.randint(0, len(neighbor_c))
            n_state = neighbor_c[action_index]
        else:
            # Select the action (neighbor) with the maximum Q-value
            q_values = Q[current_state, :len(neighbor_c)]
            action_index = np.argmax(q_values)  # Find the index of the maximum Q-value
            n_state = neighbor_c[action_index]  # Determine the next state based on the action

        deta = np.linalg.norm(get_position(current_state) - get_position(tast)) - \
            np.linalg.norm(get_position(n_state) - get_position(tast))
        cache = adj_list[n_state]

        reward2 = (10 * (deta / abs(deta)) if deta != 0 else 0)

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
        q_max_next_state = np.max(Q[n_state, :len(cache)])

        # Update the Q-value for the current state and chosen action
        Q[current_state, action_index] += alpha * (reward + gamma * q_max_next_state - Q[current_state, action_index])

        if n_state == tast:
            break

        current_state = n_state

    # Convergence check todo

    # 使用 TensorBoard 记录每个指标
    with writer.as_default():
        summary.scalar("Total Reward", episode_reward, step=episode)


def visualize_best_actions(G, pos, Q, adj_list):
    """
    Visualizes the graph with the best action from each state highlighted.
    :param G: The graph
    :param pos: Dictionary of positions for each node
    :param Q: Q-table with state-action values
    :param adj_list: Adjacency list of the graph
    """
    # 绘制整个图
    nx.draw(G, pos, node_color='lightblue', with_labels=True, edge_color='gray', node_size=300)

    # 为每个状态绘制最佳动作的箭头
    for state, actions in enumerate(Q):
        if state in adj_list:
            best_action_index = np.argmax(actions[:len(adj_list[state])])
            best_neighbor = adj_list[state][best_action_index] if adj_list[state] else state
            
            # 绘制箭头从当前状态指向最佳邻居
            if best_neighbor != state:  # 防止自环
                nx.draw_networkx_edges(
                    G, pos, 
                    edgelist=[(state, best_neighbor)], 
                    width=2, 
                    edge_color='red', 
                    style='solid',  # 使用实线
                    arrows=True, 
                    arrowstyle='-|>',  # 箭头样式
                    arrowsize=20      # 箭头大小
                )

    # 显示图
    plt.title('Best Actions Visualization')
    plt.show()

# 创建图的位置信息
pos = {node: get_position(node, return_nparray=False) for node in range(num_points)}

# 重新构造图G，因为我们使用的索引方式可能已改变
G = nx.Graph()
for u in range(num_points):
    for v in adj_list[u]:
        G.add_edge(u, v)

# visualize_best_actions(G, pos, Q, adj_list)

def visualize_shortest_path(G, pos, Q, adj_list, start_node, end_node, save_dir='result'):
    """
    Visualizes and saves the shortest path from start to end on the graph G with given positions based on Q-values
    and also the path obtained using NetworkX's shortest path algorithm.
    :param G: The graph
    :param pos: Dictionary of positions for each node
    :param Q: Q-table with state-action values
    :param adj_list: Adjacency list of the graph
    :param start_node: Starting node of the path
    :param end_node: Ending node of the path
    :param save_dir: Directory to save the resulting visualization
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)  # Ensure the directory exists

    # Draw the entire graph
    plt.figure(figsize=(10, 8))
    nx.draw(G, pos, node_color='lightblue', with_labels=False, edge_color='gray', node_size=300)
    
    # Find and draw the shortest path based on Q-values
    current_node = start_node
    path = [current_node]
    while current_node != end_node:
        if current_node in adj_list:
            best_action_index = np.argmax(Q[current_node][:len(adj_list[current_node])])
            current_node = adj_list[current_node][best_action_index]
            path.append(current_node)
        else:
            break  # Break if there is no path forward

    # Draw the path from Q-values
    path_edges = list(zip(path[:-1], path[1:]))
    nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='red', width=2, style='solid', arrows=True, arrowstyle='-|>', arrowsize=20)

    # Calculate and draw the shortest path using NetworkX
    try:
        nx_path = nx.shortest_path(G, source=start_node, target=end_node)
        nx_path_edges = list(zip(nx_path[:-1], nx_path[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=nx_path_edges, edge_color='green', width=2, style='dashed', arrows=True, arrowstyle='-|>', arrowsize=15)
    except nx.NetworkXNoPath:
        print("No path exists between the start and end nodes using NetworkX.")
    print(len(nx_path_edges), len(path_edges))
    # Save and display the graph
    file_name = f"shortest_path_{start_node}_to_{end_node}.png"
    # file_name = f"graph_3_2.png"
    file_path = os.path.join(save_dir, file_name)
    plt.title(f'Shortest Path from {start_node} to {end_node}')
    plt.savefig(file_path)
    plt.close()  # Close the plot to free up resources

# Example usage:
visualize_shortest_path(G, pos, Q, adj_list, start_node=inst, end_node=tast)



