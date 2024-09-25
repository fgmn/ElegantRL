import networkx as nx
import matplotlib.pyplot as plt
import random

def create_grid_graph(dim_x, dim_y):
    G = nx.grid_2d_graph(dim_x, dim_y)
    pos = dict((n, n) for n in G.nodes())  # 使用节点本身的标签作为坐标
    return G, pos

def remove_edges_randomly(G, percentage=10):
    # 计算要删除的边数
    num_edges_to_remove = len(G.edges()) * percentage // 100
    edges = list(G.edges())
    random.shuffle(edges)

    removed = 0
    for edge in edges:
        if removed < num_edges_to_remove:
            # 移除边，并检查图是否仍然连通
            G.remove_edge(*edge)
            if nx.is_connected(G):
                removed += 1
            else:
                # 如果移除后图变得不连通，重新添加该边
                G.add_edge(*edge)
        else:
            break

def draw_graph(G, pos, filename='grid_graph.png'):
    plt.figure(figsize=(8, 8))
    nx.draw(G, pos, with_labels=True, node_size=50, node_color='skyblue', edge_color='#333333')
    plt.title('Modified Grid Graph')
    
    # 保存图形到文件
    plt.savefig(filename)
    plt.close()  # 关闭图形，释放内存

def save_graph_to_file(G, filename):
     nx.write_adjlist(G, filename, delimiter=":")

def main():
    dim_x, dim_y = 10, 10
    G, pos = create_grid_graph(dim_x, dim_y)
    remove_edges_randomly(G, 40)  # 删除大约40%的边  (4*10*10+4*10)/2*60%=132
    draw_graph(G, pos)
    save_graph_to_file(G, "grid_graph.adjlist")  # 保存修改后的图

if __name__ == "__main__":
    main()
