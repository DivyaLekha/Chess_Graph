import matplotlib.pyplot as plt
import networkx as nx
from .graph import board_to_graph

def plot_board_graph(board, path=None):
    data=board_to_graph(board)
    G=nx.Graph()
    G.add_nodes_from(range(data.num_nodes))
    G.add_edges_from(data.edge_index.t().tolist())
    pos={}
    for i in range(64):
        pos[i]=(i%8,i//8)
    for i in range(64,data.num_nodes):
        pos[i]=(i%8+0.08,i//8+0.08)
    plt.figure(figsize=(7,7))
    nx.draw_networkx(G,pos=pos,node_size=80,with_labels=False)
    plt.axis("off")
    if path: plt.savefig(path,bbox_inches="tight",dpi=180)
    return G
