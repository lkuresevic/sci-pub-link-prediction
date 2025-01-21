from torch_geometric.datasets import Planetoid
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import networkx as nx
import random
from torch_geometric.utils import to_networkx
import torch_geometric.transforms as T

def load_cora():
    return Planetoid(root='/tmp/Cora', name='Cora')

def convert_to_networkx(graph, n_sample=None):

    g = to_networkx(graph, node_attrs=["x"])
    y = graph.y.numpy()

    if n_sample is not None:
        sampled_nodes = random.sample(list(g.nodes), n_sample)
        g = g.subgraph(sampled_nodes)
        y = y[sampled_nodes]

    return g, y

def plot_graph(g, y):

    plt.figure(figsize=(9, 7))
    nx.draw_spring(g, node_size=30, arrows=False, node_color=y)
    plt.show() 

def split_data(graph):
    split = T.RandomLinkSplit(
        num_val=0.05,
        num_test=0.1,
        is_undirected=True,
        add_negative_train_samples=False,
        neg_sampling_ratio=1.0,
    )
    train_data, val_data, test_data = split(graph)
    
    return train_data, val_data, test_data    
    
