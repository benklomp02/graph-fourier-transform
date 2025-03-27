import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


def visualize_graph_from_weights(weights: np.ndarray):
    nx_graph = nx.from_numpy_array(weights, create_using=nx.DiGraph)
    visualize_graph(nx_graph)


def visualize_graph(G: nx.Graph):
    pos = nx.spring_layout(G)
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color="lightblue",
        edge_color="gray",
        node_size=500,
        font_size=10,
    )
    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.show()
