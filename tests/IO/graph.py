import numpy as np
import networkx as nx
from itertools import combinations
from typing import Tuple

from plot.utils.visualisation import visualize_graph_from_weights


# ---- some graph generators ---
def random_simple_graph(
    n: int, is_directed: bool, show_visualization=False
) -> np.ndarray:
    """Builds a (strongly) connected random simple graph with n vertices and p as the probability of an edge.

    Args:
        n (int): Number of vertices.
        is_directed (bool): True, if the graph should be directed.
        show_visualization (bool, optional): Plots the graph if True. Defaults to False.

    Returns:
        np.ndarray: An weighted adjacency matrix of the graph.
    """
    if is_directed:
        return _compute_directed(n, show_visualization)
    else:
        return _compute_undirected(n, show_visualization)


def _compute_distance(x: Tuple[int, int], y: Tuple[int, int]) -> float:
    """Computes the distance between two points."""
    return np.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2)


def _compute_undirected(n, show_visualization: bool = False) -> np.ndarray:
    """Creates a undirected, simple and connected graph."""
    assert n >= 3
    thres = 1 / np.sqrt(n)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    while True:
        points = [(np.random.rand(), np.random.rand()) for _ in range(n)]
        indices = list(range(n))
        list.sort(indices, key=points.__getitem__)
        for i in range(1, n):
            for j in range(i):
                if _compute_distance(points[indices[i]], points[indices[j]]) < thres:
                    G.add_edge(indices[i], indices[j], weight=1)
        if nx.is_connected(G):
            break
        thres = thres * 1.1  # Increase the threshold
    weights = nx.to_numpy_array(G)
    if show_visualization:
        visualize_graph_from_weights(weights)
    return weights


def _compute_directed(n, show_visualization) -> np.ndarray:
    """Creates a directed, simple and weakly connected graph."""
    assert n >= 3
    weights = _compute_undirected(n, False)
    for i, j in combinations(range(n), 2):
        if np.random.rand() < 0.5:
            weights[j][i] = 0
        else:
            weights[i][j] = 0
    if show_visualization:
        visualize_graph_from_weights(weights)
    return weights


def write_graph_input(n: int, weights: np.ndarray, f):
    """Writes a graph input to a file."""
    m = np.count_nonzero(weights)
    print(n, m, file=f)
    for i in range(n):
        for j in range(n):
            if weights[i][j] != 0:
                print(i, j, weights[i][j], file=f)


def read_graph_input(f):
    """Reads the next graph input from a file."""
    line = f.readline()
    if line == "":
        raise EOFError
    n, m = map(int, line.split())
    # Read the adjacency list and store as adjacency matrix
    DiG = nx.DiGraph()
    DiG.add_nodes_from(range(n))
    for _ in range(m):
        line = f.readline()
        x, y, wt = line.split()
        DiG.add_edge(int(x), int(y), weight=float(wt))
    weights = nx.to_numpy_array(DiG)
    return n, weights


# ---- some random signal generators ----


def random_signal(n: int) -> np.ndarray:
    """Generates a random signal of length n of only 0 and 1."""
    return np.random.randint(0, 2, size=n)


def random_signal_matrix(n: int, m: int) -> np.ndarray:
    """Generates a random signal matrix of shape (n, m)."""
    return np.random.randint(0, 2, size=(n, m))


# ---- some example usage ----


def _example():
    """Example usage of the graph generator."""
    n = 10
    is_directed = False
    show_visualization = True

    # Generate a random simple graph
    weights = random_simple_graph(n, is_directed, show_visualization)

    # Print the generated graph
    print("Generated Graph (Adjacency Matrix):")
    print(weights)


if __name__ == "__main__":
    _example()
