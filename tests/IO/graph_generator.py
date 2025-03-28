import numpy as np
import networkx as nx
from itertools import combinations
from random import shuffle

from plot.utils.visualisation import visualize_graph
from tests.IO.constants import MAX_WEIGHT


# ---- some graph generators ---
def random_simple_graph(
    n: int, is_directed: bool, p: float = 0.5, show_visualization=False
) -> np.ndarray:
    if is_directed:
        return _simple_graph_directed(n, p, show_visualization)
    else:
        return _simple_graph_undirected(n, p, show_visualization)


def _simple_graph_undirected(
    n: int, p: float = 0.5, show_visualization=False
) -> np.ndarray:
    """Creates a undirected, simple and connected graph."""
    assert n >= 3
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for x, y in combinations(range(n), 2):
        if np.random.rand() < p:
            wt = np.random.randint(1, MAX_WEIGHT)
            G.add_edge(x, y, weight=wt)
    if not nx.is_connected(G):
        zhks = list(nx.connected_components(G))
        for A, B in zip(zhks, zhks[1:]):
            x, y = np.random.choice(list(A)), np.random.choice(list(B))
            wt = np.random.randint(1, MAX_WEIGHT)
            G.add_edge(x, y, weight=wt)
    if show_visualization:
        visualize_graph(G)
    try:
        assert nx.is_connected(G)
    except AssertionError:
        print("The graph is not connected.")
        visualize_graph(G)
        exit(1)
    weights = nx.to_numpy_array(G)
    # Remove self-loops
    np.fill_diagonal(weights, 0)
    return weights


def _simple_graph_directed(
    n: int, p: float = 0.3, show_visualization=False
) -> np.ndarray:
    """Creates a directed, simple and strongly connected graph."""
    assert n >= 3
    DiG = nx.DiGraph()
    DiG.add_nodes_from(range(n))
    # Hardcode the base case
    DiG.add_edge(0, 1, weight=np.random.randint(1, MAX_WEIGHT))
    DiG.add_edge(1, 2, weight=np.random.randint(1, MAX_WEIGHT))
    DiG.add_edge(2, 0, weight=np.random.randint(1, MAX_WEIGHT))
    i = 3
    while i < n:
        # let us create a strongly connected component
        _n = np.random.randint(1, n - i + 1)
        _adj = list(range(i, i + _n))
        shuffle(_adj)
        for x, y in zip(_adj, _adj[1:]):
            DiG.add_edge(x, y, weight=np.random.randint(1, MAX_WEIGHT))
        DiG.add_edge(_adj[-1], _adj[0], weight=np.random.randint(1, MAX_WEIGHT))
        # Then we connect the strongly connected components
        x_new, y_con = np.random.choice(_adj), np.random.choice(range(i))
        DiG.add_edge(x_new, y_con, weight=np.random.randint(1, MAX_WEIGHT))
        y_new, x_con = np.random.choice(_adj), np.random.choice(range(i))
        while x_con == y_con:  # There are at least three nodes...
            x_con = np.random.choice(range(i))
        DiG.add_edge(x_con, y_new, weight=np.random.randint(1, MAX_WEIGHT))
        i += _n
    # Let's add some random edges
    vertices = list(range(n))
    shuffle(vertices)
    for x, y in combinations(range(n), 2):
        if np.random.rand() < p:
            DiG.add_edge(x, y, weight=np.random.randint(1, MAX_WEIGHT))
    if show_visualization:
        visualize_graph(DiG)
    # Check if the graph is strongly connected
    try:
        assert nx.is_strongly_connected(DiG)
    except AssertionError:
        print("The graph is not strongly connected")
        visualize_graph(DiG)
        exit(1)
    weights = nx.to_numpy_array(DiG)
    # Remove self-loops
    np.fill_diagonal(weights, 0)
    return weights


# --- read test cases ---
def next_graph_input(f):
    line = f.readline()
    if line == "":
        raise EOFError
    n = int(line)
    weights = np.array([list(map(float, f.readline().split())) for _ in range(n)])
    return n, weights
