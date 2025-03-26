import numpy as np
import networkx as nx
from itertools import combinations
from random import shuffle

from plot.utils.visualisation import visualize_graph

# ---- some constants ---
MAX_SIZE_LARGE = 20
MAX_SIZE_SMALL = 7
MAX_WEIGHT = 10
NUM_TESTS = 100


# ---- some graph generators ---
def simple_graph_undirected(
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
    return nx.to_numpy_array(G, weight="weight")


def simple_graph_directed(
    n: int, p: float = 0.3, show_visualization=False
) -> np.ndarray:
    """Creates a directed, simple and strongly connected graph."""
    assert n >= 3
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    # Hardcode the base case
    G.add_edge(0, 1, weight=np.random.randint(1, MAX_WEIGHT))
    G.add_edge(1, 2, weight=np.random.randint(1, MAX_WEIGHT))
    G.add_edge(2, 0, weight=np.random.randint(1, MAX_WEIGHT))
    i = 3
    while i < n:
        # let us create a strongly connected component
        _n = np.random.randint(1, n - i + 1)
        _adj = list(range(i, i + _n))
        shuffle(_adj)
        for x, y in zip(_adj, _adj[1:]):
            G.add_edge(x, y, weight=np.random.randint(1, MAX_WEIGHT))
        G.add_edge(_adj[-1], _adj[0], weight=np.random.randint(1, MAX_WEIGHT))
        # Then we connect the strongly connected components
        x_new, y_con = np.random.choice(_adj), np.random.choice(range(i))
        G.add_edge(x_new, y_con, weight=np.random.randint(1, MAX_WEIGHT))
        y_new, x_con = np.random.choice(_adj), np.random.choice(range(i))
        while x_con == y_con:  # There are at least three nodes...
            x_con = np.random.choice(range(i))
        G.add_edge(x_con, y_new, weight=np.random.randint(1, MAX_WEIGHT))
        i += _n
    # Let's add some random edges
    vertices = list(range(n))
    shuffle(vertices)
    for x, y in combinations(range(n), 2):
        if np.random.rand() < p:
            G.add_edge(x, y, weight=np.random.randint(1, MAX_WEIGHT))
    if show_visualization:
        visualize_graph(G)
    # Check if the graph is strongly connected
    try:
        assert nx.is_strongly_connected(G)
    except AssertionError:
        print("The graph is not strongly connected")
        visualize_graph(G)
        exit(1)
    return nx.to_numpy_array(G, weight="weight")


# --- read test cases ---
def next_graph_input(f):
    line = f.readline()
    if line == "":
        raise EOFError
    n = int(line)
    weights = [list(map(int, f.readline().split())) for _ in range(n)]
    return n, weights


# --- write test cases ---
def write_test_case(
    create_graph, max_size: int, is_directed: bool, num_tests=NUM_TESTS
):
    assert max_size >= 3
    with open(
        f"public/input/{"directed" if is_directed else "undirected"}/input_Nmax{max_size - 1}_t{num_tests}.txt",
        "w",
    ) as f:
        print(NUM_TESTS, file=f)
        for _ in range(NUM_TESTS):
            n = np.random.randint(3, max_size)
            print(n, file=f)
            weights = create_graph(n)
            for row in weights:
                print(*row, file=f)


def write_test_case_fixed_size(
    create_graph, fixed_size: int, is_directed: bool, num_tests=NUM_TESTS
):
    with open(
        f"public/input/{"directed" if is_directed else "undirected"}/input_N{fixed_size}_N{num_tests}.txt",
        "w",
    ) as f:
        print(num_tests, file=f)
        for _ in range(num_tests):
            print(fixed_size, file=f)
            weights = create_graph(n=fixed_size)
            for row in weights:
                print(*row, file=f)


if __name__ == "__main__":
    write_test_case(simple_graph_directed, MAX_SIZE_SMALL, is_directed=True)
    write_test_case(simple_graph_directed, MAX_SIZE_LARGE, is_directed=True)
