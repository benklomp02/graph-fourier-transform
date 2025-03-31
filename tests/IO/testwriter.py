import numpy as np

from tests.IO.constants import *
from tests.IO.graph import random_simple_graph, write_graph_input


def _write_test_case(max_size: int, p: float = 0.5):
    assert max_size >= 3
    with open(
        f"public/input/{"directed" if IS_DIRECTED else "undirected"}/Nmax{max_size}_t{NUM_TESTS}.txt",
        "w",
    ) as f:
        print(NUM_TESTS, file=f)
        for _ in range(NUM_TESTS):
            n = np.random.randint(3, 1 + max_size)
            weights = random_simple_graph(n, IS_DIRECTED, p)
            write_graph_input(n, weights, f)


def _write_test_case_fixed_size(N: int, p: float = 0.5):
    with open(
        f"public/input/{"directed" if IS_DIRECTED else "undirected"}/N{N}_t{NUM_TESTS}.txt",
        "w",
    ) as f:
        print(NUM_TESTS, file=f)
        for _ in range(NUM_TESTS):
            weights = random_simple_graph(N, IS_DIRECTED, p)
            write_graph_input(N, weights, f)


def _write_test_case_set_fixed_size():
    """Writes test cases for either directed and undirected graphs."""
    assert 3 <= FIXED_SIZE_N_LOW <= FIXED_SIZE_N_HIGH
    for N in range(FIXED_SIZE_N_LOW, FIXED_SIZE_N_HIGH + 1):
        _write_test_case_fixed_size(N)


if __name__ == "__main__":
    _write_test_case_set_fixed_size()
    # _write_test_case(max_size=40)
