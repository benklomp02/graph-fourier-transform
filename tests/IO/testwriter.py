import numpy as np

from tests.IO.constants import *
from tests.IO.graph import random_simple_graph, write_graph_input


def _write_test_case(
    max_size: int, p: float = 0.5, is_directed=IS_DIRECTED, num_tests=NUM_TESTS
):
    assert max_size >= 3
    with open(
        f"public/input/{"directed" if is_directed else "undirected"}/Nmax{max_size}_t{num_tests}.txt",
        "w",
    ) as f:
        print(num_tests, file=f)
        for _ in range(num_tests):
            n = np.random.randint(3, 1 + max_size)
            weights = random_simple_graph(n, is_directed)
            write_graph_input(n, weights, f)


def _write_test_case_fixed_size(
    N: int, p: float = 0.5, is_directed=IS_DIRECTED, num_tests=NUM_TESTS
):
    with open(
        f"public/input/{"directed" if is_directed else "undirected"}/N{N}_t{num_tests}.txt",
        "w",
    ) as f:
        print(num_tests, file=f)
        for _ in range(num_tests):
            weights = random_simple_graph(N, is_directed)
            write_graph_input(N, weights, f)


def main():
    for N in [5, 20, 30, 40, 50, 60]:
        for num_tests in [20, 100]:
            _write_test_case(N, num_tests=num_tests, is_directed=True)
            _write_test_case(N, num_tests=num_tests, is_directed=False)
    for N in [3, 4, 5, 6, 7, 8, 20, 30, 40, 50, 60]:
        for num_tests in [20, 100]:
            _write_test_case_fixed_size(N, num_tests=num_tests, is_directed=True)
            _write_test_case_fixed_size(N, num_tests=num_tests, is_directed=False)


if __name__ == "__main__":
    # main()
    _write_test_case_fixed_size(1024, num_tests=1, is_directed=True)
    _write_test_case_fixed_size(1024, num_tests=1, is_directed=False)
