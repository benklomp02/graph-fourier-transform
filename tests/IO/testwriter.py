import numpy as np

from tests.IO.constants import *
from tests.IO.graph_generator import random_simple_graph


def _write_test_case(max_size: int):
    assert max_size >= 3
    with open(
        f"public/input/{"directed" if IS_DIRECTED else "undirected"}/input_Nmax{max_size}_t{num_tests}.txt",
        "w",
    ) as f:
        print(NUM_TESTS, file=f)
        for _ in range(NUM_TESTS):
            n = np.random.randint(3, 1 + max_size)
            print(n, file=f)
            weights = random_simple_graph(n, IS_DIRECTED)
            for row in weights:
                print(*row, file=f)


def _write_test_case_fixed_size(N: int):
    with open(
        f"public/input/{"directed" if IS_DIRECTED else "undirected"}/input_N{N}_t{NUM_TESTS}.txt",
        "w",
    ) as f:
        print(NUM_TESTS, file=f)
        for _ in range(NUM_TESTS):
            print(N, file=f)
            weights = random_simple_graph(N, IS_DIRECTED)
            for row in weights:
                print(*row, file=f)


def _write_test_case_set_fixed_size():
    """Writes test cases for either directed and undirected graphs."""
    assert 3 <= FIXED_SIZE_N_LOW <= FIXED_SIZE_N_HIGH
    for N in range(FIXED_SIZE_N_LOW, FIXED_SIZE_N_HIGH + 1):
        _write_test_case_fixed_size(N)


def _write_test_case_set():
    """Writes test cases for either directed and undirected graphs."""
    _write_test_case(
        MAX_NUM_VERT_LOW,
    )
    _write_test_case(
        MAX_NUM_VERT_HIGH,
    )


if __name__ == "__main__":
    # _write_test_case_set_fixed_size()
    # _write_test_case_set()
    ...
