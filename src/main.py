import numpy as np
import time

from src.algorithms.greedy import compute_greedy_basis
from src.algorithms.laplacian import compute_laplacian_basis
from src.algorithms.l1_norm import compute_l1_norm_basis
from plot.utils.visualisation import visualize_graph_from_weights
from tests.IO.examples import comet
from tests.IO.graph_generator import next_graph_input
from tests.utils.verifications import is_orthonormal_basis


def _run_file(
    compute_basis,
    file_path: str,
    console_output: bool = False,
    visualize: bool = False,
):
    try:
        with open(file_path, "r") as f:
            num_tests = int(f.readline())
            for _ in range(num_tests):
                _run_example(
                    compute_basis,
                    *next_graph_input(f),
                    console_output,
                    visualize,
                )
    except FileNotFoundError:
        print(f"\033[91mERROR: File not found: {file_path}\033[0m")


def _run_example(
    compute_basis,
    n: int,
    weights: np.ndarray,
    console_output: bool = False,
    visualize: bool = False,
):
    try:
        start_time = time.time()
        basis = compute_basis(n, weights)
        assert is_orthonormal_basis(basis)
        end_time = time.time()
        if console_output:
            print(f"\033[92m{compute_basis.__name__}: {end_time - start_time:.5f}s")
            with np.printoptions(linewidth=100, threshold=1000, precision=3):
                print(basis)
            print("\033[0m")
    except AssertionError:
        if console_output:
            print(f"\033[91mERROR: {compute_basis.__name__}\n")
            with np.printoptions(linewidth=100, threshold=1000, precision=3):
                print(basis)
            print("\033[0m")
        if visualize:
            visualize_graph_from_weights(weights)
        raise


if __name__ == "__main__":
    # Example usage
    from functools import partial

    _run_file(
        lambda args: compute_greedy_basis(*args, is_directed=False),
        "public/input/directed/input_N8_t100.txt",
        console_output=False,
    )
