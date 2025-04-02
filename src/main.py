import numpy as np
import time
from typing import Callable
from line_profiler import profile

from src.algorithms.greedy import compute_greedy_basis
from src.algorithms.greedy_lg import compute_greedy_basis_lg
from src.algorithms.laplacian import compute_laplacian_basis
from src.algorithms.l1_norm import compute_l1_norm_basis
from src.algorithms.l1_norm_cpp import compute_l1_norm_basis_cpp
from plot.utils.visualisation import visualize_graph_from_weights
from tests.IO.examples import comet
from tests.IO.graph import read_graph_input
from tests.utils.verifications import is_orthonormal_basis


def _run_file(
    compute_basis: Callable[[int, np.ndarray], np.ndarray],
    file_path: str,
    console_output: bool = False,
    visualize: bool = False,
):
    """Runs the example from a file."""
    try:
        with open(file_path, "r") as f:
            num_tests = int(f.readline())
            for _ in range(num_tests):
                _run_example(
                    compute_basis,
                    *read_graph_input(f),
                    console_output,
                    visualize,
                )
    except FileNotFoundError:
        print(f"\033[91mERROR: File not found: {file_path}\033[0m")
    finally:
        print(
            f"\033[92mFinished running all examples using {compute_basis.__name__}.\033[0m"
        )


def _run_example(
    compute_basis: Callable[[int, np.ndarray], np.ndarray],
    n: int,
    weights: np.ndarray,
    console_output: bool = True,
    print_basis: bool = False,
    visualize: bool = False,
):
    """Runs a single example."""
    try:
        start_time = time.time()
        basis = compute_basis(n, weights)
        assert is_orthonormal_basis(basis)
        end_time = time.time()
        if console_output:
            print(
                f"\033[92m{compute_basis.__name__}: {end_time - start_time:.5f}s\033[0m"
            )
        if print_basis:
            print(basis)
    except AssertionError:
        if console_output:
            print(f"\033[91mERROR: {compute_basis.__name__}\n")
            if n < 10:
                print(basis)
            print("\033[0m")
        if visualize:
            visualize_graph_from_weights(weights)
        raise


def _run_on_every_basis(
    n: int,
    weights: np.ndarray,
    console_output: bool = False,
    print_basis=False,
    visualize: bool = False,
):
    _run_example(compute_l1_norm_basis, n, weights, console_output, visualize)
    _run_example(compute_l1_norm_basis_cpp, n, weights, console_output, visualize)
    _run_example(compute_greedy_basis, n, weights, console_output, visualize)
    _run_example(compute_greedy_basis_lg, n, weights, console_output, visualize)
    _run_example(compute_laplacian_basis, n, weights, console_output, visualize)


def example():
    """Main function to run the comet example."""
    np.set_printoptions(threshold=1000, precision=3, suppress=True)
    n = 8
    weights = comet(n)
    _run_on_every_basis(n, weights, True, True, False)


def main():
    _run_file(
        compute_greedy_basis,
        "public/input/directed/N60_t100.txt",
        True,
        False,
    )


if __name__ == "__main__":
    # example()
    main()
