import time
from statistics import mean
import numpy as np

from tests.IO.graph_generator import next_graph_input
from src.utils.objectives import S_undirected


# --- Measures of performance ---
def average_time(f, compute_basis):
    num_tests = int(f.readline())

    def compute_and_stop():
        n, weights = next_graph_input(f)
        start = time.time()
        compute_basis(n, weights)
        return time.time() - start

    return mean(compute_and_stop() for _ in range(num_tests))


# --- Measures of quality ---
def relative_error_single_vector(
    basis: np.ndarray, alt_basis: np.ndarray, weights: np.ndarray
) -> float:
    n = basis.shape[0]
    assert basis.shape == alt_basis.shape == (n,)
    return (
        S_undirected(basis, weights) - (s_alt := S_undirected(alt_basis, weights))
    ) / s_alt


def relative_error(
    basis: np.ndarray, alt_basis: np.ndarray, weights: np.ndarray
) -> float:
    assert basis.shape == alt_basis.shape
    return (
        sum(S_undirected(u, weights) for u in basis.T)
        - (s_alt := sum(S_undirected(u, weights) for u in alt_basis.T))
    ) / s_alt
