import numpy as np
from itertools import combinations
import time
from statistics import mean
from typing import List
from src.utils.graph_generator import next_graph_input


def W(A: int, B: int, weights: List[List[int]]) -> float:
    return sum(
        weights[i][j]
        for i in range(0, 32)
        for j in range(0, 32)
        if (A >> i) & 1 and (B >> j) & 1
    )


def F(A: int, B: int, weights: List[List[int]]) -> float:
    return W(A, B, weights) / (A.bit_count() * B.bit_count())


def S(x: np.ndarray, weights: List[List[int]]) -> float:
    x = np.ravel(x)
    assert len(x) == len(weights)
    return sum(
        weights[i][j] * abs(x[i] - x[j]) for i, j in combinations(range(len(x)), 2)
    )


def S_directed(x: np.ndarray, weights: List[List[int]]) -> float:
    x = np.ravel(x)
    assert len(x) == len(weights)
    return sum(
        max((x[i] - x[j]) * weights[i][j], (x[j] - x[i]) * weights[j][i])
        for i, j in combinations(range(len(x)), 2)
    )


# --- Measures of perfomance ---
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
    basis: np.ndarray, alt_basis: np.ndarray, weights: List[List[int]]
) -> float:
    n = basis.shape[0]
    assert basis.shape == alt_basis.shape == (n,)
    return (S(basis, weights) - (s_alt := S(alt_basis, weights))) / s_alt


def relative_error(
    basis: np.ndarray, alt_basis: np.ndarray, weights: List[List[int]]
) -> float:
    assert basis.shape == alt_basis.shape
    return (
        sum(S(u, weights) for u in basis.T)
        - (s_alt := sum(S(u, weights) for u in alt_basis.T))
    ) / s_alt
