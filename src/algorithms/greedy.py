import numpy as np
from line_profiler import profile

from src.utils.c_linalg import build_masked_array, arg_max_greedy


def compute_greedy_basis(n: int, weights: np.ndarray) -> np.ndarray:
    """Computes the greedy basis.

    Args:
        n (int): The number of vertices
        weights (np.ndarray): The weights for a graph input for n vertices
    Returns:
        np.ndarray: An orthonormal basis
    """
    return _compute(n, weights)


@profile
def _compute(n: int, weights: np.ndarray) -> np.ndarray:
    assert n <= 60
    tau = np.left_shift(1, np.arange(n))
    memo = weights.copy()
    basis = []
    for k in range(n - 1):
        gi, gj = arg_max_greedy(n - k, tau, memo)
        memo[gi] += memo[gj]
        memo[gi][gi] = 0
        mask = np.arange(memo.shape[0]) != gj
        memo = memo[mask][:, mask]
        set_i, set_j = tau[gi], tau[gj]
        nbit_i, nbit_j = set_i.bit_count(), set_j.bit_count()
        t = 1 / np.sqrt(nbit_i * nbit_j * (nbit_i + nbit_j))
        a = build_masked_array(set_i, n)
        b = build_masked_array(set_j, n)
        u = -t * nbit_j * a + t * nbit_i * b
        tau[gi] |= set_j
        tau = np.delete(tau, gj)
        basis.append(u)
    basis.append(np.sqrt(1 / n) * np.ones(n))
    return np.column_stack(basis[::-1])
