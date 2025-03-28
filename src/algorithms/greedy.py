import numpy as np
from line_profiler import profile

from src.utils.linalg import build_masked_array, arg_max_greedy


def compute_greedy_basis(n: int, weights: np.ndarray) -> np.ndarray:
    """Computes the greedy basis.

    Args:
        n (int): The number of vertices
        weights (np.ndarray): The weights for a graph input for n vertices
    Returns:
        np.ndarray: An orthonormal basis
    """
    return _compute(n, weights)


def compute_greedy_basis_dir(n: int, weights: np.ndarray) -> np.ndarray:
    """Computes the greedy basis.

    Args:
        n (int): The number of vertices
        weights (np.ndarray): The weights for a graph input for n vertices
    Returns:
        np.ndarray: An orthonormal basis
    """
    return _compute_dir(n, weights)


def _compute(n: int, weights: np.ndarray) -> np.ndarray:
    tau = np.array([(1 << i) for i in range(n)])
    memo = weights.copy()
    basis = []
    for k in range(n - 1):
        gi, gj = arg_max_greedy(n - k, tau, memo)
        memo[gi] += memo[gj]
        memo[gi][gi] = 0
        mask = np.arange(memo.shape[0]) != gj
        memo = memo[mask][:, mask]
        t = 1 / np.sqrt(
            tau[gi].bit_count()
            * tau[gj].bit_count()
            * (tau[gi].bit_count() + tau[gj].bit_count())
        )
        a = build_masked_array(tau[gi], n)
        b = build_masked_array(tau[gj], n)
        u = -t * tau[gj].bit_count() * a + t * tau[gi].bit_count() * b
        tau[gi] |= tau[gj]
        tau = np.delete(tau, gj)
        basis.append(u)
    basis.append(np.sqrt(1 / n) * np.ones(n))
    return np.column_stack(basis[::-1])


def _compute_dir(n: int, weights: np.ndarray) -> np.ndarray:
    tau = {(1 << i) for i in range(n)}
    memo = {(1 << i, 1 << j): weights[i][j] for j in range(n) for i in range(n)}
    basis = []
    for _ in range(n - 1):
        A, B = max(
            [(A, B) for A in tau for B in tau if A != B],
            key=lambda ab: memo[*ab] / (ab[0].bit_count() * ab[1].bit_count()),
        )
        del memo[A, B]
        del memo[B, A]
        tau -= {A, B}
        memo.update({(A | B, C): memo[A, C] + memo[B, C] for C in tau})
        memo.update({(C, A | B): memo[C, A] + memo[C, B] for C in tau})
        tau.add(A | B)
        t = 1 / np.sqrt(A.bit_count() * B.bit_count() * (A.bit_count() + B.bit_count()))
        a = build_masked_array(A, n)
        b = build_masked_array(B, n)
        u = -t * B.bit_count() * a + t * A.bit_count() * b
        basis.append(u)
    basis.append(np.sqrt(1 / n) * np.ones(n))
    return np.column_stack(basis[::-1])
