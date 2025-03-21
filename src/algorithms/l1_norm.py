import numpy as np
from functools import reduce, partial
from typing import List

from src.utils.measurements import S
from src.utils.verifications import is_orthonormal_basis
from src.utils.graphs import comet

MAX_SIZE = 8


def get_all_partition_matrices(n: int, m: int):
    """Generating all partition matrices for a signal of size n with m different values."""
    assert MAX_SIZE >= n >= m >= 2
    M = np.zeros((n, m))

    def f(i, free, toBeUsed):
        if i == n:
            yield M
        else:
            for j in range(m):
                if (toBeUsed >> j) & 0x1:
                    M[i][j] = 1
                    yield from f(i + 1, free ^ (1 << j), toBeUsed ^ (1 << j))
                    M[i][j] = 0
            if n - i > toBeUsed.bit_count():
                for j in range(m):
                    if (free >> j) & 0x1:
                        M[i][j] = 1
                        yield from f(i + 1, free, toBeUsed)
                        M[i][j] = 0

    yield from f(0, 0, (1 << m) - 1)


def compute_first_x(M: np.ndarray) -> np.ndarray:
    """Solves the minimisation problem for constant signal."""
    c1, c2 = np.sum(M, axis=0)
    a = np.array([1.0, -c1 / c2])
    _x = M @ a
    return _x / np.linalg.norm(_x)


def compute_x(M: np.ndarray, U: np.ndarray) -> np.ndarray:
    """Solves the general minimisation problem."""
    _, _, Vh = np.linalg.svd(U.T @ M)
    _x = M @ Vh[-1, :]
    return _x / np.linalg.norm(_x)


def expand_basis_set(
    basis: List[np.ndarray], k: int, weights: List[List], n: int
) -> List[np.ndarray]:
    """Computes the kth iteration of the l1 norm algorithm."""
    assert k >= 3
    U = np.column_stack(basis)
    uk = min(
        map(partial(compute_x, U=U), get_all_partition_matrices(n, k)),
        key=partial(S, weights=weights),
    )
    return basis + [uk]


def compute_l1_norm_basis_undirected(n: int, weights: List[List[int]]) -> np.ndarray:
    """Computes the l1 norm basis under the function S(x) in exponential time.

    Args:
        n (int): The number of vertices
        weights (List[List[int]]): The weights for an undirected graph input for n vertices

    Returns:
        np.ndarray: An orthonormal basis
    """
    assert 2 <= n <= MAX_SIZE
    assert len(weights) == n and all(len(row) == n for row in weights)
    u1 = np.ones(n) / np.sqrt(n)
    u2 = min(
        map(compute_first_x, get_all_partition_matrices(n, 2)),
        key=partial(S, weights=weights),
    )
    basis = reduce(
        partial(expand_basis_set, weights=weights, n=n), range(3, 1 + n), [u1, u2]
    )
    return np.column_stack(basis)


def run_example(n, weights):
    basis = compute_l1_norm_basis_undirected(n, weights)
    print("l1 norm basis:")
    print(basis)
    assert is_orthonormal_basis(basis)


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True, linewidth=100)
    run_example(*comet(MAX_SIZE))
