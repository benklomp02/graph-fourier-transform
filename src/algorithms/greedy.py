import numpy as np
from itertools import combinations
from typing import List

from src.utils.verifications import is_orthonormal_basis
from src.utils.graphs import path, comet, sensor
from src.utils.measurements import F, W


# Avoid KeyError by using a sorted tuple as key
def make_tuple(a: int, b: int) -> int:
    return (a, b) if a < b else (b, a)


def compute_greedy_basis_undirected(n: int, weights: List[List[int]]) -> np.ndarray:
    """Computes the greedy basis.

    Args:
        n (int): The number of vertices
        weights (List[List[int]]): The weights for an undirected graph input for n vertices

    Returns:
        np.ndarray: An orthonormal basis
    """
    tau = {(1 << i) for i in range(n)}
    memo = {
        make_tuple(1 << i, 1 << j): weights[i][j] for j in range(n) for i in range(j)
    }
    basis = []
    for _ in range(n - 1):
        A, B = max(
            combinations(tau, 2),
            key=lambda ab: memo[make_tuple(*ab)]
            / (ab[0].bit_count() * ab[1].bit_count()),
        )
        del memo[make_tuple(A, B)]
        memo.update(
            {
                make_tuple(A | B, C): memo[make_tuple(A, C)] + memo[make_tuple(B, C)]
                for C in (tau - {A, B})
            }
        )
        tau = tau - {A, B} | {A | B}
        t = 1 / np.sqrt(A.bit_count() * B.bit_count() * (A.bit_count() + B.bit_count()))
        a = np.isin(np.arange(n), [i for i in range(32) if (A >> i) & 1]).astype(int)
        b = np.isin(np.arange(n), [i for i in range(32) if (B >> i) & 1]).astype(int)
        u = -t * B.bit_count() * a + t * A.bit_count() * b
        basis.append(u)
    basis.append(np.sqrt(1 / n) * np.ones(n))
    return np.column_stack(basis[::-1])


def run_example(n: int, weights: List[List[int]]):
    basis = compute_greedy_basis_undirected(n, weights)
    np.set_printoptions(precision=3, suppress=True, linewidth=100)
    print("Greedy basis:")
    print(basis)
    assert is_orthonormal_basis(basis)


if __name__ == "__main__":
    run_example(*comet(5))
