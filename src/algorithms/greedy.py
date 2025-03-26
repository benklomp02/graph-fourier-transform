import numpy as np
from itertools import combinations

from tests.utils.verifications import is_orthonormal_basis
from tests.IO.examples import comet


# Avoid KeyError by using a sorted tuple as key
def make_tuple(a: int, b: int) -> int:
    return (a, b) if a < b else (b, a)


def compute_greedy_basis_undirected(n: int, weights: np.ndarray) -> np.ndarray:
    """Computes the greedy basis.

    Args:
        n (int): The number of vertices
        weights (np.ndarray): The weights for an undirected graph input for n vertices

    Returns:
        np.ndarray: An orthonormal basis
    """
    tau = {(1 << i) for i in range(n)}
    memo = {(1 << i, 1 << j): weights[i][j] for j in range(n) for i in range(j)}
    basis = []
    for _ in range(n - 1):
        A, B = max(
            combinations(tau, 2),
            key=lambda ab: memo[make_tuple(*ab)]
            / (ab[0].bit_count() * ab[1].bit_count()),
        )
        del memo[make_tuple(A, B)]
        tau -= {A, B}
        memo.update(
            {
                make_tuple(A | B, C): memo[make_tuple(A, C)] + memo[make_tuple(B, C)]
                for C in tau
            }
        )
        tau.add(A | B)
        t = 1 / np.sqrt(A.bit_count() * B.bit_count() * (A.bit_count() + B.bit_count()))
        a = np.isin(np.arange(n), [i for i in range(32) if (A >> i) & 0x1]).astype(int)
        b = np.isin(np.arange(n), [i for i in range(32) if (B >> i) & 0x1]).astype(int)
        u = -t * B.bit_count() * a + t * A.bit_count() * b
        basis.append(u)
    basis.append(np.sqrt(1 / n) * np.ones(n))
    return np.column_stack(basis[::-1])


def compute_greedy_basis_directed(n: int, weights: np.ndarray) -> np.ndarray:
    """Computes the greedy basis.

    Args:
        n (int): The number of vertices
        weights (np.ndarray): The weights for a directed graph input for n vertices

    Returns:
        np.ndarray: An orthonormal basis
    """
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
        a = np.isin(np.arange(n), [i for i in range(32) if (A >> i) & 0x1]).astype(int)
        b = np.isin(np.arange(n), [i for i in range(32) if (B >> i) & 0x1]).astype(int)
        u = -t * B.bit_count() * a + t * A.bit_count() * b
        basis.append(u)
    basis.append(np.sqrt(1 / n) * np.ones(n))
    return np.column_stack(basis[::-1])


def run_example(n: int, weights: np.ndarray, is_directed: bool = False):
    if is_directed:
        basis = compute_greedy_basis_directed(n, weights)
        print("Greedy basis directed:")
        print(basis)
        assert is_orthonormal_basis(basis)
    else:
        basis = compute_greedy_basis_undirected(n, weights)
        print("Greedy basis undirected:")
        print(basis)
        assert is_orthonormal_basis(basis)


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True, linewidth=100)
    run_example(*comet(6), is_directed=False)
    run_example(*comet(6), is_directed=True)
