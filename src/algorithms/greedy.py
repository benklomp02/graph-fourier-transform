import numpy as np
from itertools import combinations
from line_profiler import profile

from src.utils.linalg import build_masked_array, arg_max_greedy
from tests.utils.verifications import is_orthonormal_basis
from tests.IO.examples import comet


@profile
def make_tuple(a: int, b: int) -> int:
    # Avoid KeyError by using a sorted tuple as key
    return (a, b) if a < b else (b, a)


@profile
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
        a = np.array([1 if A & (1 << i) else 0 for i in range(n)])
        b = np.array([1 if B & (1 << i) else 0 for i in range(n)])
        u = -t * B.bit_count() * a + t * A.bit_count() * b
        basis.append(u)
    basis.append(np.sqrt(1 / n) * np.ones(n))
    return np.column_stack(basis[::-1])


def arg_max_greedy_stupid(n, tau, memo):
    max_val = 0
    best_a = best_b = 0
    for j in range(1, n):
        for i in range(j):
            val = memo[i, j] / (tau[i].bit_count() * tau[j].bit_count())
            if val > max_val:
                max_val = val
                best_a, best_b = i, j
    return best_a, best_b


@profile
def compute_greedy_basis_undirected_opt(n: int, weights: np.ndarray) -> np.ndarray:
    """Computes the greedy basis.

    Args:
        n (int): The number of vertices
        weights (np.ndarray): The weights for an undirected graph input for n vertices

    Returns:
        np.ndarray: An orthonormal basis
    """
    # N groups
    tau = np.array([(1 << i) for i in range(n)])
    memo = weights.copy()
    basis = []
    for k in range(n - 1):
        gi, gj = arg_max_greedy(n - k, tau, memo)
        memo[gi] += memo[gj]
        memo[gi, gi] = 0
        memo = np.delete(memo, gj, axis=0)
        memo = np.delete(memo, gj, axis=1)
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
        a = build_masked_array(A, n)
        b = build_masked_array(B, n)
        u = -t * B.bit_count() * a + t * A.bit_count() * b
        basis.append(u)
    basis.append(np.sqrt(1 / n) * np.ones(n))
    return np.column_stack(basis[::-1])


def run_example(n: int, weights: np.ndarray, is_directed: bool = False):
    if is_directed:
        basis = compute_greedy_basis_directed(n, weights)
        assert is_orthonormal_basis(basis)
    else:
        basis = compute_greedy_basis_undirected(n, weights)
        print("greedy basis undirected")
        print(basis)
        assert is_orthonormal_basis(basis)
        basis = compute_greedy_basis_undirected_opt(n, weights)
        print("greedy basis undirected opt")
        print(basis)
        assert is_orthonormal_basis(basis)


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True, linewidth=100)
    run_example(*comet(20), is_directed=False)
