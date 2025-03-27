import numpy as np
from itertools import combinations
from line_profiler import profile
from plot.utils.visualisation import visualize_graph_from_weights

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
    # N groups
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


def run_file(file_path: str, is_directed: bool = False):
    with open(file_path, "r") as f:
        num_tests = int(f.readline())
        for _ in range(num_tests):
            n = int(f.readline())
            weights = np.array([list(map(int, f.readline().split())) for _ in range(n)])
            run_example(n, weights, is_directed)


def run_example(n: int, weights: np.ndarray, is_directed: bool = False):
    try:
        if is_directed:
            basis = compute_greedy_basis_directed(n, weights)
            assert is_orthonormal_basis(basis)
        else:
            basis = compute_greedy_basis_undirected(n, weights)
            assert is_orthonormal_basis(basis)
        print("Greedy basis is orthonormal!")
    except AssertionError:
        print("Greedy basis is not orthonormal:")
        print(basis)
        visualize_graph_from_weights(weights)
        raise


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True, linewidth=100)
    # run_example(*comet(6), is_directed=False)
    run_file("public/input/undirected/input_Nmax6_t100.txt")
