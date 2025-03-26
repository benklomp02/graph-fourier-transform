import numpy as np
from functools import reduce, partial
from typing import List
from line_profiler import profile

from src.utils.objectives import S_undirected, S_directed
from tests.utils.verifications import is_orthonormal_basis
from tests.IO.examples import comet
from src.utils.solver import solve_minimisation_problem
from src.utils.partition_matrix import (
    get_all_partition_matrices,
    get_all_solution_vectors_par,
)

MAX_SIZE = 8

# --- Sequential version ---


@profile
def expand_basis_set(
    basis: List[np.ndarray], k: int, weights: np.ndarray, n: int
) -> List[np.ndarray]:
    """Computes the kth iteration of the l1 norm algorithm."""
    assert k >= 3
    U = np.column_stack(basis)
    uk = min(
        map(
            partial(solve_minimisation_problem, U=U, is_constant=False),
            get_all_partition_matrices(n, k),
        ),
        key=partial(S_undirected, weights=weights),
    )
    return basis + [uk]


@profile
def compute_l1_norm_basis_undirected(n: int, weights: np.ndarray) -> np.ndarray:
    """Computes the l1 norm basis under the function S(x) in exponential time.

    Args:
        n (int): The number of vertices
        weights (np.ndarray): The weights for an undirected graph input for n vertices

    Returns:
        np.ndarray: An orthonormal basis
    """
    u1 = np.ones(n) / np.sqrt(n)
    u2 = min(
        map(
            partial(solve_minimisation_problem, is_constant=True),
            get_all_partition_matrices(n, 2),
        ),
        key=partial(S_undirected, weights=weights),
    )
    basis = reduce(
        partial(expand_basis_set, weights=weights, n=n), range(3, 1 + n), [u1, u2]
    )
    return np.column_stack(basis)


def compute_l1_norm_basis_directed(n: int, weights: np.ndarray) -> np.ndarray:
    """Computes the l1 norm basis under the function S(x) in exponential time.

    Args:
        n (int): The number of vertices
        weights (List[List[int]]): The weights for an undirected graph input for n vertices

    Returns:
        np.ndarray: An orthonormal basis
    """
    u1 = np.ones(n) / np.sqrt(n)
    u2 = min(
        map(
            partial(solve_minimisation_problem, is_constant=True),
            get_all_partition_matrices(n, 2),
        ),
        key=partial(S_directed, weights=weights),
    )
    basis = reduce(
        partial(expand_basis_set, weights=weights, n=n), range(3, 1 + n), [u1, u2]
    )
    return np.column_stack(basis)


# --- Parallel version ---
def argmin_par(n, k, solve_fn, score_fn):
    return min(get_all_solution_vectors_par(n, k, solve_fn), key=score_fn)


def expand_basis_set_par(
    basis: List[np.ndarray], k: int, weights: np.ndarray, n: int
) -> List[np.ndarray]:
    """Computes the kth iteration of the l1 norm algorithm."""
    assert k >= 3
    U = np.column_stack(basis)
    solve_fn = partial(solve_minimisation_problem, U=U, is_constant=False)
    score_fn = partial(S_undirected, weights=weights)
    uk = argmin_par(n, k, solve_fn, score_fn)
    return basis + [uk]


def compute_l1_norm_basis_undirected_par(n: int, weights: np.ndarray) -> np.ndarray:
    """Computes the l1 norm basis under the function S(x) in exponential time.

    Args:
        n (int): The number of vertices
        weights (List[List[int]]): The weights for an undirected graph input for n vertices

    Returns:
        np.ndarray: An orthonormal basis
    """
    u1 = np.ones(n) / np.sqrt(n)
    solve_fn = partial(solve_minimisation_problem, U=None, is_constant=True)
    score_fn = partial(S_undirected, weights=weights)
    u2 = argmin_par(n, 2, solve_fn, score_fn)
    basis = reduce(
        partial(expand_basis_set_par, weights=weights, n=n), range(3, 1 + n), [u1, u2]
    )
    return np.column_stack(basis)


# --- Example ---


def run_example(n: int, weights: np.ndarray):
    print("l1 norm basis:")
    basis = compute_l1_norm_basis_undirected(n, weights)
    print(basis)
    assert is_orthonormal_basis(basis)


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True, linewidth=100)
    n, weights = comet(8)
    run_example(n, np.array(weights))
