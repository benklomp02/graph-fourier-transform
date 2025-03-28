import numpy as np
from functools import reduce, partial
from typing import List
from line_profiler import profile

from src.utils.objectives import S
from src.utils.solver import solve_minimisation_problem
from src.utils.partition_matrix import (
    get_all_partition_matrices,
    get_all_solution_vectors_parallel,
)

MAX_SIZE = 8

# --- Sequential version ---


def compute_l1_norm_basis(
    n: int, weights: np.ndarray, run_parallel: bool = False
) -> np.ndarray:
    """Computes the l1 norm basis under the function S(x) in exponential time.

    Args:
        n (int): The number of vertices
        weights (np.ndarray): The weights for a simple graph input for n vertices
        run_parallel (bool): Whether to run the parallel version

    Returns:
        np.ndarray: An orthonormal basis
    """
    if run_parallel:
        return _compute_parallel(n, weights)
    return _compute(n, weights)


def _expand_basis_set(
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
        key=partial(S, weights=weights),
    )
    return basis + [uk]


def _compute(n: int, weights: np.ndarray) -> np.ndarray:
    """Computes the l1 norm basis under the function S(x) in exponential time.

    Args:
        n (int): The number of vertices
        weights (np.ndarray): The weights for a simple graph input for n vertices

    Returns:
        np.ndarray: An orthonormal basis
    """
    u1 = np.ones(n) / np.sqrt(n)
    u2 = min(
        map(
            partial(solve_minimisation_problem, is_constant=True),
            get_all_partition_matrices(n, 2),
        ),
        key=partial(S, weights=weights),
    )
    basis = reduce(
        partial(_expand_basis_set, weights=weights, n=n), range(3, 1 + n), [u1, u2]
    )
    return np.column_stack(basis)


# --- Parallel version ---
def _argmin_parallel(n, k, solve_fn, score_fn):
    return min(get_all_solution_vectors_parallel(n, k, solve_fn), key=score_fn)


def _expand_basis_set_parallel(
    basis: List[np.ndarray], k: int, weights: np.ndarray, n: int
) -> List[np.ndarray]:
    """Computes the kth iteration of the l1 norm algorithm."""
    assert k >= 3
    U = np.column_stack(basis)
    solve_fn = partial(solve_minimisation_problem, U=U, is_constant=False)
    score_fn = partial(S, weights=weights)
    uk = _argmin_parallel(n, k, solve_fn, score_fn)
    return basis + [uk]


def _compute_parallel(n: int, weights: np.ndarray) -> np.ndarray:
    """Computes the l1 norm basis under the function S(x) in exponential time.

    Args:
        n (int): The number of vertices
        weights (List[List[int]]): The weights for a simple graph input for n vertices

    Returns:
        np.ndarray: An orthonormal basis
    """
    u1 = np.ones(n) / np.sqrt(n)
    solve_fn = partial(solve_minimisation_problem, U=None, is_constant=True)
    score_fn = partial(S, weights=weights)
    u2 = _argmin_parallel(n, 2, solve_fn, score_fn)
    basis = reduce(
        partial(_expand_basis_set_parallel, weights=weights, n=n),
        range(3, 1 + n),
        [u1, u2],
    )
    return np.column_stack(basis)
