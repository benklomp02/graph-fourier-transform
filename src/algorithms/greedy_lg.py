import numpy as np
from itertools import combinations
from functools import partial

from tests.IO.examples import comet


def _objective(i, j, tau, memo):
    return memo[i, j] / (len(tau[i]) * len(tau[j]))


def compute_greedy_basis_lg(n: int, weights: np.ndarray):
    tau = [{i} for i in range(n)]
    memo = weights.copy()
    basis = []
    for _ in range(n - 1):
        # Find the pair of sets with the maximum objective function
        i, j = max(
            [(i, j) for i in range(len(tau)) for j in range(len(tau)) if i != j],
            key=lambda x: _objective(x[0], x[1], tau, memo),
        )
        # Compute the orthonormal basis vector
        a, b = np.zeros(n), np.zeros(n)
        a[list(tau[i])] = 1
        b[list(tau[j])] = 1
        t = 1 / np.sqrt(len(tau[i]) * len(tau[j]) * (len(tau[i]) + len(tau[j])))
        u = -t * len(tau[j]) * a + t * len(tau[i]) * b
        basis.append(u)
        # Prepare for the next iteration
        tau[i] |= tau[j]
        tau.pop(j)
        memo[i] += memo[j]
        memo[i][j] = 0
        mask = np.arange(memo.shape[0]) != j
        memo = memo[mask][:, mask]
    u1 = np.ones(n) / np.sqrt(n)
    basis.append(u1)
    return np.column_stack(basis[::-1])
