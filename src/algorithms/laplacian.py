import numpy as np
from typing import List

from src.utils.verifications import is_orthonormal_basis
from src.utils.graphs import comet


def laplacian_matrix(n: int, weights: List[List[int]]):
    degree_matrix = np.zeros((n, n))
    for i in range(n):
        degree_matrix[i][i] = sum(weights[i])
    return degree_matrix - np.array(weights)


def compute_laplacian_basis(n: int, weights: List[List[int]]):
    return np.linalg.eig(laplacian_matrix(n, weights)).eigenvectors


def run_example(n, weights):
    basis = compute_laplacian_basis(n, weights)
    np.set_printoptions(precision=3, suppress=True, linewidth=100)
    print("laplacian basis:")
    print(basis)
    assert is_orthonormal_basis(basis)


if __name__ == "__main__":
    run_example(*comet(6))
