import numpy as np
from itertools import combinations
from collections import deque


def is_normalized(x: np.ndarray) -> bool:
    return np.isclose(np.linalg.norm(x), 1)


def is_orthogonal(A: np.ndarray, x: np.ndarray) -> bool:
    return np.allclose(A @ x, 0)


def is_orthogonal_basis(A: np.ndarray) -> bool:
    return np.allclose(A.T @ A, np.eye(A.shape[1]))


def is_normalized_basis(A: np.ndarray) -> bool:
    return np.allclose(np.linalg.norm(A, axis=0), 1)


def is_orthonormal_basis(A: np.ndarray) -> bool:
    return is_orthogonal_basis(A) and is_normalized_basis(A)
