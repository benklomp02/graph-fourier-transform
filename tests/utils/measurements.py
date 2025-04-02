import time
from statistics import mean
import numpy as np
from scipy.linalg import subspace_angles, orthogonal_procrustes

from tests.IO.graph import read_graph_input
from src.utils.c_objectives import S

from src.algorithms.laplacian import laplacian_matrix


# --- Measures of performance ---
def average_time(f, compute_basis):
    num_tests = int(f.readline())

    def compute_and_stop():
        n, weights = read_graph_input(f)
        start = time.time()
        compute_basis(n, weights)
        return time.time() - start

    return mean(compute_and_stop() for _ in range(num_tests))


# --- Relative measures of quality---
def relative_error_single_vector(
    basis: np.ndarray, alt_basis: np.ndarray, weights: np.ndarray
) -> float:
    n = basis.shape[0]
    assert basis.shape == alt_basis.shape == (n,)
    return (S(basis, weights) - (s_alt := S(alt_basis, weights))) / s_alt


def relative_error(
    basis: np.ndarray,
    alt_basis: np.ndarray,
    weights: np.ndarray,
) -> float:
    assert basis.shape == alt_basis.shape
    s_alt = sum(S(u, weights) for u in alt_basis.T)
    return (sum(S(u, weights) for u in basis.T) - s_alt) / (s_alt + 1e-10)


def projection_difference(basis: np.ndarray, alt_basis: np.ndarray) -> float:
    """
    Compute the projection difference between two orthonormal bases.
    """
    assert basis.shape == alt_basis.shape
    assert basis.shape[0] == basis.shape[1]
    # Compute the projection matrices
    P = basis @ basis.T
    Q = alt_basis @ alt_basis.T
    return np.linalg.norm(P - Q, ord="fro") / np.linalg.norm(Q, ord="fro")


def procrustes_alignment_error(basis: np.ndarray, alt_basis: np.ndarray):
    """
    Compute the Procrustes alignment error between two orthonormal bases.
    """
    assert basis.shape == alt_basis.shape
    assert basis.shape[0] == basis.shape[1]
    # Compute the Procrustes alignment
    R, _ = orthogonal_procrustes(basis, alt_basis)
    aligned_basis = alt_basis @ R
    return np.linalg.norm(basis - aligned_basis, "fro") / np.linalg.norm(basis, "fro")


# --- Absolute measures of quality ---


def max_principal_angle(basis: np.ndarray, alt_basis: np.ndarray) -> float:
    """
    Compute the principal angles (in degrees) between two orthonormal bases.
    """
    assert basis.shape == alt_basis.shape
    assert basis.shape[0] == basis.shape[1]
    angles = subspace_angles(basis, alt_basis)
    return np.rad2deg(np.max(angles))


def laplacian_diagonalization_error(basis: np.ndarray, weights: np.ndarray):
    L = laplacian_matrix(weights)
    M = basis.T @ L @ basis
    return np.linalg.norm(M - np.diag(np.diag(M)), ord="fro") / np.linalg.norm(
        M, ord="fro"
    )


def graph_smoothness(basis: np.ndarray, weights: np.ndarray):
    """
    Compute the graph smoothness trace(U^T * U) of a basis U.
    """
    assert basis.shape[0] == basis.shape[1]
    L = laplacian_matrix(weights)
    return np.trace(basis.T @ L @ basis)


def graph_spectral_energy_ratio(basis: np.ndarray, X: np.ndarray):
    """
    Compute the graph spectral energy of a basis.
    """
    assert basis.shape[0] == basis.shape[1]
    assert basis.shape[0] == X.shape[0]
    denom = np.linalg.norm(X, ord="fro") ** 2
    return np.linalg.norm(basis.T @ X, ord="fro") ** 2 / (denom + 1e-10)
