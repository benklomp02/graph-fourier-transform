import numpy as np
import networkx as nx
from scipy.linalg import eigh
from line_profiler import profile


def laplacian_matrix(weights: np.ndarray, use_in_degrees: bool = False) -> np.ndarray:
    """Computes the Laplacian matrix of a graph."""
    DiG = nx.from_numpy_array(np.array(weights), create_using=nx.DiGraph)
    if use_in_degrees:
        return nx.laplacian_matrix(DiG.reverse(copy=False)).toarray().T
    return nx.laplacian_matrix(DiG).toarray()


def compute_laplacian_basis(
    n: int, weights: np.ndarray, use_in_degrees: bool = False
) -> np.ndarray:
    """Computes the Laplacian basis of a graph."""
    return _compute(n, weights, use_in_degrees)


def _compute(n, weights, use_in_degrees):
    laplacian = laplacian_matrix(weights, use_in_degrees)
    _, eigenvectors = eigh(laplacian)
    return eigenvectors
