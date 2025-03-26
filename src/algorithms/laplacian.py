import numpy as np
import networkx as nx
from scipy.linalg import eigh

from tests.utils.verifications import is_orthonormal_basis
from tests.IO.examples import comet


def laplacian_matrix(n: int, weights: np.ndarray):
    g = nx.from_numpy_array(np.array(weights), create_using=nx.DiGraph)
    return nx.laplacian_matrix(g).toarray()


def compute_laplacian_basis(n: int, weights: np.ndarray):
    laplacian = laplacian_matrix(n, weights)
    _, eigenvectors = eigh(laplacian)
    return eigenvectors


def run_example(n, weights):
    basis = compute_laplacian_basis(n, weights)
    np.set_printoptions(precision=3, suppress=True, linewidth=100)
    print("laplacian basis:")
    print(basis)
    assert is_orthonormal_basis(basis)


if __name__ == "__main__":
    run_example(*comet(6))
