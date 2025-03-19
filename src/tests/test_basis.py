import numpy as np
import pytest

from src.algorithms.greedy import compute_greedy_basis_undirected
from src.algorithms.l1_norm import compute_l1_norm_basis_undirected
from src.algorithms.laplacian import compute_laplacian_basis
from src.utils.verifications import is_orthonormal_basis
from src.utils.graphs import next_graph_input


# --- Tests ---
@pytest.mark.parametrize("file_name", ["small.txt"])
def _test_is_orthonormal_basis_laplacian(file_name):
    with open(f"public/input/{file_name}", "r") as f:
        num_tests = int(f.readline())
        assert all(
            is_orthonormal_basis(compute_laplacian_basis(*next_graph_input(f)))
            for _ in range(num_tests)
        )


@pytest.mark.parametrize("file_name", ["small.txt"])
def test_is_orthonormal_basis_greedy(file_name):
    with open(f"public/input/{file_name}", "r") as f:
        num_tests = int(f.readline())
        assert all(
            is_orthonormal_basis(compute_greedy_basis_undirected(*next_graph_input(f)))
            for _ in range(num_tests)
        )


@pytest.mark.parametrize("file_name", ["small.txt"])
def test_is_orthonormal_basis_l1_norm(file_name):
    with open(f"public/input/{file_name}", "r") as f:
        num_tests = int(f.readline())
        assert all(
            is_orthonormal_basis(compute_l1_norm_basis_undirected(*next_graph_input(f)))
            for _ in range(num_tests)
        )
