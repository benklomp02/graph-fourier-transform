import pytest

from src.algorithms.laplacian import compute_laplacian_basis
from src.algorithms.greedy import compute_greedy_basis
from src.algorithms.l1_norm import compute_l1_norm_basis
from tests.utils.verifications import is_orthonormal_basis
from tests.IO.graph_generator import next_graph_input


# --- Tests ---
@pytest.mark.parametrize("file_path", ["public/input/directed/input_Nmax6_t100.txt"])
def test_laplacian_directed_small(file_path):
    with open(file_path, "r") as f:
        num_tests = int(f.readline())
        assert all(
            is_orthonormal_basis(compute_laplacian_basis(*next_graph_input(f)))
            for _ in range(num_tests)
        )


@pytest.mark.parametrize("file_path", ["public/input/directed/input_Nmax20_t100.txt"])
def test_laplacian_directed_large(file_path):
    with open(file_path, "r") as f:
        num_tests = int(f.readline())
        assert all(
            is_orthonormal_basis(compute_laplacian_basis(*next_graph_input(f)))
            for _ in range(num_tests)
        )


@pytest.mark.parametrize("file_path", ["public/input/directed/input_Nmax6_t100.txt"])
def test_greedy_directed_small(file_path):
    with open(file_path, "r") as f:
        num_tests = int(f.readline())
        assert all(
            is_orthonormal_basis(
                compute_greedy_basis(*next_graph_input(f), is_directed=True)
            )
            for _ in range(num_tests)
        )


@pytest.mark.parametrize("file_path", ["public/input/directed/input_Nmax20_t100.txt"])
def test_greedy_directed_large(file_path):
    with open(file_path, "r") as f:
        num_tests = int(f.readline())
        assert all(
            is_orthonormal_basis(
                compute_greedy_basis(*next_graph_input(f), is_directed=True)
            )
            for _ in range(num_tests)
        )


@pytest.mark.parametrize("file_path", ["public/input/directed/input_Nmax6_t100.txt"])
def test_l1_norm_directed_small(file_path):
    with open(file_path, "r") as f:
        num_tests = int(f.readline())
        assert all(
            is_orthonormal_basis(compute_l1_norm_basis(*next_graph_input(f)))
            for _ in range(num_tests)
        )
