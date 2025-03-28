import pytest

from src.algorithms.laplacian import compute_laplacian_basis
from src.algorithms.greedy import compute_greedy_basis
from src.algorithms.l1_norm import compute_l1_norm_basis
from tests.utils.verifications import is_orthonormal_basis
from tests.IO.graph import next_graph_input

test_files_sm = [
    "public/input/undirected/input_Nmax6_t100.txt",
    "public/input/directed/input_Nmax6_t100.txt",
]
test_files = [
    "public/input/undirected/input_Nmax6_t100.txt",
    "public/input/undirected/input_Nmax20_t100.txt",
    "public/input/directed/input_Nmax6_t100.txt",
    "public/input/directed/input_Nmax20_t100.txt",
]


# --- Tests ---
@pytest.mark.parametrize(
    "file_path",
    test_files_sm,
)
def test_l1_norm(file_path):
    with open(file_path, "r") as f:
        num_tests = int(f.readline())
        assert all(
            is_orthonormal_basis(compute_l1_norm_basis(*next_graph_input(f)))
            for _ in range(num_tests)
        )


@pytest.mark.parametrize("file_path", test_files)
def test_laplacian(file_path):
    with open(file_path, "r") as f:
        num_tests = int(f.readline())
        assert all(
            is_orthonormal_basis(compute_laplacian_basis(*next_graph_input(f)))
            for _ in range(num_tests)
        )


@pytest.mark.parametrize("file_path", test_files)
def test_greedy(file_path):
    with open(file_path, "r") as f:
        num_tests = int(f.readline())
        assert all(
            is_orthonormal_basis(compute_greedy_basis(*next_graph_input(f)))
            for _ in range(num_tests)
        )
