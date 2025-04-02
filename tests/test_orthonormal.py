import pytest
from itertools import chain

from src.algorithms.laplacian import compute_laplacian_basis
from src.algorithms.greedy import compute_greedy_basis
from src.algorithms.greedy_lg import compute_greedy_basis_lg
from src.algorithms.l1_norm import compute_l1_norm_basis
from src.algorithms.l1_norm_cpp import compute_l1_norm_basis_cpp
from tests.utils.verifications import is_orthonormal_basis
from tests.IO.graph import read_graph_input

PREFIX_PATH = "public/input"
test_files_sm = [
    f"{PREFIX_PATH}/undirected/Nmax6_t100.txt",
    f"{PREFIX_PATH}/directed/Nmax6_t100.txt",
]
test_files = [
    f"{PREFIX_PATH}/undirected/Nmax6_t100.txt",
    f"{PREFIX_PATH}/undirected/Nmax20_t100.txt",
    f"{PREFIX_PATH}/directed/Nmax6_t100.txt",
    f"{PREFIX_PATH}/directed/Nmax20_t100.txt",
]

test_files_lg = [
    f"{PREFIX_PATH}/undirected/Nmax40_t100.txt",
    f"{PREFIX_PATH}/directed/Nmax40_t100.txt",
    f"{PREFIX_PATH}/undirected/Nmax60_t100.txt",
    f"{PREFIX_PATH}/directed/Nmax60_t100.txt",
]

test_files_xlg = [
    f"{PREFIX_PATH}/undirected/N1024_t1.txt",
    f"{PREFIX_PATH}/directed/N1024_t1.txt",
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
            is_orthonormal_basis(compute_l1_norm_basis(*read_graph_input(f)))
            for _ in range(num_tests)
        )


@pytest.mark.parametrize(
    "file_path",
    test_files_sm,
)
def test_l1_norm_cpp(file_path):
    with open(file_path, "r") as f:
        num_tests = int(f.readline())
        assert all(
            is_orthonormal_basis(compute_l1_norm_basis_cpp(*read_graph_input(f)))
            for _ in range(num_tests)
        )


@pytest.mark.parametrize("file_path", test_files + test_files_lg)
def test_laplacian(file_path):
    with open(file_path, "r") as f:
        num_tests = int(f.readline())
        assert all(
            is_orthonormal_basis(compute_laplacian_basis(*read_graph_input(f)))
            for _ in range(num_tests)
        )


@pytest.mark.parametrize("file_path", test_files + test_files_lg)
def test_greedy(file_path):
    with open(file_path, "r") as f:
        num_tests = int(f.readline())
        assert all(
            is_orthonormal_basis(compute_greedy_basis(*read_graph_input(f)))
            for _ in range(num_tests)
        )


@pytest.mark.parametrize(
    "file_path",
    test_files + test_files_lg,
)
def test_greedy_python(file_path):
    with open(file_path, "r") as f:
        num_tests = int(f.readline())
        assert all(
            is_orthonormal_basis(compute_greedy_basis_lg(*read_graph_input(f)))
            for _ in range(num_tests)
        )


@pytest.mark.parametrize(
    "file_path",
    test_files_xlg,
)
def test_greedy_python_xlg(file_path):
    with open(file_path, "r") as f:
        num_tests = int(f.readline())
        assert all(
            is_orthonormal_basis(compute_greedy_basis_lg(*read_graph_input(f)))
            for _ in range(num_tests)
        )
