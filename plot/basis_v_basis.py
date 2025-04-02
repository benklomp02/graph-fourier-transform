import time

from plot.constants import *
from plot.utils.experiment_runner import *
from src.algorithms.greedy import compute_greedy_basis
from src.algorithms.laplacian import compute_laplacian_basis
from tests.utils.measurements import *
from tests.IO.graph import *


def _name(method):
    if hasattr(method, "__name__"):
        # remove the "compute_" prefix and the "basis" suffix
        method_name = method.__name__.replace("compute_", "").replace("_basis", "")
        return method_name
    else:
        return "unknown"


def plot_max_principal_angle(
    n_range,
    num_tests,
    compute_basis,
    compute_alt_basis,
    is_directed,
    save=True,
    show=False,
):
    start_time = time.time()
    print(f"Running max principal angle experiment with {num_tests} tests...")

    def metric_fn(n, weights, compute_basis_fn):
        U = compute_basis_fn(n, weights)
        V = compute_alt_basis(n, weights)
        return max_principal_angle(U, V)

    results = run_experiment(n_range, num_tests, compute_basis, metric_fn, is_directed)
    results_alt = run_experiment(
        n_range, num_tests, compute_alt_basis, metric_fn, is_directed
    )
    end_time = time.time()
    print(
        f"Max principal angle experiment completed in {end_time - start_time:.2f} seconds."
    )
    xname, yname = _name(compute_basis).title(), _name(compute_alt_basis).title()
    title = f"Max Principal Angle Comparison ({xname},{yname})"
    plot_and_save(
        title,
        list(n_range),
        [results, results_alt],
        [xname, yname],
        "N",
        "Angle (degrees)",
        num_tests,
        save,
        show,
        is_directed,
    )


def plot_projection_difference(
    n_range,
    num_tests,
    compute_basis,
    compute_alt_basis,
    is_directed,
    save=True,
    show=False,
):
    start_time = time.time()
    print(f"Running projection difference experiment with {num_tests} tests...")

    def metric_fn(n, weights, compute_basis_fn, compute_alt_basis_fn):
        U = compute_basis_fn(n, weights)
        V = compute_alt_basis_fn(n, weights)
        return projection_difference(U, V)

    results = run_experiment_pair(
        n_range, num_tests, compute_basis, compute_alt_basis, metric_fn, is_directed
    )
    end_time = time.time()
    print(
        f"Projection difference experiment completed in {end_time - start_time:.2f} seconds."
    )
    xname, yname = _name(compute_basis).title(), _name(compute_alt_basis).title()
    title = f"Projection Difference Comparison ({xname},{yname})"
    plot_and_save(
        title,
        list(n_range),
        [results],
        ["Projection Difference"],
        "N",
        "Relative Frobenius norm",
        num_tests,
        save,
        show,
        is_directed,
    )


def plot_spectral_energy(
    n_range,
    num_tests,
    compute_basis,
    compute_alt_basis,
    signal_fn,
    is_directed,
    save=True,
    show=False,
):
    start_time = time.time()
    print(f"Running spectral energy experiment with {num_tests} tests...")

    def metric_fn(n, weights, compute_basis_fn):
        U = compute_basis_fn(n, weights)
        X = signal_fn(n, n)
        return graph_spectral_energy_ratio(U, X)

    results_basis = run_experiment(
        n_range, num_tests, compute_basis, metric_fn, is_directed
    )
    results_alt_basis = run_experiment(
        n_range, num_tests, compute_alt_basis, metric_fn, is_directed
    )
    end_time = time.time()
    print(
        f"Spectral energy experiment completed in {end_time - start_time:.2f} seconds."
    )
    xname, yname = _name(compute_basis).title(), _name(compute_alt_basis).title()
    title = f"Spectral Energy Comparison ({xname},{yname})"
    plot_and_save(
        title,
        list(n_range),
        [results_basis, results_alt_basis],
        [xname, yname],
        "N",
        "Energy (||UᵀX||²)",
        num_tests,
        save,
        show,
        is_directed,
    )


def plot_procrustes_error(
    n_range,
    num_tests,
    compute_basis,
    compute_alt_basis,
    is_directed,
    save=True,
    show=False,
):
    start_time = time.time()
    print(f"Running Procrustes alignment error experiment with {num_tests} tests...")

    def metric_fn(n, weights, compute_basis_fn, compute_alt_basis_fn):
        U = compute_basis_fn(n, weights)
        V = compute_alt_basis_fn(n, weights)
        return procrustes_alignment_error(U, V)

    results = run_experiment_pair(
        n_range, num_tests, compute_basis, compute_alt_basis, metric_fn, is_directed
    )

    end_time = time.time()
    print(f"Procrustes experiment completed in {end_time - start_time:.2f} seconds.")

    xname, yname = _name(compute_basis).title(), _name(compute_alt_basis).title()
    title = f"Procrustes Alignment Error ({xname},{yname})"
    plot_and_save(
        title,
        list(n_range),
        [results],
        ["Procrustes Error"],
        "N",
        "Relative Frobenius norm",
        num_tests,
        save,
        show,
        is_directed,
    )


def plot_laplacian_diag_error(
    n_range,
    num_tests,
    compute_basis,
    compute_alt_basis,
    is_directed,
    save=True,
    show=False,
):
    start_time = time.time()
    print(
        f"Running Laplacian diagonalization error experiment with {num_tests} tests..."
    )

    def metric_fn(n, weights, compute_basis_fn):
        U = compute_basis_fn(n, weights)
        return laplacian_diagonalization_error(U, weights)

    results = run_experiment(n_range, num_tests, compute_basis, metric_fn, is_directed)
    results_alt = run_experiment(
        n_range, num_tests, compute_alt_basis, metric_fn, is_directed
    )

    end_time = time.time()
    print(
        f"Laplacian diag error experiment completed in {end_time - start_time:.2f} seconds."
    )

    xname, yname = _name(compute_basis).title(), _name(compute_alt_basis).title()
    title = f"Laplacian Diagonalization Error ({xname},{yname})"
    plot_and_save(
        title,
        list(n_range),
        [results, results_alt],
        [xname, yname],
        "N",
        "Relative Frobenius norm",
        num_tests,
        save,
        show,
        is_directed,
    )


def plot_relative_energy_error(
    n_range,
    num_tests,
    compute_basis,
    compute_alt_basis,
    is_directed,
    save=True,
    show=False,
):
    start_time = time.time()
    print(f"Running relative energy error experiment with {num_tests} tests...")

    def metric_fn(n, weights, compute_basis_fn):
        U = compute_basis_fn(n, weights)
        V = compute_alt_basis(n, weights)
        return relative_error(U, V, weights)

    results = run_experiment(n_range, num_tests, compute_basis, metric_fn, is_directed)
    end_time = time.time()
    print(
        f"Relative energy error experiment completed in {end_time - start_time:.2f} seconds."
    )

    xname, yname = _name(compute_basis).title(), _name(compute_alt_basis).title()
    title = f"Relative Energy Error ({xname},{yname})"
    plot_and_save(
        title,
        list(n_range),
        [results],
        ["Relative Error"],
        "N",
        "Relative Error",
        num_tests,
        save,
        show,
        is_directed,
    )


def main():
    n_range = [4, 8, 20, 30, 40, 50, 60]
    num_tests = NUM_TESTS
    is_directed = IS_DIRECTED
    save = SAVE_FIG
    show = SHOW_PLOT

    # Define basis computation methods
    compute_basis = compute_greedy_basis
    compute_alt_basis = compute_laplacian_basis

    plot_max_principal_angle(
        n_range, num_tests, compute_basis, compute_alt_basis, is_directed, save, show
    )
    plot_projection_difference(
        n_range, num_tests, compute_basis, compute_alt_basis, is_directed, save, show
    )
    plot_spectral_energy(
        n_range,
        num_tests,
        compute_basis,
        compute_alt_basis,
        random_signal_matrix,
        is_directed,
        save,
        show,
    )
    plot_procrustes_error(
        n_range, num_tests, compute_basis, compute_alt_basis, is_directed, save, show
    )
    plot_laplacian_diag_error(
        n_range, num_tests, compute_basis, compute_alt_basis, is_directed, save, show
    )
    plot_relative_energy_error(
        n_range, num_tests, compute_basis, compute_alt_basis, is_directed, save, show
    )


if __name__ == "__main__":
    main()
