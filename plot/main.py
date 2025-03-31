from typing import Callable, List
from statistics import mean
import time

from plot.constants import *

from plot.utils.experiment_runner import *
from src.algorithms.greedy import compute_greedy_basis
from src.algorithms.l1_norm import compute_l1_norm_basis
from src.algorithms.laplacian import compute_laplacian_basis
from tests.utils.measurements import *
from tests.utils.approx import *


def comparison_of_variation_single_vector(k: int = 2):
    """
    WARNING: This function is computationally intensive.

    This function compares the variation of the greedy and Laplacian basis
    for a single vector.

    Args:
        k (int, optional): The chosen vector indexed by 1. Defaults to 2.
    """
    print(f"Starting comparison of variation on single vector k={k}.")
    start_time = time.time()
    assert 3 <= MIN_N <= MAX_N and 0 <= k < MIN_N
    x_axis = range(MIN_N, MAX_N + 1)

    metric_fn = lambda n, weights, basis: relative_error_single_vector(
        basis(n, weights)[:, (k - 1)],
        compute_l1_norm_basis(n, weights)[:, (k - 1)],
        weights,
    )

    greedy_errors = run_experiment(
        x_axis, NUM_TESTS, compute_greedy_basis, metric_fn, IS_DIRECTED
    )
    laplacian_errors = run_experiment(
        x_axis, NUM_TESTS, compute_laplacian_basis, metric_fn, IS_DIRECTED
    )

    print(
        f"Finished comparison of variation on single vector k={k} in {time.time()-start_time:.2f} seconds."
    )
    plot_title = f"Comparison of variation on single vector k={k}"
    plot_and_save(
        plot_title,
        x_axis,
        [greedy_errors, laplacian_errors],
        ["Greedy", "Laplacian"],
        "N",
        "Average relative error",
        num_tests=NUM_TESTS,
        save_fig=SAVE_FIG,
        show_plot=SHOW_PLOT,
        is_directed=IS_DIRECTED,
    )


def comparison_of_variation():
    """
    WARNING: This function is computationally intensive.

    This function compares the variation of the greedy and Laplacian basis
    for all vectors.
    """
    assert 3 <= MIN_N <= MAX_N
    print("Starting comparison of variation.")
    start_time = time.time()
    x_axis = range(MIN_N, MAX_N + 1)

    metric_fn = lambda n, weights, basis: relative_error(
        basis(n, weights), compute_l1_norm_basis(n, weights), weights
    )

    greedy_errors = run_experiment(
        x_axis, NUM_TESTS, compute_greedy_basis, metric_fn, IS_DIRECTED
    )
    laplacian_errors = run_experiment(
        x_axis, NUM_TESTS, compute_laplacian_basis, metric_fn, IS_DIRECTED
    )
    print(f"Finished comparison of variation in {time.time()-start_time:.2f} seconds.")
    plot_title = "Comparison of variation"
    plot_and_save(
        plot_title,
        x_axis,
        [greedy_errors, laplacian_errors],
        ["Greedy", "Laplacian"],
        "N",
        "Average relative error",
        num_tests=NUM_TESTS,
        save_fig=NUM_TESTS,
        show_plot=SHOW_PLOT,
        is_directed=IS_DIRECTED,
    )


DEFAULT_FUNCTIONS = [
    compute_l1_norm_basis,
    compute_greedy_basis,
    compute_laplacian_basis,
]

DEFAULT_LABELS = ["L1 Norm", "Greedy", "Laplacian"]


def comparison_of_time(
    xfunc: List[Callable] = DEFAULT_FUNCTIONS,
    series_labels: List[str] = DEFAULT_LABELS,
):
    """
    WARNING: This function may be computationally intensive depending on the provided functions.

    Args:
        xfunc (List[Callable], optional): A series of functions for comparison.
        series_labels (List[str], optional): A series of labels for the provided functions.
    """
    print("Starting comparison of time.")
    start_time = time.time()
    assert 3 <= MIN_N <= MAX_N
    x_axis = range(MIN_N, MAX_N + 1)

    time_metric = lambda f, basis: average_time(f, basis)
    series = [
        run_experiment_file(x_axis, NUM_TESTS, basis, time_metric, IS_DIRECTED)
        for basis in xfunc
    ]
    print(f"Finished comparison of time in {time.time()-start_time:.2f} seconds.")
    plot_title = "Comparison of time"
    plot_and_save(
        plot_title,
        x_axis,
        series,
        series_labels,
        "N",
        "Average time",
        num_tests=NUM_TESTS,
        save_fig=SAVE_FIG,
        show_plot=SHOW_PLOT,
        is_directed=IS_DIRECTED,
    )


def comparison_of_nterm_approx():
    """
    This function compares the n-term approximation of the greedy and Laplacian basis.
    """
    print("Starting comparison of n-term approximation.")
    start_time = time.time()

    x_axis = [3, 4, 5, 6, 7, 8, 20, 30, 40, 50, 60]

    def metric_fn(n, weights, compute_basis):
        # Compute a random laplacian signal
        signal = compute_random_laplacian_signal(n, weights)
        # Compute the approximated error using the given basis function
        error = compute_nterm_error(n, signal, weights, compute_basis)
        return error

    greedy_errors = run_experiment(
        x_axis,
        NUM_TESTS,
        compute_greedy_basis,
        metric_fn,
        IS_DIRECTED,
    )

    laplacian_errors = run_experiment(
        x_axis, NUM_TESTS, compute_laplacian_basis, metric_fn, IS_DIRECTED
    )
    print(
        f"Finished comparison of n-term approximation in {time.time()-start_time:.2f} seconds."
    )
    plot_title = "Comparison of n-term approximation"
    plot_and_save(
        plot_title,
        x_axis,
        [greedy_errors, laplacian_errors],
        ["Greedy", "Laplacian"],
        "n",
        "Average relative error",
        num_tests=NUM_TESTS,
        save_fig=SAVE_FIG,
        show_plot=SHOW_PLOT,
        is_directed=IS_DIRECTED,
    )


def run_all():
    comparison_of_variation_single_vector()
    comparison_of_variation()
    comparison_of_time()
    comparison_of_nterm_approx()


if __name__ == "__main__":
    comparison_of_nterm_approx()
