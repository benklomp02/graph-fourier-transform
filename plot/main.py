from typing import Callable, List
from statistics import mean
import time

from plot.constants import *

from plot.utils.experiment_runner import (
    run_experiment,
    run_experiment_file,
    plot_and_save,
)
from src.algorithms.greedy import compute_greedy_basis
from src.algorithms.l1_norm import compute_l1_norm_basis
from src.algorithms.laplacian import compute_laplacian_basis
from tests.utils.measurements import (
    relative_error,
    relative_error_single_vector,
    average_time,
)
from tests.utils.approx import (
    compute_nterm_error,
    compute_random_laplacian_signal,
    approx_error,
)


def comparison_of_variation_single_vector(k: int = 2):
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


def comparison_of_time(
    xfunc: List[Callable] = [
        compute_l1_norm_basis,
        compute_greedy_basis,
        compute_laplacian_basis,
    ],
    series_labels: List[str] = ["L1 Norm", "Greedy", "Laplacian"],
):
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
    assert 3 <= MIN_N <= MAX_N
    print("Starting comparison of n-term approximation.")
    start_time = time.time()

    x_axis = range(MIN_N, MAX_N + 1)

    def metric_fn(n, weights, compute_basis):
        signal = compute_random_laplacian_signal(n, weights)
        errors = []
        for n_approx in range(1, n + 1):
            y_n = compute_nterm_error(n_approx, signal, weights, compute_basis)
            errors.append(approx_error(signal, y_n))
        return mean(errors)

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
    comparison_of_variation_single_vector()
