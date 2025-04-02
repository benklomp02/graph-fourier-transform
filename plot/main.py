from typing import Callable, List
import time

from plot.constants import *

from plot.utils.experiment_runner import *
from src.algorithms.greedy import compute_greedy_basis
from src.algorithms.l1_norm_cpp import compute_l1_norm_basis_cpp
from src.algorithms.laplacian import compute_laplacian_basis
from tests.utils.measurements import *
from tests.utils.approx import *


def comparison_of_variation(
    num_tests: int = NUM_TESTS,
    is_directed: bool = IS_DIRECTED,
    save_fig: bool = SAVE_FIG,
    show_plot: bool = SHOW_PLOT,
    min_n: int = MIN_N,
    max_n: int = MAX_N,
):
    """
    WARNING: This function is computationally intensive.

    This function compares the variation of the greedy and Laplacian basis
    for all vectors.
    """
    assert 3 <= min_n <= max_n
    print("Starting comparison of variation.")
    start_time = time.time()
    x_axis = range(min_n, max_n + 1)

    def metric_fn(n, weights, compute_basis):
        basis = compute_basis(n, weights)
        return relative_total_variation(n, basis, weights)

    greedy_errors = run_experiment(
        x_axis, num_tests, compute_greedy_basis, metric_fn, is_directed
    )
    exact_errors = run_experiment(
        x_axis, num_tests, compute_l1_norm_basis_cpp, metric_fn, is_directed
    )
    print(f"Finished comparison of variation in {time.time()-start_time:.2f} seconds.")
    plot_title = "Comparison of variation"
    plot_and_save(
        plot_title,
        x_axis,
        [greedy_errors, exact_errors],
        ["Greedy", "L1 Norm"],
        "N",
        "Relative error compared to N",
        num_tests=num_tests,
        save_fig=save_fig,
        show_plot=show_plot,
        is_directed=is_directed,
    )


DEFAULT_FUNCTIONS = [
    compute_l1_norm_basis_cpp,
    compute_greedy_basis,
    compute_laplacian_basis,
]

DEFAULT_LABELS = ["L1 Norm", "Greedy", "Laplacian"]


def comparison_of_time(
    xfunc: List[Callable] = DEFAULT_FUNCTIONS,
    series_labels: List[str] = DEFAULT_LABELS,
    num_tests: int = NUM_TESTS,
    is_directed: bool = IS_DIRECTED,
    save_fig: bool = SAVE_FIG,
    show_plot: bool = SHOW_PLOT,
    min_n: int = MIN_N,
    max_n: int = MAX_N,
):
    """
    WARNING: This function may be computationally intensive depending on the provided functions.

    Args:
        xfunc (List[Callable], optional): A series of functions for comparison.
        series_labels (List[str], optional): A series of labels for the provided functions.
    """
    print("Starting comparison of time.")
    start_time = time.time()
    assert 3 <= min_n <= max_n
    x_axis = range(min_n, max_n + 1)

    time_metric = lambda f, basis: average_time(f, basis)
    series = [
        run_experiment_file(x_axis, num_tests, basis, time_metric, is_directed)
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
        num_tests=num_tests,
        save_fig=save_fig,
        show_plot=show_plot,
        is_directed=is_directed,
    )


def main():
    comparison_of_variation(
        num_tests=20,
        is_directed=False,
        save_fig=True,
        show_plot=False,
        min_n=3,
        max_n=8,
    )
    comparison_of_variation(
        num_tests=20,
        is_directed=True,
        save_fig=True,
        show_plot=False,
        min_n=3,
        max_n=8,
    )
    comparison_of_time(
        xfunc=DEFAULT_FUNCTIONS,
        series_labels=DEFAULT_LABELS,
        num_tests=20,
        is_directed=False,
        save_fig=True,
        show_plot=False,
        min_n=3,
        max_n=8,
    )
    comparison_of_time(
        xfunc=DEFAULT_FUNCTIONS,
        series_labels=DEFAULT_LABELS,
        num_tests=20,
        is_directed=True,
        save_fig=True,
        show_plot=False,
        min_n=3,
        max_n=8,
    )


if __name__ == "__main__":
    main()
