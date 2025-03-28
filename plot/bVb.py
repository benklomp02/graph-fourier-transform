import time

from plot.constants import *
from plot.utils.experiment_runner import run_experiment, plot_and_save
from src.algorithms.greedy import compute_greedy_basis
from src.algorithms.laplacian import compute_laplacian_basis
from tests.utils.measurements import relative_error

# This script compares the performance of two algorithms for computing a basis of a graph.


def compare_algorithms(
    xcompute_basis,
    ycompute_basis,
):
    """
    Compare the performance of two base algorithms.
    """
    print("Starting comparison of algorithms.")
    start_time = time.time()
    assert 3 <= MIN_N <= MAX_N
    x_axis = range(MIN_N, MAX_N + 1)

    metric_fn = lambda n, weights, basis: relative_error(
        basis(n, weights), compute_laplacian_basis(n, weights), weights
    )

    xerrors = run_experiment(
        x_axis, NUM_TESTS, xcompute_basis, metric_fn, is_directed=IS_DIRECTED
    )
    yerrors = run_experiment(
        x_axis, NUM_TESTS, ycompute_basis, metric_fn, is_directed=IS_DIRECTED
    )

    print(f"Finished comparison in {time.time()-start_time:.2f} seconds.")
    plot_title = "Comparison of Algorithms"
    plot_and_save(
        plot_title,
        x_axis,
        [xerrors, yerrors],
        [xcompute_basis.__name__, ycompute_basis.__name__],
        "N",
        "Average Relative Error",
        num_tests=NUM_TESTS,
        save_fig=SAVE_FIG,
        show_plot=SHOW_PLOT,
    )


if __name__ == "__main__":

    compare_algorithms(
        compute_greedy_basis,
        compute_laplacian_basis,
    )
