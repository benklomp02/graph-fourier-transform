import time

from src.algorithms.greedy import compute_greedy_basis
from src.algorithms.laplacian import compute_laplacian_basis
from plot.utils.experiment_runner import *
from plot.constants import *
from tests.utils.measurements import *
from tests.utils.approx import *


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
    
if __name__ == "__main__":
    comparison_of_nterm_approx()